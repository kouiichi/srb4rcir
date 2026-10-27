"""Fixed and limited kinematic research camera mounts for DESDynI."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from srb.nmpc.quaternion import quaternion_to_rotation


def onboard_camera_spec(actor_bottom, resolution_wh, horizontal_fov_deg):
    """Look along body -Z using Isaac's +X-forward camera convention."""
    width, height = resolution_wh
    aperture = 20.955
    return {
        'position_B_m': (np.asarray(actor_bottom)+[0.,0.,-.02]).tolist(),
        'quaternion_BC_wxyz': [np.sqrt(.5),0.,np.sqrt(.5),0.],
        'orientation_convention': 'world',
        'resolution_wh': [int(width),int(height)],
        'horizontal_fov_deg': float(horizontal_fov_deg),
        'horizontal_aperture': aperture,
        'vertical_aperture': aperture*height/width,
        'focal_length': aperture/(2*np.tan(np.deg2rad(horizontal_fov_deg)/2)),
        'clipping_range_m': [.01,1000.],
        'sensor_model': 'virtual pinhole RGB/depth camera; fixed mount, no hardware mass or measurement noise',
    }


def gimbal_camera_spec(actor_bottom, actor_points, resolution_wh, horizontal_fov_deg):
    """Place a virtual pan/tilt pivot beyond the +X and -Z mesh faces."""
    spec = onboard_camera_spec(actor_bottom, resolution_wh, horizontal_fov_deg)
    spec['position_B_m'] = [float(np.max(actor_points, axis=0)[0]+.02),
                            float(actor_bottom[1]), float(actor_bottom[2]-.02)]
    spec['quaternion_BC_wxyz'] = [float(np.cos(np.pi/8)), 0., float(np.sin(np.pi/8)), 0.]
    spec['sensor_model'] = 'virtual pinhole RGB/depth camera; limited kinematic pan/tilt, no hardware mass, motor dynamics or measurement noise'
    return spec


def camera_intrinsic(spec):
    width, height = spec['resolution_wh']
    focal = width/(2*np.tan(np.deg2rad(spec['horizontal_fov_deg'])/2))
    return np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])


class LimitedCameraGimbal:
    """Two independent limited joints; initialization points at the reset target.

    Yaw rotates about mount +Z. Positive pitch elevates camera +X. Tracking
    uses simulator target position and is separate from the vehicle NMPC.
    """

    def __init__(self, position_B, base_quaternion_BC,
                 yaw_limits_rad=(-2*np.pi/3, 2*np.pi/3),
                 pitch_limits_rad=(-4*np.pi/9, 4*np.pi/9),
                 max_rate_rad_s=np.pi/6):
        self.position_B = np.asarray(position_B, dtype=float).copy()
        self.base_rotation = quaternion_to_rotation(base_quaternion_BC)
        self.limits = np.asarray([yaw_limits_rad, pitch_limits_rad], dtype=float)
        self.max_rate = float(max_rate_rad_s)
        self.angles_rad = np.zeros(2)
        self.rates_rad_s = np.zeros(2)
        self.requested_rad = np.zeros(2)
        self.angle_limited = self.rate_limited = False

    @property
    def quaternion_BC(self):
        yaw, pitch = self.angles_rad
        rotation = self.base_rotation @ Rotation.from_euler('z', yaw).as_matrix() @ Rotation.from_euler('y', -pitch).as_matrix()
        xyzw = Rotation.from_matrix(rotation).as_quat()
        return xyzw[[3, 0, 1, 2]]

    def camera_pose(self, state, com_B):
        rotation = quaternion_to_rotation(state[6:10])
        position = state[:3]+rotation@(self.position_B-np.asarray(com_B))
        xyzw = Rotation.from_matrix(rotation@quaternion_to_rotation(self.quaternion_BC)).as_quat()
        return position, xyzw[[3, 0, 1, 2]]

    def _command(self, state, com_B, target_position):
        position, _ = self.camera_pose(state, com_B)
        bearing = self.base_rotation.T@quaternion_to_rotation(state[6:10]).T@(np.asarray(target_position)-position)
        self.requested_rad = np.array([np.arctan2(bearing[1], bearing[0]),
                                       np.arctan2(bearing[2], np.hypot(bearing[0], bearing[1]))])
        command = np.clip(self.requested_rad, self.limits[:, 0], self.limits[:, 1])
        self.angle_limited = bool(np.any(np.abs(command-self.requested_rad)>1e-12))
        return command

    def initialize(self, state, com_B, target_position):
        self.angles_rad = self._command(state, com_B, target_position)
        self.rates_rad_s[:] = 0.
        self.rate_limited = False

    def advance(self, state, com_B, target_position, dt):
        command = self._command(state, com_B, target_position)
        delta = command-self.angles_rad
        applied = np.clip(delta, -self.max_rate*dt, self.max_rate*dt)
        self.rate_limited = bool(np.any(np.abs(applied-delta)>1e-12))
        self.rates_rad_s = applied/dt
        self.angles_rad = self.angles_rad+applied

    def snapshot(self):
        return {'angles_rad': self.angles_rad.copy(), 'rates_rad_s': self.rates_rad_s.copy(),
                'requested_angles_rad': self.requested_rad.copy(),
                'angle_limited': self.angle_limited, 'rate_limited': self.rate_limited}


def publish_gimbal_camera(camera, gimbal, state, com_B, target_position_w):
    """Match the initialized CPU camera view's float32 parent transforms."""
    position, quaternion = gimbal.camera_pose(state, com_B)
    camera.set_world_poses(np.asarray(position+target_position_w, dtype=np.float32)[None],
                          np.asarray(quaternion, dtype=np.float32)[None], convention='world')


def framing_trace(gimbal, states, com_B, target_points, target_center, intrinsic,
                  resolution_wh, dt, required_margin_px):
    """Check the whole convex target at every prediction node without mutation.

    A failed check rejects the candidate command; it does not add symbolic
    constraints to the optimizer. Projection containment does not test occlusion.
    """
    from .replay_capture import projected_mesh_bounds
    predicted = deepcopy(gimbal)
    width, height = resolution_wh
    margins, depths = [], []
    for index, state in enumerate(states):
        if index:
            predicted.advance(state, com_B, target_center, dt)
        position, quaternion = predicted.camera_pose(state, com_B)
        projection = projected_mesh_bounds(target_points, position, quaternion, intrinsic, width, height)
        margin = projection['edge_margin_px']
        margins.append(float(margin) if margin is not None and projection['all_vertices_in_front'] else -np.inf)
        depths.append(float(np.min((np.asarray(target_points)-position)@quaternion_to_rotation(quaternion)[:, 0])))
    minimum, minimum_depth = min(margins), min(depths)
    return {'accepted': bool(minimum>=required_margin_px and minimum_depth>=.01),
            'minimum_edge_margin_px': minimum, 'minimum_depth_m': minimum_depth,
            'required_edge_margin_px': float(required_margin_px), 'edge_margins_px': margins}
