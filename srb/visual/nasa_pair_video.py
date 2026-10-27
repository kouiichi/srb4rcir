"""External video capture at measured physical times; never advances physics."""
import numpy as np


def tracking_view(points_world, horizontal_fov_deg, aspect):
    """Fit all original mesh vertices inside a perspective view with a margin."""
    points=np.asarray(points_world)
    center=(points.min(axis=0)+points.max(axis=0))/2
    outward=np.array([-.08,-1.,.45]);outward/=np.linalg.norm(outward)
    right=np.cross(-outward,[0.,0.,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,-outward)
    relative=points-center
    tan_h=np.tan(np.deg2rad(horizontal_fov_deg)/2)
    tan_v=tan_h/aspect
    extent=np.maximum(np.abs(relative@right)/tan_h,np.abs(relative@up)/tan_v)
    distance=max(1.4,float(np.max(relative@outward+1.2*extent)))
    return center+distance*outward,center


def record_measured_frame(env, writer, publish, actor_position, target_position, *, mesh_points_world=None,
                          horizontal_fov_deg=60., aspect=1280/720, camera_pose=None):
    """Publish CPU PhysX transforms and flush the selected view before RGB read.

    A supplied eye/lookat pose stays fixed regardless of spacecraft motion.
    """
    if camera_pose is None:
        actor, target = np.asarray(actor_position), np.asarray(target_position)
        points=np.vstack((actor,target)) if mesh_points_world is None else mesh_points_world
        eye,center=tracking_view(points,horizontal_fov_deg,aspect)
    else:
        eye,center=np.asarray(camera_pose)[:3],np.asarray(camera_pose)[3:]
    env.sim.set_camera_view(eye, center)
    publish()
    # Rendering twice at the same physical time flushes the camera pose and
    # current USD transforms without submitting forces or stepping PhysX.
    env.sim.render()
    env.sim.render()
    rgb = env.render(recompute=True)
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise RuntimeError('Renderer did not return a nonempty RGB frame')
    writer.append_data(rgb)
    return np.r_[eye, center]
