"""Geometry for a stationary target -X face / actor +Z face experiment.

Mesh separation is a signed supporting-plane gap, not closest-point distance.
A positive gap certifies disjoint meshes at the evaluated pose. Face centers
refer to bus panels; appendages are included separately in the safety guard.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .quaternion import quaternion_to_rotation


def largest_axis_face(points, faces, outward_normal):
    """Find the largest planar polygon with the selected outward normal."""
    points = np.asarray(points, dtype=float)
    normal = np.asarray(outward_normal, dtype=float)
    candidates = []
    for indices in faces:
        p = points[indices]
        area_vector = np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum(axis=0) / 2
        area = np.linalg.norm(area_vector)
        if area > 1e-12 and np.dot(area_vector / area, normal) > 1 - 1e-6:
            if np.ptp(p @ normal) < 1e-6:
                candidates.append((area, p.mean(axis=0)))
    if not candidates:
        raise ValueError(f'No planar face with outward normal {normal}')
    return max(candidates, key=lambda pair: pair[0])[1]


def read_actor_mesh(stage, root_path):
    """Read actual USD meshes in actor-root coordinates, including appendages."""
    from pxr import Gf, Usd, UsdGeom
    root = stage.GetPrimAtPath(root_path)
    cache = UsdGeom.XformCache()
    all_points, all_faces, paths = [], [], []
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix, _ = cache.ComputeRelativeTransform(prim, root)
        points = np.asarray([
            matrix.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            for p in mesh.GetPointsAttr().Get()
        ])
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        offsets = np.r_[0, np.cumsum(mesh.GetFaceVertexCountsAttr().Get())]
        all_faces.extend([indices[a:b] + count for a, b in zip(offsets[:-1], offsets[1:])])
        all_points.append(points)
        count += len(points)
        paths.append(str(prim.GetPath()))
    if not all_points:
        raise ValueError(f'No meshes beneath {root_path}')
    return np.concatenate(all_points), all_faces, paths


def face_goal(target_position, target_quaternion, target_face, actor_top, gap_m):
    """Actor pose whose +Z top faces the target -X front with aligned +Y axes."""
    if not np.isfinite(gap_m) or gap_m <= 0:
        raise ValueError('surface gap must be positive and finite')
    rt = quaternion_to_rotation(target_quaternion)
    # +Z_actor -> +X_target; +Y_actor -> +Y_target. A proper +90deg Y rotation.
    ra = rt @ np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    q_xyzw = Rotation.from_matrix(ra).as_quat()
    q = q_xyzw[[3, 0, 1, 2]]
    p = np.asarray(target_position) + rt @ (np.asarray(target_face) + gap_m*np.array([-1.,0.,0.])) - ra @ actor_top
    return p, q


def surface_sample(state, com, actor_top, target_face, actor_points, target_points,
                   target_quaternion=(1., 0., 0., 0.)):
    """Evaluate a COM state relative to a stationary target actor origin."""
    x = np.asarray(state)
    rt, ra = quaternion_to_rotation(target_quaternion), quaternion_to_rotation(x[6:10])
    normal = rt @ [-1., 0., 0.]
    actor_position = x[:3] - ra @ com
    delta = actor_position + ra @ actor_top - rt @ target_face
    gap = float(delta @ normal)
    actor_support = (np.asarray(actor_points) @ ra.T + actor_position) @ normal
    target_support = np.asarray(target_points) @ rt.T @ normal
    return {
        'face_gap_m': gap,
        'lateral_error_m': float(np.linalg.norm(delta-gap*normal)),
        'facing_error_deg': float(np.rad2deg(np.arccos(np.clip((ra @ [0.,0.,1.]) @ -normal, -1, 1)))),
        'mesh_plane_gap_m': float(np.min(actor_support)-np.max(target_support)),
    }


def synchronous_reference(times, actor_start, quaternion_start, actor_goal,
                          quaternion_goal, com, duration_s):
    """Translate and turn together with one quintic progress profile.

    Positions supplied by the caller are actor roots; returned states are COM
    states with the rotational velocity offset and body angular velocity.
    After duration_s the reference holds the final pose with zero velocity.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('sync duration must be positive and finite')
    times = np.asarray(times, dtype=float)
    progress = np.clip(times / duration_s, 0., 1.)
    progress_rate = (30 * progress**2 * (1 - progress)**2) / duration_s
    progress = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
    r0 = Rotation.from_matrix(quaternion_to_rotation(quaternion_start))
    rg = Rotation.from_matrix(quaternion_to_rotation(quaternion_goal))
    rotation_vector = (r0.inv() * rg).as_rotvec()
    rotations = r0 * Rotation.from_rotvec(progress[:, None] * rotation_vector)
    omega = progress_rate[:, None] * rotation_vector
    delta = np.asarray(actor_goal) - actor_start
    result = np.zeros((len(times), 13))
    result[:, :3] = np.asarray(actor_start) + progress[:, None] * delta + rotations.apply(com)
    result[:, 3:6] = progress_rate[:, None] * delta + rotations.apply(np.cross(omega, com))
    result[:, 6:10] = rotations.as_quat()[:, [3, 0, 1, 2]]
    result[:, 10:13] = omega
    return result


def apply_display_materials(stage, actor_paths, target_paths, actor_top, target_face):
    """Temporary evaluation colours for these assets' all-black baked albedos.

    Only material bindings are authored. The selected planar bus faces are
    highlighted; original cache assets, mesh points and physics are untouched.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    for name, paths, face, axis, colour in (
        ('actor', actor_paths, actor_top, 2, (.22,.48,.72)),
        ('target', target_paths, target_face, 0, (.80,.56,.22)),
    ):
        materials = []
        for suffix, rgb in (('body',colour), ('face',(.18,.88,.63))):
            material = UsdShade.Material.Define(stage, f'/World/replay_materials/{name}_{suffix}')
            shader = UsdShade.Shader.Define(stage, str(material.GetPath())+'/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.55)
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.1)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            materials.append(material)
        # The validated procedural cubesats have one merged mesh with identity
        # relative transform; fail rather than highlighting incorrect faces.
        if len(paths) != 1:
            raise ValueError('Display face highlighting expects the validated single-mesh cubesat')
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(paths[0]))
        points = np.asarray(mesh.GetPointsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        offsets = np.r_[0,np.cumsum(mesh.GetFaceVertexCountsAttr().Get())]
        selected = [i for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:]))
                    if np.all(np.abs(points[indices[a:b],axis]-face[axis]) < 1e-6)]
        if not selected:
            raise ValueError('No display face found at the measured bus face coordinate')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[0])
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, 'approach_face', UsdGeom.Tokens.face,
                                                 selected, 'materialBind')
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(materials[1])
