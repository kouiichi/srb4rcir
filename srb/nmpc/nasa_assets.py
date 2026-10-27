"""Compose NASA visuals and the derived NEAR antenna-replacement variant."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from .nasa_pair import bottom_to_top_goal
from .quaternion import quaternion_to_rotation
from .surface_approach import read_actor_mesh


def original_geometry(root, desdyni_scale=1., near_scale=1., gap_m=.5):
    """Use DESDynI's open rim and the derived NEAR boss top as the target plane."""
    from pxr import Gf, Usd, UsdGeom
    root = Path(root)
    actor_path = root/'assets/nasa_candidates/desdyni/visual/desdyni_original.usdc'
    near_visual = root/'assets/nasa_candidates/near_shoemaker/visual'
    source_target_path = near_visual/'near_shoemaker_original.usdc'
    target_path = near_visual/'near_shoemaker_antenna_boss.usdc'
    metadata_path = target_path.with_suffix('.metadata.json')
    current_manifest = None
    if metadata_path.is_file():
        try:
            current_manifest = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            current_manifest = None
    source_hash = hashlib.sha256(source_target_path.read_bytes()).hexdigest()
    if (not target_path.is_file() or current_manifest is None
            or current_manifest.get('generator_version') != 1
            or current_manifest.get('source_sha256') != source_hash):
        from .near_variant import build_near_variant
        build_near_variant(source_target_path, target_path)
    actor_stage,target_stage = (Usd.Stage.Open(str(p)) for p in (actor_path,target_path))
    actor_points,_,actor_meshes = read_actor_mesh(actor_stage,str(actor_stage.GetDefaultPrim().GetPath()))
    target_points,_,target_meshes = read_actor_mesh(target_stage,str(target_stage.GetDefaultPrim().GetPath()))
    boss_path = str(target_stage.GetDefaultPrim().GetPath().AppendChild('antenna_replacement_boss'))
    boss_points,_,_ = read_actor_mesh(target_stage,boss_path)
    boss_transform,_ = UsdGeom.XformCache().ComputeRelativeTransform(
        target_stage.GetPrimAtPath(boss_path),target_stage.GetDefaultPrim())
    boss_points = np.asarray([
        boss_transform.Transform(Gf.Vec3d(*map(float, point))) for point in boss_points
    ])
    if np.ptp(boss_points[:,2])<=0:
        raise ValueError('NEAR antenna-replacement boss has no height')
    actor_points,target_points,boss_points = actor_points*desdyni_scale,target_points*near_scale,boss_points*near_scale
    rim = actor_points[actor_points[:,2] <= actor_points[:,2].min()+1e-5*desdyni_scale]
    bottom = (rim.min(axis=0)+rim.max(axis=0))/2
    top = (boss_points.min(axis=0)+boss_points.max(axis=0))/2
    top[2] = boss_points[:,2].max()
    q = np.array([np.sqrt(.5),0.,-np.sqrt(.5),0.])
    goal,goal_q = bottom_to_top_goal(np.zeros(3),q,bottom,top,gap_m)
    normal = quaternion_to_rotation(q)@[0.,0.,1.]
    final_actor = actor_points@quaternion_to_rotation(goal_q).T+goal
    target_world = target_points@quaternion_to_rotation(q).T
    return {
        'actor_usd':str(actor_path.resolve()),'target_usd':str(target_path.resolve()),
        'actor_mesh_paths':actor_meshes,'target_mesh_paths':target_meshes,
        'actor_points':actor_points,'target_points':target_points,
        'actor_support_points':actor_points[ConvexHull(actor_points).vertices],
        'target_support_points':target_points[ConvexHull(target_points).vertices],
        'actor_bottom':bottom,'target_top':top,'target_quaternion':q,
        'goal_position':goal,'goal_quaternion':goal_q,
        'desdyni_scale_m_per_source_unit':desdyni_scale,'near_scale_m_per_source_unit':near_scale,
        'desired_face_gap_m':gap_m,
        'target_top_definition':'NEAR antenna-replacement boss top face',
        'target_variant':'near_antenna_boss',
        'actor_bottom_definition':'plane through original minimum-Z open rim; no closure panel',
        'goal_mesh_support_gap_m':float(np.min(final_actor@normal)-np.max(target_world@normal)),
        'mesh_support_gap_scope':'positive certifies disjoint meshes; negative is inconclusive for the open cavity',
    }


def write_vehicle_usd(output, source_usd, scale, mass_properties=None):
    """Reference the untouched visual and optionally author explicit rigid properties."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(Path(output).resolve()))
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    root = UsdGeom.Xform.Define(stage,'/Spacecraft')
    stage.SetDefaultPrim(root.GetPrim())
    visual = UsdGeom.Xform.Define(stage,'/Spacecraft/Visuals')
    visual.GetPrim().GetReferences().AddReference(str(Path(source_usd).resolve()))
    visual.AddScaleOp().Set(Gf.Vec3d(float(scale)))
    if mass_properties is not None:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass.CreateMassAttr(float(mass_properties['mass_kg']))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,mass_properties['com_B_m'])))
        inertia = np.asarray(mass_properties['inertia_B_kg_m2'],dtype=float)
        moments,axes = np.linalg.eigh(inertia)
        if np.min(moments)<=0 or not np.allclose(inertia,inertia.T):
            raise ValueError('explicit inertia must be symmetric positive definite')
        # For a body-aligned tensor, keep its axis order and an identity mass
        # frame. This also keeps SRB's proportional wet-inertia tensor setters
        # in the same frame as their body-frame inertia readback.
        if np.allclose(inertia,np.diag(np.diag(inertia)),rtol=0.,atol=1e-12):
            moments,axes=np.diag(inertia).copy(),np.eye(3)
        if np.linalg.det(axes)<0:
            axes[:,-1] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,moments)))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
    stage.GetRootLayer().Save()
    return stage


def write_pair_scene(output, geometry, actor_position, actor_quaternion, target_position):
    """Write a visual review stage using the selected DESDynI and NEAR assets."""
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(Path(output).resolve()))
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    world = UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    for name,path,scale,pos,q in (
        ('DESDynI',geometry['actor_usd'],geometry['desdyni_scale_m_per_source_unit'],np.asarray(actor_position)+target_position,actor_quaternion),
        ('NEAR',geometry['target_usd'],geometry['near_scale_m_per_source_unit'],target_position,geometry['target_quaternion']),
    ):
        body = UsdGeom.Xform.Define(stage,'/World/'+name)
        body.AddTranslateOp().Set(Gf.Vec3d(*map(float,pos)))
        body.AddOrientOp().Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:]))))
        visual = UsdGeom.Xform.Define(stage,'/World/'+name+'/Visuals')
        visual.GetPrim().GetReferences().AddReference(path)
        visual.AddScaleOp().Set(Gf.Vec3d(float(scale)))
    stage.GetRootLayer().Save()


def desdyni_thruster_geometry(root, actor=None):
    """Keep the four measured nozzles, then append explicit virtual RCS channels.

    Measured offsets use source units and the configured scale. Virtual offsets
    are already body-frame metres; all directions describe unit exhaust axes.
    """
    info = json.loads((Path(root)/'assets/nasa_candidates/desdyni/measurements/thruster_geometry.json').read_text())
    actor = {} if actor is None else actor
    offsets = np.asarray([n['mouth_position'] for n in info['nozzles']])*actor.get('scale_m_per_source_unit', 1.)
    directions = np.tile(info['nominal_exhaust_axis'], (len(offsets), 1))
    virtual = actor.get('virtual_thrusters', [])
    if virtual:
        offsets = np.vstack((offsets, [n['offset_B_m'] for n in virtual]))
        directions = np.vstack((directions, [n['exhaust_direction_B'] for n in virtual]))
    return offsets, directions


def load_physics_config(path):
    """Require explicit scale and physics; the visual source supplies neither."""
    cfg = json.loads(Path(path).read_text())
    if cfg.get('parameter_basis') not in ('research','engineering') or not cfg.get('source_or_rationale'):
        raise ValueError('physics config requires parameter_basis and source_or_rationale')
    for name in ('desdyni','near_shoemaker'):
        vehicle = cfg[name]
        for field in ('scale_m_per_source_unit','mass_kg'):
            value = vehicle[field]
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{name}.{field} must be positive finite')
        inertia,com = np.asarray(vehicle['inertia_B_kg_m2']),np.asarray(vehicle['com_B_m'])
        if (inertia.shape!=(3,3) or com.shape!=(3,) or not np.all(np.isfinite(inertia))
                or not np.all(np.isfinite(com)) or not np.allclose(inertia,inertia.T)
                or np.linalg.eigvalsh(inertia).min()<=0):
            raise ValueError(f'{name} requires explicit finite COM and positive definite inertia')
    actor = cfg['desdyni']
    virtual = actor.get('virtual_thrusters', [])
    for nozzle in virtual:
        position = np.asarray(nozzle['offset_B_m'], dtype=float)
        direction = np.asarray(nozzle['exhaust_direction_B'], dtype=float)
        if (position.shape != (3,) or direction.shape != (3,)
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(direction))
                or not np.isclose(np.linalg.norm(direction), 1., rtol=0., atol=1e-8)):
            raise ValueError('desdyni.virtual_thrusters require finite offsets and unit exhaust directions')
    powers = np.asarray(actor['thruster_power_N'],dtype=float)
    if powers.shape!=(4+len(virtual),) or not np.all(np.isfinite(powers)) or np.any(powers<=0):
        raise ValueError('explicit positive thruster powers must cover four original and all virtual nozzles')
    for field in ('fuel_capacity_kg','fuel_consumption_rate'):
        if not np.isfinite(actor[field]) or actor[field]<0:
            raise ValueError(f'desdyni.{field} must be nonnegative finite')
    if actor['fuel_capacity_kg']==0:
        raise ValueError('desdyni.fuel_capacity_kg must be positive for rendezvous fuel observations')
    return cfg
