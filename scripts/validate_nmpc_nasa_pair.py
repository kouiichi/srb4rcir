#!/usr/bin/env python3
"""Inspect, plan or execute original DESDynI-bottom / NEAR-top rendezvous.

Geometry inspection uses source units. Planning and PhysX execution require
an explicit physics JSON; no engineering scale, mass or thruster power is
inferred from a visual model. Original asset files remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
KIT_THREAD_SETTINGS = ('/plugins/carb.tasking.plugin/threadCount',
                       '/plugins/omni.tbb.globalcontrol/maxThreadCount', '/persistent/physics/numThreads')

from srb.nmpc.nasa_assets import desdyni_thruster_geometry, load_physics_config, original_geometry, write_pair_scene, write_vehicle_usd
from srb.nmpc.nasa_pair import actuation_diagnostics, full_mesh_plane_gaps, plane_sample, thruster_wrench_matrix, turn_then_approach_reference
from srb.nmpc.quaternion import quaternion_to_rotation
from srb.nmpc.surface_approach import synchronous_reference


def save(path,value):
    def convert(x):
        if isinstance(x,np.ndarray): return convert(x.tolist())
        if isinstance(x,np.generic): return convert(x.item())
        if isinstance(x,float) and not np.isfinite(x): return None
        if isinstance(x,dict): return {k:convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        return x
    Path(path).write_text(json.dumps(convert(value),indent=2,allow_nan=False)+'\n')


def initial_state(args, com):
    """Build a COM state from the randomized root pose and body angular rate."""
    from scipy.spatial.transform import Rotation
    rpy = getattr(args, 'initial_attitude_rpy_deg', (0., 0., 0.))
    q = Rotation.from_euler('xyz', rpy, degrees=True).as_quat()[[3, 0, 1, 2]]
    state = np.zeros(13)
    state[:3] = np.asarray(args.start) + quaternion_to_rotation(q) @ com
    state[3:6] = getattr(args, 'initial_linear_velocity', (0., 0., 0.))
    state[6:10] = q
    state[10:13] = getattr(args, 'initial_angular_velocity', (0., 0., 0.))
    return state


def apply_initial_state(env, args):
    """Write the requested root state after reset, before the first force."""
    import torch
    state = initial_state(args, np.zeros(3))
    root = env._robot.data.root_state_w.clone()
    def tensor(values):
        return torch.as_tensor(values, dtype=root.dtype, device=root.device)
    root[:, :3] = tensor(np.asarray(args.start) + args.target_position) + env.scene.env_origins
    root[:, 3:7] = tensor(state[6:10])
    root[:, 7:10] = tensor(state[3:6])
    root[:, 10:13] = tensor(quaternion_to_rotation(state[6:10]) @ state[10:13])
    env._robot.write_root_state_to_sim(root)
    env._robot.reset()


def direct_physx_state13(env, com):
    """Read PhysX itself, independent of same-timestamp asset data caches."""
    actor = env._robot.root_physx_view
    target = env._target.root_physx_view
    pose = np.asarray(actor.get_transforms()[0], dtype=float)
    velocity = np.asarray(actor.get_velocities()[0], dtype=float)
    target_pose = np.asarray(target.get_transforms()[0], dtype=float)
    target_velocity = np.asarray(target.get_velocities()[0], dtype=float)
    quaternion = pose[3:7][[3,0,1,2]]
    rotation = quaternion_to_rotation(quaternion)
    return np.r_[pose[:3]+rotation@com-target_pose[:3], velocity[:3]-target_velocity[:3],
                 quaternion, rotation.T@velocity[3:6]]


def reference_states(args, geometry, times, com):
    """Use one shared progress profile for translation and attitude by default."""
    common = (times, np.asarray(args.start), initial_state(args, com)[6:10],
              geometry['goal_position'], geometry['goal_quaternion'], com)
    if args.reference_mode == 'synchronous':
        return synchronous_reference(*common, args.sync_seconds)
    return turn_then_approach_reference(*common, args.turn_seconds, args.approach_seconds)


def goal_sample_is_good(sample, state, gap):
    return (abs(sample['face_gap_m']-gap)<.02 and sample['lateral_error_m']<.02
            and sample['facing_error_deg']<5. and np.linalg.norm(state[3:6])<.05
            and np.linalg.norm(state[10:13])<.05)


def goal_hold_complete(consecutive_samples, dt, fuel):
    """A one-second hold spans 21 measured samples at 20 Hz."""
    return (consecutive_samples-1)*dt>=1. and fuel>1e-6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('geometry','plan','replay'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physics-config',type=Path,help='Explicit scales, masses, inertia, COM, fuel, nozzle powers and optional virtual RCS')
    parser.add_argument('--start',nargs=3,type=float,default=(-3.,.2,.15))
    parser.add_argument('--initial-linear-velocity',nargs=3,type=float,default=(0.,0.,0.),help='Initial COM velocity in world axes (m/s)')
    parser.add_argument('--initial-attitude-rpy-deg',nargs=3,type=float,default=(0.,0.,0.))
    parser.add_argument('--initial-angular-velocity',nargs=3,type=float,default=(0.,0.,0.),help='Initial body angular velocity (rad/s)')
    parser.add_argument('--target-position',nargs=3,type=float,default=(0.,0.,0.))
    parser.add_argument('--surface-gap',type=float,default=.5)
    parser.add_argument('--reference-mode',choices=('synchronous','turn-then-approach'),default='synchronous')
    parser.add_argument('--sync-seconds',type=float,default=50.,help='Shared translation and attitude duration')
    parser.add_argument('--turn-seconds',type=float,default=20.)
    parser.add_argument('--approach-seconds',type=float,default=30.)
    parser.add_argument('--horizon-steps',type=int,default=40)
    parser.add_argument('--execute-steps',type=int,default=3)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--stop-on-success',action='store_true',help='Finish after a strict one-second measured goal hold with fuel remaining; --steps is the upper limit')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda:1',help='Kit device; environment and PhysX tensors remain CPU')
    parser.add_argument('--kit-worker-threads',type=int,help='Explicit Carbonite/TBB/physics worker limit for concurrent runs')
    parser.add_argument('--physics-only',action='store_true')
    parser.add_argument('--video',action='store_true',help='Record actual PhysX steps with the external RTX camera')
    parser.add_argument('--record-camera',action='store_true',help='Capture RGB/depth from the DESDynI camera; requires --video')
    parser.add_argument('--stream-camera',action='store_true',help='Store camera pixels per frame instead of caching the entire episode')
    parser.add_argument('--camera-readback-device',choices=('cpu','cuda'),default='cpu',help='Replicator image transfer buffers; environment, camera tensors and PhysX remain CPU')
    parser.add_argument('--profile-loop',action='store_true',help='Save cProfile timings for the control/capture loop before Kit shuts down')
    parser.add_argument('--camera-tracking',choices=('fixed','limited-gimbal'),default='fixed')
    parser.add_argument('--camera-fov-gate',action='store_true',help='Reject plans whose target projection violates the limited-gimbal camera rectangle')
    parser.add_argument('--camera-framing-margin',type=float,default=8.,help='Required full-target margin inside every image edge, in pixels')
    parser.add_argument('--camera-resolution',nargs=2,type=int,default=(640,480),metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--camera-horizontal-fov',type=float,default=90.)
    parser.add_argument('--camera-save-every',type=int,default=20,help='Save raw onboard images/depth every N control steps; video records every step')
    args = parser.parse_args(argv)
    if args.mode!='geometry' and args.physics_config is None:
        parser.error('plan/replay require --physics-config; original NASA visuals do not specify physics')
    if args.mode=='replay' and args.physics_only == args.video:
        parser.error('replay requires exactly one of --physics-only or --video')
    if args.video and args.mode!='replay':
        parser.error('--video requires replay: frames must come from actual PhysX steps')
    if args.record_camera and not args.video:
        parser.error('--record-camera requires --video and replay')
    if args.stream_camera and not args.record_camera:
        parser.error('--stream-camera requires --record-camera')
    if args.kit_worker_threads is not None and args.kit_worker_threads <= 0:
        parser.error('--kit-worker-threads must be positive')
    if args.camera_fov_gate and args.camera_tracking!='limited-gimbal':
        parser.error('--camera-fov-gate requires --camera-tracking limited-gimbal')
    if not np.isfinite(args.camera_framing_margin) or args.camera_framing_margin<0:
        parser.error('camera framing margin must be finite and nonnegative')
    if (min(args.camera_resolution)<=0 or args.camera_save_every<=0
            or not np.isfinite(args.camera_horizontal_fov) or not 0<args.camera_horizontal_fov<180):
        parser.error('camera resolution/save interval must be positive and FOV within (0,180) degrees')
    if args.horizon_steps<=0 or args.horizon_steps%20 or not 1<=args.execute_steps<=args.horizon_steps or args.steps<=0:
        parser.error('positive horizon multiple of 20, valid execute prefix, and positive steps required')
    values = [*args.start,*args.target_position,*args.initial_linear_velocity,
              *args.initial_attitude_rpy_deg,*args.initial_angular_velocity,
              args.surface_gap,args.sync_seconds,args.turn_seconds,args.approach_seconds]
    if not np.all(np.isfinite(values)) or min(args.surface_gap,args.sync_seconds,args.turn_seconds,args.approach_seconds)<=0:
        parser.error('poses must be finite; gap and reference durations positive finite')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'invocation.json',{
        'cwd':str(Path.cwd()),'argv':sys.argv if argv is None else argv,
        'command':shlex.join(sys.argv),'python':sys.executable,'mode':args.mode,
        'kit_device':args.device,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'CUDA_DEVICE_ORDER':os.environ.get('CUDA_DEVICE_ORDER'),
        'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
        'physics_only':args.physics_only,'video':args.video,'record_camera':args.record_camera,'seed':args.seed,
    })
    # Kit must register its USD/PhysX schemas before any source USD is opened.
    launcher=launch_replay(args) if args.mode=='replay' else None
    if args.mode=='replay' and launcher is None:
        return 2
    try:
        return prepare_experiment(args,launcher)
    finally:
        if launcher:launcher.app.close()


def launch_replay(args):
    import torch
    preflight={'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'kit_device':args.device}
    save(args.output/'gpu_preflight.json',preflight)
    if args.device.startswith('cuda') and (not preflight['cuda_available'] or int(args.device.split(':')[1])>=preflight['cuda_device_count']):
        save(args.output/'runtime_error.json',{'type':'GPUAccessUnavailable','requested_steps':args.steps,'executed_steps':0,
            'message':'Requested Kit GPU unavailable; no PhysX or control steps executed.'})
        return None
    from isaacsim import SimulationApp
    SimulationApp.DEFAULT_LAUNCHER_CONFIG['create_new_stage']=False
    from srb.core.app import AppLauncher
    from srb.utils.path import SRB_APPS_DIR
    return AppLauncher(headless=True,enable_cameras=args.video,multi_gpu=False,device=args.device,
        kit_args=' '.join(f'--{path}={args.kit_worker_threads}' for path in KIT_THREAD_SETTINGS) if args.kit_worker_threads else '',
        experience=SRB_APPS_DIR/('srb.headless.rendering.kit' if args.video else 'srb.headless.kit'))


def prepare_experiment(args,launcher):
    physics = load_physics_config(args.physics_config) if args.physics_config else None
    if physics:
        save(args.output/'physics_config.json',physics)
    geometry = original_geometry(ROOT,
        physics['desdyni']['scale_m_per_source_unit'] if physics else 1.,
        physics['near_shoemaker']['scale_m_per_source_unit'] if physics else 1.,args.surface_gap)
    reference_config = {'mode':args.reference_mode}
    if args.reference_mode == 'synchronous':
        reference_config['duration_s'] = args.sync_seconds
    else:
        reference_config.update(turn_s=args.turn_seconds,approach_s=args.approach_seconds)
    save(args.output/'reference_config.json',reference_config)
    summary = {k:v for k,v in geometry.items() if k not in ('actor_points','target_points','actor_support_points','target_support_points')}
    turned_state=np.zeros(13);turned_state[:3]=args.start;turned_state[6:10]=geometry['goal_quaternion']
    after_turn=plane_sample(turned_state,np.zeros(3),geometry['actor_bottom'],geometry['target_top'],geometry['target_quaternion'])
    summary.update({'physical_scale_verified_here':False,
                    'scale_source':physics['parameter_basis'] if physics else 'uncalibrated source coordinates',
                    'initial_actor_root_relative_m':args.start,'initial_actor_quaternion_wxyz':initial_state(args,np.zeros(3))[6:10],
                    'initial_actor_com_velocity_world_m_s':args.initial_linear_velocity,
                    'initial_actor_angular_velocity_body_rad_s':args.initial_angular_velocity,
                    'target_initial_linear_angular_velocity':[0.]*6,'target_free_floating':True,
                    'target_pose_is_initial_orientation_not_imposed_spin':True,
                    'reference':reference_config,
                    'components_added':True,
                    'target_variant':geometry['target_variant'],
                    'source_target_geometry_preserved':True,
                    'virtual_rcs_channels':len(physics['desdyni'].get('virtual_thrusters',[])) if physics else 0,
                    'onboard_camera_added':args.record_camera})
    if args.reference_mode=='turn-then-approach':
        summary.update(after_stationary_turn_plane_metrics=after_turn,
                       forward_approach_after_turn=after_turn['face_gap_m']>=args.surface_gap)
    save(args.output/'geometry.json',summary)
    write_pair_scene(args.output/'initial_scene.usda',geometry,np.asarray(args.start),initial_state(args,np.zeros(3))[6:10],np.asarray(args.target_position))
    write_pair_scene(args.output/'goal_scene.usda',geometry,geometry['goal_position'],geometry['goal_quaternion'],np.asarray(args.target_position))
    nozzle_info = json.loads((ROOT/'assets/nasa_candidates/desdyni/measurements/thruster_geometry.json').read_text())
    actor = physics['desdyni'] if physics else None
    offsets,directions = desdyni_thruster_geometry(ROOT,actor)
    powers = np.asarray(actor['thruster_power_N']) if actor else np.ones(len(offsets))
    com = np.asarray(actor['com_B_m']) if actor else np.zeros(3)
    matrix = thruster_wrench_matrix(offsets,directions,powers,com)
    diagnosis = actuation_diagnostics(matrix,[0.,0.,-1.])
    diagnosis.update({'force_scale':'configured newtons per unit duty' if actor else 'unit force per nozzle for geometry-only rank test',
                      'identified_nozzles':nozzle_info['identified_nozzle_count'],
                      'virtual_nozzles':len(offsets)-nozzle_info['identified_nozzle_count'],
                      'offsets_B_m':offsets,'powers_N':powers,'rcs_matrix':matrix,
                      'fitted_directions': [n['exhaust_direction'] for n in nozzle_info['nozzles']],
                      'nominal_directions':directions,
                      'task_implication':('fixed-attitude forward and reverse thrust available; closed-loop convergence requires testing'
                          if diagnosis['can_brake_at_fixed_attitude'] else
                          'at the final fixed attitude only forward thrust is available; straight inbound motion cannot be braked with these nozzles')})
    save(args.output/'actuation_diagnostics.json',diagnosis)
    if args.mode=='geometry':
        refs = reference_states(args,geometry,np.arange(args.steps+1)*.05,com)
        np.savez_compressed(args.output/'geometric_reference.npz',states=refs,time_s=np.arange(len(refs))*.05)
        save(args.output/'result.json',{'mode':'geometry','requested_control_steps':0,'executed_control_steps':0,
            'nmpc_solves':0,'physics_started':False,'physical_scale_calibrated':False,
            'goal_plane_metrics':plane_sample(refs[-1],np.zeros(3),geometry['actor_bottom'],geometry['target_top'],geometry['target_quaternion']),
            'reference_only':True,'complete_success':False,'actuation':diagnosis})
        print('Derived NEAR geometry prepared; no NMPC/PhysX steps executed.',flush=True)
        return 0
    if args.reference_mode=='turn-then-approach' and after_turn['face_gap_m']<args.surface_gap:
        save(args.output/'runtime_error.json',{'type':'InitialGeometryIncompatible','requested_steps':args.steps,'executed_steps':0,
            'message':'At the specified scale/start, turning at the initial root puts the DESDynI bottom beyond the approach goal. Select explicit scale/start before a turn-then-forward-approach experiment.',
            'after_turn_plane_gap_m':after_turn['face_gap_m'],'requested_gap_m':args.surface_gap})
        return 2
    return run_controller(args,physics,geometry,offsets,directions,launcher)


def run_controller(args,physics,geometry,offsets,directions,launcher=None):
    from srb.nmpc import TaskParams
    from srb.nmpc.matched_executor import MatchedExecutor
    from srb.nmpc.nasa_expert import NasaPairExpert
    actor,target = physics['desdyni'],physics['near_shoemaker']
    com = np.asarray(actor['com_B_m'],dtype=float)
    powers = np.asarray(actor['thruster_power_N'],dtype=float)
    b = thruster_wrench_matrix(offsets,directions,powers,com)
    dt = .05
    params = TaskParams(mean_motion_rad_s=0.,horizon_steps=args.horizon_steps,
        action_block_lengths=(3,3,3,3,2,2,2,2)*(args.horizon_steps//20),
        koz_radius_m=float(np.linalg.norm(geometry['target_points'],axis=1).max()))
    executor = MatchedExecutor(actor['mass_kg'],actor['fuel_consumption_rate']*dt*powers,dt,.05)
    save(args.output/'controller_params.json',asdict(params))
    x = initial_state(args,com)
    physical_mass = actor['mass_kg']+actor['fuel_capacity_kg']
    physical_inertia = np.asarray(actor['inertia_B_kg_m2'])*physical_mass/actor['mass_kg']
    fuel = actor['fuel_capacity_kg']
    env,term,writer,onboard = None,None,None,None
    gimbal=None;gimbal_records=[];planner_times=[]
    loop_profiler=None
    onboard_spec=None
    if args.record_camera or args.camera_tracking=='limited-gimbal':
        from srb.visual.nasa_pair_onboard import onboard_camera_spec, gimbal_camera_spec, LimitedCameraGimbal, camera_intrinsic, framing_trace, publish_gimbal_camera
        if args.camera_tracking=='limited-gimbal':
            onboard_spec=gimbal_camera_spec(geometry['actor_bottom'],geometry['actor_points'],args.camera_resolution,args.camera_horizontal_fov)
            gimbal=LimitedCameraGimbal(onboard_spec['position_B_m'],onboard_spec['quaternion_BC_wxyz'])
            target_camera_points=geometry['target_support_points']@quaternion_to_rotation(geometry['target_quaternion']).T
            target_camera_center=(target_camera_points.min(0)+target_camera_points.max(0))/2
            camera_k=camera_intrinsic(onboard_spec)
            gimbal.initialize(x,com,target_camera_center)
            save(args.output/'gimbal_config.json',{
                'model':'limited kinematic pan/tilt; initialized pointing at reset target',
                'yaw_limits_deg':np.rad2deg(gimbal.limits[0]),'pitch_limits_deg':np.rad2deg(gimbal.limits[1]),
                'joint_max_rate_deg_s':np.rad2deg(gimbal.max_rate),
                'framing_gate':args.camera_fov_gate,'required_margin_px':args.camera_framing_margin,
                'scope':'simulator target-position tracking; rectangular full-target projection acceptance gate at prediction nodes; no motor dynamics or occlusion constraint',
                'target_center_relative_m':target_camera_center,'intrinsic':camera_k})
        else:
            onboard_spec=onboard_camera_spec(geometry['actor_bottom'],args.camera_resolution,args.camera_horizontal_fov)
        save(args.output/'onboard_camera_spec.json',onboard_spec)

    def record_gimbal(step):
        trace=framing_trace(gimbal,np.asarray([states[-1]]),com,target_camera_points,target_camera_center,
                            camera_k,args.camera_resolution,dt,args.camera_framing_margin)
        gimbal_records.append({'step':step,'time_s':step*dt,**gimbal.snapshot(),
            'position_B_m':gimbal.position_B.copy(),'quaternion_BC_wxyz':gimbal.quaternion_BC,
            'edge_margin_px':trace['minimum_edge_margin_px'],'framing_valid':trace['accepted']})
    video_steps,video_poses,video_actor_poses,video_target_poses=[],[],[],[]
    video_usd_actor_poses,video_usd_target_poses=[],[]
    video_camera_intrinsics=[]
    states,actions,records,solve_times,forecast_x,forecast_u = [x.copy()],[],[],[],[],[]
    current_mass_fuel=[]
    shooting_states,solver_references,solver_physics,telemetry=[],[],[],[]
    previous=np.zeros(len(powers));warm=None;pending=None;stopped=None;max_contact=0.
    good_streak=0;success_terminated=False
    try:
        if args.mode=='replay':
            import torch
            import omni.usd
            from omni.physx import acquire_physx_interface
            from srb.core.sensor import ContactSensorCfg
            from srb.tasks.mobile.rendezvous.nasa_pair import make_task_cfg,NasaPairTask
            from pxr import PhysxSchema
            omni.usd.get_context().new_stage();acquire_physx_interface().overwrite_gpu_setting(0)
            write_vehicle_usd(args.output/'desdyni_runtime.usda',geometry['actor_usd'],actor['scale_m_per_source_unit'],actor)
            write_vehicle_usd(args.output/'near_runtime.usda',geometry['target_usd'],target['scale_m_per_source_unit'],target)
            cfg=make_task_cfg(robot_usd=args.output/'desdyni_runtime.usda',target_usd=args.output/'near_runtime.usda',
                offsets=offsets,exhaust_directions=directions,powers=powers,fuel_capacity=fuel,
                fuel_consumption_rate=actor['fuel_consumption_rate'],start=np.asarray(args.start),target_position=np.asarray(args.target_position),
                target_quaternion=geometry['target_quaternion'],goal_position=geometry['goal_position'],
                duration_s=(args.steps+100)*dt,seed=args.seed,onboard_camera=onboard_spec if args.record_camera else None)
            if args.video:
                cfg.sim.use_fabric=False
                cfg.viewer.resolution=(1280,720)
                cfg.viewer.eye=(-1.5,-4.,1.8)
                cfg.viewer.lookat=(-1.5,0.,0.)
                for name in ('randomize_sunlight_orientation','randomize_sunlight_intensity',
                             'randomize_sunlight_angular_diameter','randomize_sunlight_color_temperature',
                             'randomize_skydome_orientation'):
                    setattr(cfg.events,name,None)
            cfg.scene.replay_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/robot',update_period=0.,history_length=1)
            class AuditedNasaTask(NasaPairTask):
                def _setup_scene(self):
                    super()._setup_scene()
                    PhysxSchema.PhysxContactReportAPI.Apply(self.sim.stage.GetPrimAtPath('/World/envs/env_0/robot')).CreateThresholdAttr(0.)
                    if args.physics_only:
                        self.sim.set_render_mode(self.sim.RenderMode.NO_GUI_OR_RENDERING)
                    acquire_physx_interface().overwrite_gpu_setting(0)
            env=AuditedNasaTask(cfg,render_mode='rgb_array' if args.video else None);env.reset(seed=args.seed)
            apply_initial_state(env,args)
            term=env.action_manager._terms[env._thrust_action_term_key];term.record_execution=True
            def numpy(t):return t.detach().cpu().numpy().astype(float)
            def measured():
                r,t=env._robot.data,env._target.data
                return np.r_[numpy(r.root_com_pos_w[0]-t.root_pos_w[0]),numpy(r.root_lin_vel_w[0]-t.root_lin_vel_w[0]),numpy(r.root_quat_w[0]),numpy(r.root_ang_vel_b[0])]
            view=env._robot.root_physx_view
            com=numpy(view.get_coms())[0,:3]
            b=thruster_wrench_matrix(numpy(term._thruster_offset),numpy(term._thruster_direction),numpy(term._thruster_power)*term.cfg.scale,com)
            x=measured();states=[x.copy()]
            expected_initial=initial_state(args,com)
            compared_initial=x.copy()
            if np.dot(compared_initial[6:10],expected_initial[6:10])<0:
                compared_initial[6:10]*=-1
            initial_matches=bool(np.allclose(compared_initial,expected_initial,rtol=0.,atol=1e-5))
            save(args.output/'initial_state_check.json',{'requested_state13':expected_initial,
                'measured_state13':x,'maximum_absolute_error':float(np.max(np.abs(compared_initial-expected_initial))),
                'absolute_tolerance':1e-5,'accepted':initial_matches})
            if not initial_matches:
                raise RuntimeError('Measured reset state does not match the requested randomized initial state')
            if gimbal:gimbal.initialize(x,com,target_camera_center)
            if str(env.device)!='cpu' or not np.isclose(env.step_dt,dt):
                raise RuntimeError('NASA replay requires CPU environment tensors and 0.05 s control period')
            executor.dry_mass=float(numpy(term._dry_masses).ravel()[0])
            executor.burn_per_duty=term.cfg.fuel_consumption_rate*cfg.agent_rate*numpy(term._thruster_power)
            physical_mass=float(numpy(view.get_masses()).ravel()[0])
            fuel=float(numpy(term.remaining_fuel).ravel()[0])
            expected_inertia=np.asarray(actor['inertia_B_kg_m2'])*physical_mass/actor['mass_kg']
            measured_inertia=numpy(view.get_inertias()).reshape(3,3)
            if (not np.isclose(physical_mass,actor['mass_kg']+fuel,atol=1e-6)
                    or not np.allclose(measured_inertia,expected_inertia,rtol=1e-5,atol=1e-7)):
                raise RuntimeError('Initialized PhysX wet mass/inertia does not match the explicit research configuration')
            if args.video:
                import imageio.v2 as imageio
                from pxr import Gf,UsdGeom,UsdLux
                from srb.visual.nasa_pair_video import record_measured_frame
                for index,yaw in enumerate((-40.,40.)):
                    light=UsdLux.DistantLight.Define(env.sim.stage,f'/World/nasa_video_fill_{index}')
                    light.CreateIntensityAttr(1.);light.CreateExposureAttr(10.);light.CreateAngleAttr(15.)
                    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(65.,yaw,0.))
                acquire_physx_interface().update_transformations(False,True,False)
                for _ in range(4):env.render()
                camera=UsdGeom.Camera(env.sim.stage.GetPrimAtPath(cfg.viewer.cam_prim_path))
                camera.CreateHorizontalApertureAttr(20.955)
                camera.CreateVerticalApertureAttr(20.955*720/1280)
                camera.CreateFocalLengthAttr(float(20.955/(2*np.tan(np.deg2rad(30.)))))
                fixed_camera_pose=np.r_[cfg.viewer.eye,cfg.viewer.lookat]
                writer=imageio.get_writer(args.output/'replay.mp4',fps=1/dt,codec='libx264',quality=7,macro_block_size=2)
                def capture(step):
                    r,t=env._robot.data,env._target.data
                    actor_p,target_p=numpy(r.root_pos_w[0]),numpy(t.root_pos_w[0])
                    def publish_with_camera():
                        acquire_physx_interface().update_transformations(False,True,False)
                        if gimbal is not None and onboard is not None:
                            publish_gimbal_camera(onboard.camera,gimbal,states[-1],com,target_p)
                            onboard.set_mount_state(gimbal.position_B,gimbal.quaternion_BC,gimbal.snapshot())
                    pose=record_measured_frame(env,writer,publish_with_camera,actor_p,target_p,
                        camera_pose=fixed_camera_pose)
                    video_steps.append(step);video_poses.append(pose)
                    video_camera_intrinsics.append([camera.GetFocalLengthAttr().Get(),camera.GetHorizontalApertureAttr().Get(),
                                                    camera.GetVerticalApertureAttr().Get()])
                    video_actor_poses.append(np.r_[actor_p,numpy(r.root_quat_w[0])])
                    video_target_poses.append(np.r_[target_p,numpy(t.root_quat_w[0])])
                    cache=UsdGeom.XformCache()
                    for name,poses in (('robot',video_usd_actor_poses),('target',video_usd_target_poses)):
                        matrix=cache.GetLocalToWorldTransform(env.sim.stage.GetPrimAtPath(f'/World/envs/env_0/{name}'))
                        q=matrix.ExtractRotationQuat()
                        poses.append(np.r_[matrix.ExtractTranslation(),q.GetReal(),q.GetImaginary()])
                    if onboard is not None and (step%args.camera_save_every==0 or step==args.steps):
                        onboard.record(step,states[-1])
                if args.record_camera:
                    from srb.visual.replay_capture import ReplayVisualCapture
                    onboard=ReplayVisualCapture(env,args.output/'onboard_capture',env.scene['cam_onboard'],dt,
                                                stream=args.stream_camera,readback_device=args.camera_readback_device)
                    onboard.settle(4)
                capture(0)
        if env:
            physical_initial=direct_physx_state13(env,com)
            expected_initial=initial_state(args,com)
            compared_initial=physical_initial.copy()
            if np.dot(compared_initial[6:10],expected_initial[6:10])<0:
                compared_initial[6:10]*=-1
            initial_matches=bool(np.allclose(compared_initial,expected_initial,rtol=0.,atol=1e-5))
            save(args.output/'initial_state_check.json',{'requested_state13':expected_initial,
                'measured_state13':states[0], 'direct_physx_state13':physical_initial,
                'maximum_absolute_error':float(np.max(np.abs(compared_initial-expected_initial))),
                'absolute_tolerance':1e-5,'accepted':initial_matches,
                'stage':'before_first_force','visual_setup_complete':True,'direct_physx_checked':True})
            if not initial_matches:
                raise RuntimeError('Direct PhysX state after visual setup does not match the randomized initial state')
        if gimbal:record_gimbal(0)
        diagnosis=json.loads((args.output/'actuation_diagnostics.json').read_text())
        diagnosis.update(actuation_diagnostics(b,[0.,0.,-1.]))
        diagnosis['rcs_matrix']=b
        save(args.output/'actuation_diagnostics.json',diagnosis)
        np.savez_compressed(args.output/'reference_states.npz',
            states=reference_states(args,geometry,np.arange(args.steps+1)*dt,com),time_s=np.arange(args.steps+1)*dt)
        expert=NasaPairExpert(params,executor,code_export_root=args.output/'acados_generated')
        if env:
            import carb
        save(args.output/'physics_audit.json',{'execution_backend':'CPU PhysX' if env else 'offline discrete matched model',
            'kit_device':args.device if env else None,'env_device':str(env.device) if env else None,
            'physics_only':args.physics_only,'video':args.video,'record_camera':args.record_camera,
            'camera_readback_device':args.camera_readback_device if args.record_camera else None,
            'kit_worker_settings':{path:carb.settings.get_settings().get(path) for path in KIT_THREAD_SETTINGS} if env else None,
            'environment_mean_motion_rad_s':env.cfg.mean_motion_rad_s if env else 0.,
            'orbital_tidal_acceleration':env.cfg.orbital_tidal_acceleration if env else False,
            'step_dt_s':dt,'horizon_steps':args.horizon_steps,'execute_steps':args.execute_steps,
            'nozzle_count':len(powers),'virtual_nozzle_count':len(actor.get('virtual_thrusters',[])),
            'rcs_matrix':b,'angular_damping':.05,'linear_damping':0.,'gyroscopic':True,
            'initial_state13':states[0],'initial_mass_kg':physical_mass,'initial_fuel_kg':fuel,
            'initial_inertia_B_kg_m2':numpy(view.get_inertias()).reshape(3,3) if env else physical_inertia,
            'initial_com_pose_xyzw':numpy(view.get_coms())[0] if env else None,
            'dry_inertia_B_kg_m2':numpy(term._dry_inertias).reshape(3,3) if env else actor['inertia_B_kg_m2'],
            'com_B_m':com,'dry_mass_kg':executor.dry_mass,'burn_kg_per_duty_step':executor.burn_per_duty,
            'parameter_basis':physics['parameter_basis'],'original_geometry_preserved':True})
        if args.profile_loop:
            import cProfile
            loop_profiler=cProfile.Profile()
            loop_profiler.enable()
        for k in range(args.steps):
            if env:
                physical_mass=float(numpy(view.get_masses()).ravel()[0]);physical_inertia=numpy(view.get_inertias()).reshape(3,3)
                fuel=float(numpy(term.remaining_fuel).ravel()[0])
            physical=np.r_[physical_mass,physical_inertia.ravel(),np.linalg.inv(physical_inertia).ravel(),b.ravel()]
            if k%args.execute_steps==0:
                planning_started=time.perf_counter()
                refs=reference_states(args,geometry,(k+np.arange(args.horizon_steps+1))*dt,com)
                pending=expert.solve(states[-1],physical,fuel,previous,refs,warm_duties=warm)
                shooting_states.append(pending.shooting_state.copy());solver_references.append(refs.copy());solver_physics.append(physical.copy())
                support_gaps=full_mesh_plane_gaps(pending.predicted_state,com,geometry['actor_support_points'],geometry['target_support_points'],geometry['target_quaternion'])
                pending.constraint_margins['full_mesh_plane_gap_m']=float(np.min(support_gaps))
                if pending.accepted and np.min(support_gaps)<.02:
                    pending.accepted=False;pending.failure_reason='predicted full-mesh plane gap below 0.02 m'
                if gimbal is not None:
                    framing=framing_trace(gimbal,pending.predicted_state,com,target_camera_points,target_camera_center,
                        camera_k,args.camera_resolution,dt,args.camera_framing_margin)
                    pending.constraint_margins['camera_edge_margin_px']=framing['minimum_edge_margin_px']
                    if args.camera_fov_gate and pending.accepted and not framing['accepted']:
                        pending.accepted=False;pending.failure_reason='predicted full-target camera framing failed'
                planner_times.append(time.perf_counter()-planning_started)
                solve_times.append(pending.runtime_s);forecast_x.append(pending.predicted_state);forecast_u.append(pending.predicted_duty)
                records.append({'step':k,'status':pending.status,'accepted':pending.accepted,'reason':pending.failure_reason,'margins':pending.constraint_margins})
                if not pending.accepted:
                    stopped='NMPC quality gate rejected command';break
                consumed=min(args.execute_steps,args.horizon_steps)
                warm=np.vstack((pending.predicted_duty[consumed:],np.repeat(pending.predicted_duty[-1:],consumed,axis=0)))
            duty=pending.predicted_duty[k%args.execute_steps].copy()
            corrected=np.clip(duty,0.,1.)
            if np.max(np.abs(corrected-duty))>1e-6:
                raise RuntimeError('solver duty exceeds actuator bounds beyond numerical tolerance')
            duty=corrected
            if env:
                output=env.step(torch.as_tensor(duty[None,:],dtype=torch.float32,device=env.device))
                x=measured()
                force=float(torch.linalg.vector_norm(env.scene['replay_contacts'].data.net_forces_w,dim=-1).max().item())
                max_contact=max(max_contact,force)
                if force>1e-4:stopped='contact detected'
                if bool(output[2].any() or output[3].any()):stopped='episode ended or reset'
                physical_mass=float(numpy(view.get_masses()).ravel()[0]);fuel=float(numpy(term.remaining_fuel).ravel()[0])
                telemetry.append({key:numpy(value) for key,value in term.last_execution.items()})
            else:
                extended=executor.step(np.r_[states[-1],physical_mass,fuel],duty,physical)
                x,physical_mass,fuel=extended[:13],extended[13],extended[14]
                physical_inertia=np.asarray(actor['inertia_B_kg_m2'])*physical_mass/actor['mass_kg']
            actions.append(duty);previous=duty
            states.append(x.copy());current_mass_fuel.append([physical_mass,fuel])
            if gimbal:
                gimbal.advance(x,com,target_camera_center,dt)
                record_gimbal(k+1)
                if args.camera_fov_gate and not gimbal_records[-1]['framing_valid']:
                    stopped='measured full-target camera framing failed'
            if writer:capture(k+1)
            if np.linalg.norm(x[:3])<=params.koz_radius_m:stopped='measured center entered target KOZ'
            if full_mesh_plane_gaps(x[None],com,geometry['actor_support_points'],geometry['target_support_points'],geometry['target_quaternion'])[0]<.02:
                stopped='measured full-mesh plane gap below 0.02 m'
            sample=plane_sample(x,com,geometry['actor_bottom'],geometry['target_top'],geometry['target_quaternion'])
            good_streak=good_streak+1 if goal_sample_is_good(sample,x,args.surface_gap) else 0
            if args.stop_on_success and not stopped and goal_hold_complete(good_streak,dt,fuel):
                success_terminated=True
                break
            if k%100==0:print(f'NASA pair step={k} position={x[:3]}',flush=True)
            if stopped:break
        states=np.asarray(states);actions=np.asarray(actions).reshape(-1,len(powers))
        samples=[plane_sample(x,com,geometry['actor_bottom'],geometry['target_top'],geometry['target_quaternion']) for x in states]
        series={key:np.asarray([s[key] for s in samples]) for key in samples[0]}
        series['mesh_plane_gap_m']=full_mesh_plane_gaps(states,com,geometry['actor_support_points'],geometry['target_support_points'],geometry['target_quaternion'])
        good=np.asarray([goal_sample_is_good(sample,state,args.surface_gap) for sample,state in zip(samples,states)])
        streak,longest=0,0
        for value in good:
            streak=streak+1 if value else 0;longest=max(streak,longest)
        hot=np.asarray(solve_times[1:]);hold=max(0,longest-1)*dt
        planner_hot=np.asarray(planner_times[1:])
        np.savez_compressed(args.output/'trajectory.npz',states=states,duties=actions,time_s=np.arange(len(states))*dt,mass_fuel=current_mass_fuel)
        np.savez_compressed(args.output/'surface_metrics.npz',**series,success_sample=good)
        np.savez_compressed(args.output/'predictions.npz',states=forecast_x,duties=forecast_u,
                            shooting_states=shooting_states,references=solver_references,physical_parameters=solver_physics)
        if telemetry:
            np.savez_compressed(args.output/'executor_telemetry.npz',**{k:np.asarray([r[k] for r in telemetry]) for k in telemetry[0]})
        save(args.output/'solver_records.json',records)
        tracking_success=(len(actions)==args.steps or success_terminated) and not stopped and hold>=1. and fuel>1e-6
        save(args.output/'result.json',{'mode':args.mode,'execution_backend':'CPU PhysX' if env else 'offline discrete model',
            'requested_steps':args.steps,'executed_steps':len(actions),'tracking_success':tracking_success,
            'complete_success':bool(env is not None and tracking_success),'physics_contact_checked':env is not None,
            'termination':'goal_reached' if success_terminated else ('failure' if stopped else 'step_limit'),
            'stop_on_success':args.stop_on_success,'final_fuel_kg':fuel,
            'stopped_reason':stopped,'longest_surface_hold_s':hold,'final_plane_metrics':samples[-1],
            'maximum_contact_force_N':max_contact if env else None,
            'minimum_full_mesh_plane_gap_m':float(series['mesh_plane_gap_m'].min()),
            'minimum_center_koz_clearance_m':float(np.linalg.norm(states[:,:3],axis=1).min()-params.koz_radius_m),
            'solver_first_s_including_build':solve_times[0] if solve_times else None,
            'solver_hot_p50_ms':float(np.percentile(hot,50)*1000) if len(hot) else None,
            'solver_hot_p99_ms':float(np.percentile(hot,99)*1000) if len(hot) else None,
            'replan_deadline_s':args.execute_steps*dt,
            'solver_replan_deadline_miss_fraction':float(np.mean(hot>args.execute_steps*dt)) if len(hot) else None,
            'planner_hot_p50_ms':float(np.percentile(planner_hot,50)*1000) if len(planner_hot) else None,
            'planner_hot_p99_ms':float(np.percentile(planner_hot,99)*1000) if len(planner_hot) else None,
            'planner_replan_deadline_miss_fraction':float(np.mean(planner_hot>args.execute_steps*dt)) if len(planner_hot) else None,
            'camera_framing_gate':args.camera_fov_gate,
            'camera_minimum_edge_margin_px':min(r['edge_margin_px'] for r in gimbal_records) if gimbal_records else None,
            'camera_all_measured_frames_valid':all(r['framing_valid'] for r in gimbal_records) if gimbal_records else None,
            'success_criteria':'gap error <2 cm, lateral error <2 cm, facing error <5 deg, speed/angular speed <0.05 for >=1 s',
            'safety_scope':'target center KOZ, full-mesh supporting-plane checks at predicted/measured nodes, discrete PhysX contact stop; not continuous collision checking',
            'parameter_basis':physics['parameter_basis'],'flight_dynamics_validated':False})
        return 0 if not stopped else 1
    except Exception as exc:
        save(args.output/'runtime_error.json',{'type':type(exc).__name__,'message':str(exc),'requested_steps':args.steps,'executed_steps':len(actions)})
        if len(states):
            np.savez_compressed(args.output/'partial_trajectory.npz',states=np.asarray(states),duties=np.asarray(actions).reshape(-1,len(powers)))
        raise
    finally:
        if loop_profiler is not None:
            loop_profiler.disable()
            loop_profiler.dump_stats(str(args.output/'loop_profile.prof'))
        if gimbal_records:
            save(args.output/'gimbal_states.json',gimbal_records)
            np.savez_compressed(args.output/'gimbal_states.npz',**{
                key:np.asarray([row[key] for row in gimbal_records]) for key in gimbal_records[0]})
        if planner_times:
            np.savez_compressed(args.output/'planning_times.npz',solver_s=solve_times,total_planner_s=planner_times)
        try:
            if writer:
                try:
                    writer.close()
                except Exception as exc:
                    save(args.output/'video_error.json',{'type':type(exc).__name__,'message':str(exc)})
                    raise
                finally:
                    np.savez_compressed(args.output/'video_frames.npz',control_step=video_steps,time_s=np.asarray(video_steps)*dt,
                                        camera_eye_lookat=video_poses,actor_root_pose_w=video_actor_poses,target_root_pose_w=video_target_poses,
                                        usd_actor_root_pose_w=video_usd_actor_poses,usd_target_root_pose_w=video_usd_target_poses,
                                        camera_intrinsics_usd=video_camera_intrinsics)
                    save(args.output/'video_config.json',{'fps':1/dt,'resolution':[1280,720],'frames':len(video_steps),
                        'frame_source':'env.render() from measured CPU PhysX poses at time 0 and after each executed control step',
                        'timestamp_source':'executed control step multiplied by 0.05 s; nominal physical time',
                        'physics_only':False,'use_fabric':False,'stable_lighting':True,'original_materials_preserved':True,
                        'camera_horizontal_fov_deg':60.,'camera_mode':'fixed',
                        'camera_eye':fixed_camera_pose[:3],'camera_lookat':fixed_camera_pose[3:],
                        'camera_framing':'fixed eye, lookat and focal length throughout the approach',
                        'camera_intrinsics_columns':['focal_length','horizontal_aperture','vertical_aperture'],
                        'camera_intrinsics_unit':'tenths of a scene unit',
                        'stage_meters_per_unit':UsdGeom.GetStageMetersPerUnit(env.sim.stage),
                        'added_spacecraft_components':False,
                        'derived_target_component_added':True,
                        'onboard_camera_added':args.record_camera,
                        'recorded_step_range':[video_steps[0],video_steps[-1]] if video_steps else None})
        finally:
            try:
                if onboard is not None:onboard.finalize()
            finally:
                if env:env.close()


if __name__=='__main__':
    raise SystemExit(main())
