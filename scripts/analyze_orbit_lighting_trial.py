"""Summarize the actual opt-in lighting trial and prepare inspectable images.

Run with the project wrapper/Isaac Python; original render PNGs are untouched.
"""
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srb.visual.replay_capture import write_json


def read(path):
    return json.loads(path.read_text())


def project(points, params):
    view = np.array(params['cameraViewTransform']).reshape(4, 4)
    projection = np.array(params['cameraProjection']).reshape(4, 4)
    points = np.atleast_2d(points)
    clip = np.c_[points, np.ones(len(points))] @ view @ projection
    xy = clip[:, :2]/clip[:, 3:4]
    width, height = params['renderProductResolution']
    return np.c_[(xy[:, 0]+1)*width/2, (1-xy[:, 1])*height/2]


def main(root):
    root = Path(root).resolve()
    run = root/'full1200'
    result, frames = read(run/'replay.json'), read(run/'capture_frames.json')
    config, lighting = read(run/'capture_config.json'), read(run/'orbit_lighting_config.json')
    obs, replay = np.load(run/'observations.npz'), np.load(run/'replay.npz')
    audit, controller = read(run/'physics_audit.json'), read(run/'controller_params.json')
    steps = [f['capture_id'] for f in frames]
    quality = {
        'frame_count': len(frames),
        'target_missing_steps': [f['capture_id'] for f in frames if f['target_roi']['pixel_count'] == 0],
        'mesh_cropped_steps': [f['capture_id'] for f in frames if not f['target_mesh_projection']['complete_in_frame']],
        'target_dark_steps': [f['capture_id'] for f in frames if (f['target_roi']['underexposed_fraction'] or 0) > .95],
        'target_saturated_steps': [f['capture_id'] for f in frames if (f['target_roi']['overexposed_fraction'] or 0) > .05],
        'max_camera_position_error_m': max(f['camera_position_error_m'] for f in frames),
        'max_camera_orientation_error_deg': max(f['camera_orientation_error_deg'] for f in frames),
        'max_state13_recording_difference': float(np.max(np.abs(obs['measured_state13']-replay['states']))),
        'max_sensor_episode_time_difference_s': max(abs(f['sensor_time_s']-f['time_s']) for f in frames),
        'max_physics_episode_time_difference_s': max(abs(f['physics_time_s']-frames[0]['physics_time_s']-f['time_s']) for f in frames),
        'thresholds': read(run/'capture_checks.json')['thresholds'],
        'visibility_scope': 'semantic visible pixels plus full mesh frustum bounds; no isolated-target visibility reference, no all-surface visibility claim',
    }
    quality['target_present_fraction'] = 1-len(quality['target_missing_steps'])/len(frames)
    quality['mesh_complete_fraction'] = 1-len(quality['mesh_cropped_steps'])/len(frames)
    quality['all_frames_pass_combined_candidate_gate'] = (
        not any(quality[k] for k in ('target_missing_steps','mesh_cropped_steps','target_dark_steps','target_saturated_steps'))
        and quality['max_camera_position_error_m'] < 5e-5 and quality['max_camera_orientation_error_deg'] < .05)
    quality['gate_limit'] = 'passing candidate gates does not establish legibility, SNR, feature tracking, or suitability for visual learning'
    for field in ('underexposed_fraction', 'overexposed_fraction', 'luma_p05_u8', 'luma_p50_u8', 'luma_p95_u8'):
        values = [f['target_roi'][field] for f in frames if f['target_roi'][field] is not None]
        quality[field+'_across_frames_p05_p50_p95'] = np.percentile(values, [5, 50, 95]).tolist()
    write_json(root/'quality_metrics.json', quality)

    before = read(run/'visual_physics_invariance.json')['before']
    after = read(run/'physics_after_run.json')
    properties = {}
    for name in ('_robot', '_target'):
        b, a = before['bodies'][name], after['bodies'][name]
        properties[name] = {'COM_unchanged': b['coms'] == a['coms'],
            'initial_mass_kg': b['masses'], 'final_mass_kg': a['masses']}
        if name == '_target':
            properties[name]['inertia_unchanged'] = b['inertias'] == a['inertias']
        else:
            ratio = np.array(a['masses'])/np.array(b['masses'])
            properties[name]['inertia_mass_scaling_max_error'] = float(np.max(np.abs(
                np.array(a['inertias'])-np.array(b['inertias'])*ratio)))
    collisions = {k:v for k,v in before['physics_attributes'].items()
                  if any(word in k.lower() for word in ('collision', 'approximation', 'contact', 'restoffset'))}
    invariance = {'bodies': properties,
        'collision_properties_unchanged': all(after['physics_attributes'].get(k) == v for k,v in collisions.items()),
        'actuator_unchanged': all(before[k] == after[k] for k in ('actuator_cfg','thruster_power','thruster_direction','thruster_offset')),
        'initial_fuel_kg': before['fuel'], 'final_fuel_kg': after['fuel'],
        'wet_mass_minus_dry_and_fuel_kg': float(np.array(after['bodies']['_robot']['masses']).item()-np.array(audit['dry_mass_kg']).item()-np.array(after['fuel']).item()),
        'render_settings_unchanged_during_control': read(run/'render_after_run.json')['settings'] == lighting['effective_runtime_settings']}
    write_json(root/'physical_property_verification.json', invariance)

    # Keyframes use exactly matching saved onboard/overview capture IDs.
    available = sorted(int(p.stem.split('_')[-1]) for p in (run/'overview').glob('frame_*.png')
                       if '_annotated' not in p.stem)
    available = sorted(set(available))
    representative = max((frames[steps.index(s)] for s in available
                          if frames[steps.index(s)]['target_mesh_projection']['complete_in_frame']
                          and frames[steps.index(s)]['target_roi']['pixel_count']>150),
                         key=lambda f:f['target_roi']['luma_p95_u8'])
    write_json(root/'representative.json', {
        'capture_id':representative['capture_id'], 'time_s':representative['time_s'],
        'selection':'brightest p95 among saved complete-mesh overview frames with >150 target pixels; representative only, all-run limitations retained',
        'target_roi':representative['target_roi']})
    errors = np.linalg.norm(replay['states'][:, :3]-replay['refs'][:, :3], axis=1)
    near = np.flatnonzero(errors < .1)
    near_step = int(near[0]) if len(near) else int(errors.argmin())
    near_saved = min(available, key=lambda x: abs(x-near_step))
    nominal = [s for s in (0, 100, near_saved, len(frames)//2, 800, frames[-1]['capture_id']) if s in available]
    nominal.append(representative['capture_id'])
    candidates = quality['mesh_cropped_steps'] + quality['target_missing_steps'] + quality['target_saturated_steps']
    if candidates:
        nominal += [candidates[len(candidates)//2]]
    selected = sorted(set(nominal))
    (root/'keyframes').mkdir(exist_ok=True)
    overview_params = read(run/'overview'/f'frame_{available[0]:04d}.json')['camera_params']
    for step in selected:
        if step not in available:
            continue
        frame = frames[steps.index(step)]
        np.save(root/'keyframes'/f'frame_{step:04d}_depth.npy', obs['depth'][steps.index(step)])
        np.save(root/'keyframes'/f'frame_{step:04d}_target_mask.npy', obs['target_mask'][steps.index(step)])
        write_json(root/'keyframes'/f'frame_{step:04d}.json', {'onboard': frame,
            'overview': read(run/'overview'/f'frame_{step:04d}.json')})
        onboard = Image.open(run/'onboard'/f'frame_{step:04d}.png').convert('RGB')
        external = Image.open(run/'overview'/f'frame_{step:04d}.png').convert('RGB')
        height = external.height
        onboard = onboard.resize((height, height), Image.Resampling.NEAREST)
        paired = Image.new('RGB', (height+external.width, height+36))
        paired.paste(onboard, (0,36)); paired.paste(external,(height,36))
        draw = ImageDraw.Draw(paired)
        draw.text((12,12), f'ONBOARD | capture {step} | episode {frame["time_s"]:.2f}s', fill='white')
        draw.text((height+12,12), 'EXTERNAL | same physical state | original render below', fill='white')
        paired.save(run/'paired'/f'frame_{step:04d}.png')
        annotated = external.copy(); draw = ImageDraw.Draw(annotated)
        points = obs['robot_root_state_w'][:steps.index(step)+1, :3]
        xy = project(points, overview_params)
        if len(xy)>1:
            draw.line([tuple(x) for x in xy], fill=(30,160,220), width=2)
        phi = np.linspace(0,2*np.pi,120)
        circle = project(np.c_[.5*np.cos(phi), .5*np.sin(phi), np.zeros(len(phi))], overview_params)
        draw.line([tuple(x) for x in circle], fill=(170,80,80), width=1)
        for label, pos in [('ACTOR',frame['robot_root_state_w'][:3]),('TARGET',frame['target_root_state_w'][:3])]:
            x,y = project(pos, overview_params)[0]
            draw.ellipse((x-4,y-4,x+4,y+4),outline='white');draw.text((x+7,y-10),label,fill='white')
        draw.text((12,12), f'ANNOTATED COPY | {frame["time_s"]:.2f}s | KOZ center radius 0.5m (experiment default)', fill='white')
        draw.text((12,29), f'Sun direction to source: {np.round(lighting["directions"]["sun_to_source"],3)}', fill='white')
        draw.text((12,46), 'View eye (-1.5,-6,3)m; cyan: actual actor-root trajectory', fill='white')
        annotated.save(run/'overview'/f'frame_{step:04d}_annotated.png')
    write_json(root/'keyframes/index.json', {'capture_ids':selected,
        'raw_onboard_dir': str(run/'onboard'), 'raw_overview_dir':str(run/'overview'),
        'paired_dir':str(run/'paired'), 'onboard_preview_resampling':'nearest, raw originals unchanged'})
    comparison = Image.new('RGB', (1200, 284*3))
    draw = ImageDraw.Draw(comparison)
    for i, name in enumerate(('original','sun_only','sun_earth')):
        onboard = Image.open(run/'static'/f'{name}_onboard.png').convert('RGB')
        overview = Image.open(run/'static'/f'{name}_overview.png').convert('RGB').resize((int(960*256/540),256))
        comparison.paste(onboard,(0,284*i+28));comparison.paste(overview,(280,284*i+28))
        draw.text((8,284*i+8), f'{name} | fixed initial state, identical camera/material/exposure; control steps=0',fill='white')
    comparison.crop((0,0,280+int(960*256/540),852)).save(root/'lighting_comparison.png')

    commands, attempts = [], []
    prefix = 'env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'invocation.json').exists():
            continue
        inv = read(folder/'invocation.json'); timeout = 2400 if folder.name=='full1200' else 600 if folder.name.startswith('smoke') else 420
        command = f'{prefix} timeout -k {20 if timeout==2400 else 10} {timeout} ./srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh -u {shlex.join(inv["argv"])} > {shlex.quote(str(root/(folder.name+".log")))} 2>&1'
        commands.append({'name':folder.name,'cwd':inv['cwd'],'command':command,'output':str(folder),'log':str(root/(folder.name+'.log'))})
        rr = read(folder/'replay.json') if (folder/'replay.json').exists() else {}
        error = read(folder/'runtime_error.json') if (folder/'runtime_error.json').exists() else None
        checks = read(folder/'capture_checks.json') if (folder/'capture_checks.json').exists() else None
        attempts.append({'name':folder.name,'executed_control_steps':rr.get('steps',0),
            'runtime_error':error,'capture_checks':checks,'replay_failure':rr.get('failure')})
    write_json(root/'commands.json',commands);write_json(root/'attempts.json',attempts)
    (root/'commands.sh').write_text('# Reproduction commands from exact recorded process argv plus wrapper/device prefix.\n# Timeout bounds reconstructed for diagnostics; final full/smoke bounds are exact.\n# Do not rerun in existing output directories.\n'+'\n\n'.join(c['command'] for c in commands)+'\n')
    hot = replay['solve_wall_s'][1:]
    summary = {k:result[k] for k in ('steps','requested_steps','duration_s','complete_success','failure','stopped_reason',
        'final_position_error_m','final_relative_speed_m_s','final_attitude_error_deg','longest_hover_hold_s','final_hover_hold_s',
        'min_koz_clearance_m','max_contact_force_N','solver_hot_p50_ms','solver_hot_p99_ms','solver_replan_deadline_miss_fraction')}
    summary.update(deadline_misses_150ms=int(sum(hot>.15)), hot_solve_count=len(hot), keyframes=selected)
    write_json(root/'experiment_summary.json',summary)
    print(json.dumps({'summary':summary,'quality':{k:quality[k] for k in ('frame_count','target_present_fraction','mesh_complete_fraction','max_camera_position_error_m','max_camera_orientation_error_deg')},'invariance':invariance},indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
