#!/usr/bin/env python3
"""Add measured telemetry to an actual surface-approach recording and plot it."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--goal-gap-m', type=float, default=.3,
                        help='Requested bus-face gap for subtitles and plot pass band')
    args = parser.parse_args()
    if not np.isfinite(args.goal_gap_m) or args.goal_gap_m <= 0:
        parser.error('--goal-gap-m must be finite and positive')
    gap_cm = args.goal_gap_m * 100
    if args.output.exists():
        raise FileExistsError(args.output)
    data = np.load(args.folder/'replay.npz')
    surface = np.load(args.folder/'surface_metrics.npz')
    result = json.loads((args.folder/'replay.json').read_text())
    dt = json.loads((args.folder/'physics_audit.json').read_text())['step_dt_s']
    video = json.loads((args.folder/'video_config.json').read_text())
    width, height = video['resolution']
    def timestamp(t):
        ticks = round(t*100)
        return f'{ticks//360000}:{ticks//6000%60:02}:{ticks//100%60:02}.{ticks%100:02}'
    subtitle = args.folder/'surface_telemetry.ass'
    lines = [f'''[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Header,DejaVu Sans,23,&H00FFFFFF,&H00FFFFFF,&H00201810,&H00201810,0,0,0,0,100,100,0,0,3,2,0,8,16,16,12,1
Style: Telemetry,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H00201810,&H00201810,0,0,0,0,100,100,0,0,3,2,0,2,14,14,12,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text''']
    duration = len(data['duties'])*dt
    lines.append(f'Dialogue: 0,0:00:00.00,{timestamp(duration)},Header,,0,0,0,,NMPC | Target-relative surface approach | N40 K3'
                 r'\N{\fs16}Blue: actor (+Z top)   Gold: target (-X front)   Green: facing bus surfaces')
    hold = 0
    sync_duration = float(result['surface_reference_sync_s'])
    for frame in range(len(data['duties'])):
        # Frame 0 records state 1 (the first actual PhysX step), not state 0.
        i = frame+1
        t = i*dt
        hold = hold+1 if surface['success_sample'][i] else 0
        hold_s = max(0,hold-1)*dt
        phase = 'SYNC APPROACH' if t <= sync_duration else 'HOLD'
        status = 'HOLD QUALIFIED' if hold_s >= 1 else 'TRACKING'
        text = (f't={t:05.2f} s  |  {phase}  |  {status}'
                rf'\NProjected bus-face gap: {100*surface["face_gap_m"][i]:.2f} cm (goal: {gap_cm:g} cm)'
                f'   Facing error: {surface["facing_error_deg"][i]:.3f} deg'
                rf'\NFull-mesh plane separation: {100*surface["mesh_plane_gap_m"][i]:.2f} cm'
                f'   Lateral error: {100*surface["lateral_error_m"][i]:.2f} cm   Hold: {hold_s:.2f} s')
        lines.append(f'Dialogue: 0,{timestamp(frame*dt)},{timestamp((frame+1)*dt)},Telemetry,,0,0,0,,{text}')
    subtitle.write_text('\n'.join(lines)+'\n')
    # Argument list keeps all paths outside the shell. ffmpeg filter paths
    # escape its own colon/apostrophe grammar independently.
    filter_path = str(subtitle.resolve()).replace('\\', '\\\\').replace(':', '\\:').replace("'", "'\\''")
    command = ['ffmpeg','-v','warning','-i',str(args.folder/'replay.mp4'),'-vf',f"ass='{filter_path}'",
               '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(args.output)]
    subprocess.run(command, check=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.arange(len(data['states']))*dt
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(times,surface['face_gap_m']*100)
    axes[0,0].axhspan(gap_cm-2,gap_cm+2,color='green',alpha=.2)
    axes[0,0].set(ylabel='Projected bus-face gap [cm]',title=f'Goal: {gap_cm:g} cm +/- 2 cm')
    axes[0,1].plot(times,surface['facing_error_deg'])
    axes[0,1].axhline(5,color='gray',ls='--')
    axes[0,1].set(ylabel='Facing error [deg]',title='Actor +Z opposes target -X')
    axes[1,0].plot(times,surface['mesh_plane_gap_m']*100)
    axes[1,0].axhline(2,color='red',ls='--')
    axes[1,0].set(ylabel='Full-mesh plane separation [cm]',title='Includes panels and antennas; guard: 2 cm')
    axes[1,1].plot(times,surface['lateral_error_m']*100,label='Lateral error [cm]')
    axes[1,1].plot(times,np.linalg.norm(data['states'][:,3:6],axis=1)*100,label='COM speed [cm/s]')
    axes[1,1].legend()
    for ax in axes.flat:
        ax.set(xlabel='Simulated time [s]')
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.folder/'surface_summary.png',dpi=150)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
