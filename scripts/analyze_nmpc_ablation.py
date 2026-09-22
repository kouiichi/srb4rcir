"""Unfiltered, common-window motion and command analysis of SRB ablations."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):
    return json.loads(path.read_text())


def stats(values):
    if not len(values):
        return {key:None for key in ('axis_rms','axis_max','norm_rms','norm_max','norm_p95','norm_p99')}
    norms=np.linalg.norm(values,axis=1)
    return {'axis_rms':np.sqrt(np.mean(values**2,axis=0)).tolist(),
            'axis_max':np.max(np.abs(values),axis=0).tolist(),
            'norm_rms':float(np.sqrt(np.mean(norms**2))),'norm_max':float(norms.max()),
            'norm_p95':float(np.percentile(norms,95)),'norm_p99':float(np.percentile(norms,99))}


def validated_attempt(root, name):
    status_path=Path(root)/f'full_{name}_status.json'
    if not status_path.exists() or (Path(root)/'full'/name/'runtime_error.json').exists():
        return False
    status=read(status_path)
    return status.get('returncode')==0 and status.get('debug_pass',False)


def attempt_summary(root, name):
    """Reconcile formal artifacts and independent audit; never substitute debug."""
    root=Path(root); folder=root/'full'/name
    status_path=root/f'full_{name}_status.json'
    status=read(status_path) if status_path.exists() else {}
    error_path=folder/'runtime_error.json'
    error=read(error_path).get('message') if error_path.exists() else status.get('failure')
    if not error and status.get('returncode') not in (None,0):
        code=status['returncode']
        error=f'Process exit {code}' + (' (timeout)' if code==124 else '')
    if not error and not status:
        error='Formal attempt has no process/audit status record'
    if not error and status.get('debug_pass') is False:
        error='Formal attempt failed completion or independent audit'
    if (folder/'replay.json').exists() and (folder/'replay.npz').exists():
        s=summarize(folder)
        s['recorded_complete_success']=s['complete_success']
        s['complete_success']=s['complete_success'] and validated_attempt(root,name)
    else:
        nullable=['duty_rate_per_s','jerk_weight','first30_available_s','duty_tv_with_ignition',
            'duty_max_jump_actual','channel_changes_over_010','channel_changes_over_001',
            'final_hold_start_s','fuel_used_kg','approach_direction_overshoot_m',
            'integrated_position_error_m_s','min_koz_clearance_m','max_contact_force_N',
            'solver_hot_p50_ms','solver_hot_p99_ms','solver_replan_deadline_miss_fraction',
            'slew_violation_max','boundary_acceleration_error_max','final_position_error_m']
        s=dict.fromkeys(nullable)
        s.update(case=name,folder=str(folder.resolve()),steps=None,requested_steps=1200,
            complete_success=False,recorded_complete_success=False,first30_complete=False,
            failure=error or 'No formal trajectory saved; inspect debug/status/logs',stopped_reason=None)
        for key in ('jerk_full','jerk_first30','jerk_after1'):
            s[key]=stats([])
    s['validated_attempt']=validated_attempt(root,name)
    s['attempt_failure']=error
    return s


def summarize(folder):
    folder=Path(folder)
    r=read(folder/'replay.json')
    cfg=read(folder/'smoothness_config.json')
    d=np.load(folder/'replay.npz')
    dt=read(folder/'physics_audit.json')['step_dt_s']
    x,u,ref=d['states'],d['duties'],d['refs']
    n=len(u); t=np.arange(n+1)*dt
    acceleration=np.diff(x[:,3:6],axis=0)/dt
    jerk=np.diff(acceleration,axis=0)/dt
    tj=np.arange(1,n)*dt
    # Include the initially zero duty in actuator slew auditing. Motion jerk
    # is measured between actual intervals; ignition is reported separately.
    du=np.diff(np.vstack((np.zeros(16),u)),axis=0)
    dq=np.abs(du)
    s={**r,**cfg,'case':folder.name,'folder':str(folder.resolve()),
       'first30_complete':n*dt>=30,'first30_available_s':min(n*dt,30.),
       'acceleration':stats(acceleration),'jerk_full':stats(jerk),
       'jerk_first30':stats(jerk[tj<30]),'jerk_after1':stats(jerk[tj>=1]),
       'jerk_first30_after1':stats(jerk[(tj>=1)&(tj<30)]),
       'ignition_sampled_jerk_norm':float(np.linalg.norm(acceleration[0])/dt) if n else None,
       'duty_tv_with_ignition':float(dq.sum()),'duty_tv_without_ignition':float(dq[1:].sum()),
       'duty_tv_first30':float(dq[:int(30/dt)].sum()),
       'duty_max_jump_actual':float(dq.max()) if n else None,
       'channel_changes_over_001':int((dq>1e-3).sum()),
       'channel_changes_over_010':int((dq>.1+1e-6).sum()),
       'updates_over_010':int((dq>.1+1e-6).any(axis=1).sum()),
       'slew_violation_max':float(max(0.,dq.max()-cfg['duty_rate_per_s']*dt)) if n else None,
       'fuel_used_kg':5-float(np.asarray(r['fuel_remaining_kg']).ravel()[0]),
       'final_hold_start_s':r['duration_s']-r['final_hover_hold_s'] if r['final_hover_hold_s']>=1 else None}
    counts,edges=np.histogram(dq,bins=[0,1e-6,.001,.01,.05,.1+1e-6,.2+1e-6,.5,1.00001])
    s['duty_jump_histogram']={'edges':edges.tolist(),'counts':counts.tolist()}
    error=np.linalg.norm(x[:,:3]-ref[:,:3],axis=1)
    s['integrated_position_error_m_s']=float(np.trapz(error,t))
    direction=ref[0,:3]-x[0,:3]
    direction/=max(np.linalg.norm(direction),1e-15)
    s['approach_direction_overshoot_m']=float(max(0,np.max((x[:,:3]-ref[:,:3])@direction)))
    q=x[:,6:10]/np.linalg.norm(x[:,6:10],axis=1)[:,None]
    qref=ref[:,6:10]/np.linalg.norm(ref[:,6:10],axis=1)[:,None]
    angles=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(q*qref,axis=1)),0,1)))
    s['max_attitude_error_deg']=float(angles.max())
    good=(error<.1)&(np.linalg.norm(x[:,3:6],axis=1)<.05)&(angles<5)
    starts=[k for k in range(max(0,len(good)-20)) if good[k:k+21].all()]
    s['first_hover_start_s']=starts[0]*dt if starts else None
    s['boundary_acceleration_error_max']=None
    if (folder/'predictions.npz').exists():
        pred=np.load(folder/'predictions.npz'); steps=pred['steps'].astype(int)
        if len(steps):
            expected=np.array([acceleration[k-1] if k else np.zeros(3) for k in steps])
            s['boundary_acceleration_error_max']=float(np.max(np.abs(expected-pred['previous_acceleration_W'])))
            valid=pred['accepted']&(steps<n)
            s['predicted_next_velocity_error_max']=float(np.linalg.norm(pred['states'][valid,1,3:6]-x[steps[valid]+1,3:6],axis=1).max()) if valid.any() else None
    if n and (folder/'executor.npz').exists():
        e=np.load(folder/'executor.npz')
        for key in ('force_B','torque_B'):
            y=e[key].reshape(n,3)
            dy=np.diff(np.vstack((np.zeros(3),y)),axis=0)
            s[key+'_total_variation']=float(np.linalg.norm(dy,axis=1).sum())
            s[key+'_slew']=stats(dy/dt)
        s['initial_force_mass_kg']=float(e['mass_before'].ravel()[0])
        s['executor_duty_error_max']=float(np.max(np.abs(e['duty'].reshape(n,16)-u)))
        s['fuel_used_first30_kg']=5-float(e['fuel_after'].reshape(n)[int(30/dt)-1]) if n*dt>=30 else None
    return s


def draw_case(folder,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from srb.nmpc.quaternion import quaternion_to_rotation
    d=np.load(folder/'replay.npz'); e=np.load(folder/'executor.npz')
    n=len(d['duties']); dt=.05
    x,u=d['states'],d['duties']; v=x[:,3:6]
    a=np.diff(v,axis=0)/dt; j=np.diff(a,axis=0)/dt
    t=np.arange(n+1)*dt; ta=(np.arange(n)+.5)*dt; tj=np.arange(1,n)*dt
    f=e['force_B'].reshape(n,3); torque=e['torque_B'].reshape(n,3)
    fw=np.array([quaternion_to_rotation(q)@force for q,force in zip(x[:,6:10],f)])
    np.savez_compressed(folder/'smoothness.npz',time_state_s=t,time_acceleration_s=ta,
        time_jerk_s=tj,time_command_s=t[:-1],velocity_W=v,acceleration_W=a,jerk_W=j,
        force_B=f,force_W=fw,torque_B=torque,duty=u)
    for filename,ts,values,labels in [('velocity',t,v,'vx_m_s,vy_m_s,vz_m_s'),
            ('acceleration',ta,a,'ax_m_s2,ay_m_s2,az_m_s2'),('jerk',tj,j,'jx_m_s3,jy_m_s3,jz_m_s3'),
            ('commands',t[:-1],np.c_[f,fw,torque,u],
             'Fx_B_N,Fy_B_N,Fz_B_N,Fx_W_N,Fy_W_N,Fz_W_N,Tx_B_Nm,Ty_B_Nm,Tz_B_Nm,'+','.join(f'duty_{k}' for k in range(16)))]:
        np.savetxt(folder/f'{filename}.csv',np.c_[ts,values],delimiter=',',header='time_s,'+labels,comments='')
    fig,axes=plt.subplots(4,3,figsize=(15,11),sharex=True)
    for row,(times,y,label) in enumerate([(t,v,'Velocity [m/s]'),(ta,a,'Acceleration [m/s²]'),
                                        (tj,j,'Jerk [m/s³]'),(t[:-1],f,'Body force [N]')]):
        for col in range(3):
            ax=axes[row,col]
            if baseline is not None:
                bx,bu,bf=baseline; bt=np.arange(len(bx))*dt
                ba=np.diff(bx[:,3:6],axis=0)/dt; bj=np.diff(ba,axis=0)/dt
                bts=[bt,(np.arange(len(bu))+.5)*dt,np.arange(1,len(bu))*dt,bt[:-1]][row]
                by=[bx[:,3:6],ba,bj,bf][row]
                ax.plot(bts,by[:,col],color='.65',lw=.7,label='baseline',alpha=.7)
            ax.plot(times,y[:,col],lw=.9,label=folder.name)
            ax.set_ylabel(label); ax.grid(alpha=.25)
            ax.set_xlim(0,60)
            if row==0:
                ax.set_title('XYZ'[col])
            if row==3:
                ax.set_xlabel('Time [s]')
    axes[0,0].legend(); fig.suptitle(f'{folder.name} — N40/K3 — raw measured COM motion')
    fig.tight_layout(); fig.savefig(folder/'axis_curves.png',dpi=145); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    im=axes[0].imshow(u.T,origin='lower',aspect='auto',extent=[0,n*dt,-.5,15.5],vmin=0,vmax=1,cmap='viridis')
    axes[0].set_ylabel('Thruster'); fig.colorbar(im,ax=axes[0],label='Duty')
    axes[1].step(t[:-1],f,where='post'); axes[1].set_ylabel('Body force [N]')
    axes[2].step(t[:-1],torque,where='post'); axes[2].set(ylabel='Body torque [Nm]',xlabel='Time [s]')
    fig.suptitle(folder.name); fig.tight_layout(); fig.savefig(folder/'commands.png',dpi=145); plt.close(fig)
    pred=np.load(folder/'predictions.npz'); accepted=np.flatnonzero(pred['accepted'])
    fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(*x[:,:3].T,lw=2,label='Executed COM')
    for index in accepted[::max(1,len(accepted)//18)]:
        ax.plot(*pred['states'][index,:,:3].T,color='tab:orange',alpha=.5,lw=.7)
    ax.plot([],[],[],color='tab:orange',label='Sampled 2s predictions')
    ax.scatter(*x[0,:3],color='green',label='Initial'); ax.scatter(*d['refs'][0,:3],color='black',marker='x',label='Goal')
    ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title=folder.name)
    ax.legend(); fig.tight_layout(); fig.savefig(folder/'trajectory.png',dpi=145); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    args=parser.parse_args(); root=args.root.resolve()
    from run_nmpc_smoothness_ablation import CASES
    names=[c[0] for c in CASES]+['combined']
    rows=[]
    bd=np.load(root/'full/baseline/replay.npz'); be=np.load(root/'full/baseline/executor.npz')
    baseline=(bd['states'],bd['duties'],be['force_B'].reshape(-1,3))
    for name in names:
        folder=root/'full'/name
        s=attempt_summary(root,name); rows.append(s)
        if folder.exists():
            (folder/'smoothness_metrics.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
        if s['steps']:
            draw_case(folder,None if name=='baseline' else baseline)
    (root/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    flat=[]
    for row in rows:
        fields=['case','steps','complete_success','duty_rate_per_s','jerk_weight','first30_complete',
                'validated_attempt','attempt_failure',
                'duty_tv_with_ignition','duty_max_jump_actual','channel_changes_over_010','channel_changes_over_001',
                'final_hold_start_s','fuel_used_kg','fuel_used_first30_kg','approach_direction_overshoot_m','integrated_position_error_m_s',
                'min_koz_clearance_m','max_contact_force_N','solver_hot_p50_ms','solver_hot_p99_ms',
                'solver_replan_deadline_miss_fraction','slew_violation_max','boundary_acceleration_error_max']
        f={key:row.get(key) for key in fields}
        f.update({key+'_'+metric:row[key][metric] for key in ('jerk_full','jerk_first30','jerk_after1') for metric in ('norm_rms','norm_max','norm_p99')})
        flat.append(f)
    with (root/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#777777','#90b9d8','#417fad','#114e84','#c6a1db','#9653b1','#622d82','#d85738'][:len(rows)]
    labels={'baseline':'Baseline','rate4':'4/s','rate2':'2/s','rate1':'1/s',
            'jerk001':'J=0.001','jerk01':'J=0.01','jerk1':'J=0.1','combined':'2/s + J=0.001'}
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,key,title in [(axes[0,0],'jerk_first30','First 30s jerk norm RMS [m/s³]'),
                         (axes[0,1],'duty_tv_first30','First 30s duty total variation (includes ignition)'),
                         (axes[1,0],'final_hold_start_s','Hover onset within recorded interval [s]'),
                         (axes[1,1],'fuel_used_first30_kg','First 30s fuel used [kg]')]:
        for index,row in enumerate(rows):
            value=row[key]['norm_rms'] if key=='jerk_first30' else row.get(key)
            if value is not None:
                bar=ax.bar(index,value,color=colors[index],alpha=1 if row['complete_success'] else .4)
                if not row['complete_success']:
                    bar[0].set_hatch('//')
                ax.annotate(f'{value:.3g}',(index,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
            else:
                ax.text(index,0,'N/A',ha='center',va='bottom')
        ax.set_xticks(range(len(rows)),[labels[r['case']] for r in rows],rotation=30)
        ax.set_title(title); ax.grid(axis='y',alpha=.25); ax.margins(y=.18)
    fig.suptitle('N40/K3 matched SRB ablation — hatched bars: incomplete/failed runs')
    fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=160); plt.close(fig)
    for family,chosen in [('slew',['baseline','rate4','rate2','rate1']),('jerk',['baseline','jerk001','jerk01','jerk1','combined'])]:
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for row in rows:
            if row['case'] not in chosen or not row['steps']:
                continue
            folder=Path(row['folder']); d=np.load(folder/'replay.npz'); x=d['states']; n=len(d['duties'])
            j=np.diff(x[:,3:6],n=2,axis=0)/.05**2
            axes[0].plot(np.arange(n+1)*.05,np.linalg.norm(x[:,3:6],axis=1),label=labels[row['case']],lw=1)
            axes[1].plot(np.arange(1,n)*.05,np.linalg.norm(j,axis=1),label=labels[row['case']],lw=.8)
            axes[2].plot(np.arange(n+1)*.05,np.linalg.norm(x[:,:3]-d['refs'][:,:3],axis=1),label=labels[row['case']],lw=1)
        for ax,title in zip(axes,['Speed [m/s]','Jerk norm [m/s³]','Position error [m]']):
            ax.set_ylabel(title); ax.legend(); ax.grid(alpha=.25)
        axes[-1].set_xlabel('Time [s]'); fig.suptitle(f'{family} single-factor comparison')
        fig.tight_layout(); fig.savefig(root/f'{family}_curves.png',dpi=150); plt.close(fig)
    for row in flat:
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
