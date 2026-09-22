"""Unfiltered finite-difference smoothness and recorded executor analysis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from srb.nmpc.quaternion import quaternion_to_rotation

def smoothness(v, u, dt=.05):
    a=np.diff(v,axis=0)/dt
    j=np.diff(a,axis=0)/dt
    def stats(y):
        n=np.linalg.norm(y,axis=1)
        return dict(axis_rms=np.sqrt(np.mean(y*y,axis=0)).tolist(),
                    axis_abs_max=np.max(np.abs(y),axis=0).tolist(),
                    norm_rms=float(np.sqrt(np.mean(n*n))),norm_max=float(max(n)),
                    norm_p95=float(np.percentile(n,95)),norm_p99=float(np.percentile(n,99)))
    du=np.diff(u,axis=0)
    return a,j,dict(acceleration=stats(a),jerk=stats(j),
        duty_total_variation=float(np.abs(du).sum()),duty_max_jump=float(np.max(np.abs(du))),
        duty_slew_rms=float(np.sqrt(np.mean((du/dt)**2))),
        fraction_duty_at_upper_bound=float(np.mean(u>=1-1e-6)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args(); folder=args.folder
    d=np.load(folder/'replay.npz'); e=np.load(folder/'executor.npz')
    audit=json.loads((folder/'physics_audit.json').read_text())
    dt=audit['step_dt_s']; x=d['states']; u=d['duties']; n=len(u)
    t=np.arange(n+1)*dt; ta=(np.arange(n)+.5)*dt; tj=np.arange(1,n)*dt
    a,j,s=smoothness(x[:,3:6],u,dt)
    f=e['force_B'].reshape(n,3); torque=e['torque_B'].reshape(n,3)
    fw=np.array([quaternion_to_rotation(q)@force for q,force in zip(x[:,6:10],f)])
    data=dict(time_state_s=t,time_acceleration_s=ta,time_jerk_s=tj,time_command_s=t[:-1],
        velocity_W_m_s=x[:,3:6],acceleration_W_m_s2=a,jerk_W_m_s3=j,
        force_B_N=f,force_W_N=fw,torque_B_Nm=torque,duty=u,
        mass_before_kg=e['mass_before'].reshape(n),mass_after_kg=e['mass_after'].reshape(n),
        fuel_after_kg=e['fuel_after'].reshape(n))
    np.savez_compressed(folder/'smoothness.npz',**data)
    def csv(name,times,values,labels):
        np.savetxt(folder/name,np.c_[times,values],delimiter=',',header='time_s,'+labels,comments='')
    csv('velocity.csv',t,x[:,3:6],'vx_m_s,vy_m_s,vz_m_s')
    csv('acceleration.csv',ta,a,'ax_m_s2,ay_m_s2,az_m_s2')
    csv('jerk.csv',tj,j,'jx_m_s3,jy_m_s3,jz_m_s3')
    csv('commands.csv',t[:-1],np.c_[f,torque,u],
        'Fx_B_N,Fy_B_N,Fz_B_N,Tx_B_Nm,Ty_B_Nm,Tz_B_Nm,'+','.join(f'duty_{i}' for i in range(16)))
    old=np.load(args.baseline/'replay.npz'); ox=old['states']; ou=old['duties']
    oa,oj,os=smoothness(ox[:,3:6],ou,dt)
    s['force_slew_norm_rms_N_s']=float(np.sqrt(np.mean(np.sum((np.diff(f,axis=0)/dt)**2,axis=1))))
    s['B_duty_vs_submitted_wrench_max_abs']=float(np.max(np.abs(u@np.asarray(audit['actual_rcs_matrix']).T-np.c_[f,torque])))
    pred=np.load(folder/'predictions.npz'); steps=pred['steps'].astype(int)
    valid=(steps<n)&pred['accepted']
    errors=pred['states'][valid,1,:]-x[steps[valid]+1]
    s['predicted_next_velocity_error_max_m_s']=float(np.max(np.linalg.norm(errors[:,3:6],axis=1)))
    s['predicted_next_position_error_max_m']=float(np.max(np.linalg.norm(errors[:,:3],axis=1)))
    window=min(400,n,len(ou))
    summary={'matched_full':s,'historical_full':os,
        'common_window_seconds':window*dt,
        'matched_common_window':smoothness(x[:window+1,3:6],u[:window],dt)[2],
        'historical_common_window':smoothness(ox[:window+1,3:6],ou[:window],dt)[2],
        'definitions':{'acceleration':'(v[k+1]-v[k])/dt, interval midpoint',
                       'jerk':'(a[k+1]-a[k])/dt, t=(k+1)*dt',
                       'filter':'none','reference':'COM velocity in world axes',
                       'warning':'sampled jerk only; discontinuous ZOH acceleration has no finite continuous-time peak jerk'}}
    (folder/'smoothness_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    colors=['#2678b2','#dc7025','#2f9653']
    fig,axes=plt.subplots(5,3,figsize=(15,14),sharex=True)
    for col,axis in enumerate('XYZ'):
        for row,(ts,y,label) in enumerate([(t,x[:,3:6],'Velocity [m/s]'),(ta,a,'Acceleration [m/s²]'),
                       (tj,j,'Jerk [m/s³]'),(t[:-1],f,'Body force [N]'),(t[:-1],torque,'Body torque [Nm]')]):
            axes[row,col].plot(ts,y[:,col],color=colors[col],lw=1)
            axes[row,col].set_ylabel(label); axes[row,col].grid(alpha=.25)
        axes[0,col].set_title(axis); axes[-1,col].set_xlabel('Time [s]')
    fig.suptitle('Matched free-floating NMPC — N=40, K=1 — raw 20 Hz measurements')
    fig.tight_layout(); fig.savefig(folder/'axis_smoothness.png',dpi=155); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for ts,y,ots,oy,ax,label in zip([t,ta,tj],[x[:,3:6],a,j],
            [np.arange(len(ox))*dt,(np.arange(len(ou))+.5)*dt,np.arange(1,len(ou))*dt],
            [ox[:,3:6],oa,oj],axes,['Speed [m/s]','Acceleration norm [m/s²]','Jerk norm [m/s³]']):
        ax.plot(ots,np.linalg.norm(oy,axis=1),label='Historical HCW / nominal executor',alpha=.7)
        ax.plot(ts,np.linalg.norm(y,axis=1),label='Matched free-floating',lw=1.3)
        ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel('Time [s]'); fig.tight_layout()
    fig.savefig(folder/'smoothness_comparison.png',dpi=155); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True)
    im=axes[0].imshow(u.T,aspect='auto',origin='lower',extent=[0,n*dt,-.5,15.5],vmin=0,vmax=1,cmap='viridis')
    axes[0].set_ylabel('Thruster index'); fig.colorbar(im,ax=axes[0],label='Duty')
    axes[1].plot(t[:-1],data['mass_before_kg'],label='Mass at force submission')
    axes[1].plot(t[:-1],data['fuel_after_kg'],label='Fuel after step')
    axes[1].set(xlabel='Time [s]',ylabel='Mass [kg]'); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(folder/'commands_and_mass.png',dpi=155); plt.close(fig)
    fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(*x[:,:3].T,color='#2678b2',lw=2,label='Executed COM trajectory')
    accepted_indices=np.flatnonzero(valid)
    for index in accepted_indices[::max(1,len(accepted_indices)//15)]:
        ax.plot(*pred['states'][index,:,:3].T,color='#dc7025',alpha=.45,lw=.8)
    ax.plot([],[],[],color='#dc7025',label='Sampled 2 s NMPC predictions')
    ax.scatter(*x[0,:3],label='Initial COM',color='green')
    ax.scatter(*d['refs'][-1,:3],label='Target COM',color='black',marker='x')
    ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title='Matched free-floating NMPC: predicted and executed paths')
    ax.legend(); fig.tight_layout(); fig.savefig(folder/'predicted_and_executed.png',dpi=155); plt.close(fig)
    print(json.dumps(summary,indent=2))
    inv=json.loads((folder/'invocation.json').read_text())
    result=json.loads((folder/'replay.json').read_text())
    baseline_result=json.loads((args.baseline/'replay.json').read_text())
    report=['# 自由漂浮、实际执行器匹配 NMPC 平滑性实验', '',
        '## 设置与复现', '',
        f'工作目录：`{inv["cwd"]}`。seed=0；请求1200步/60秒；N=40（2秒），K=1，控制/物理周期0.05秒；',
        '优化分块为 `(3,3,3,3,2,2,2,2)*2`，不是一次执行多个动作。保持原代价权重，不增加滤波或jerk惩罚。',
        'Kit cuda:1（本机物理Bus-ID 81）；CUDA_VISIBLE_DEVICES取消；CUDA_DEVICE_ORDER=PCI_BUS_ID；SRB/PhysX tensors=CPU；physics-only=true。',
        '初始actor位置[-3,0.2,0.15]m，单位姿态、零线/角速度；actor目标[-1.5,0,0]m、单位姿态；KOZ半径0.5m。',
        '初始COM位置是actor位置加旋转后的COM偏置；目标也作相同变换。n_NMPC=n_SRB=0。', '',
        '正式实验命令（每次必须使用新输出目录）：', '', '```bash',
        'env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout -k 15 900 ./srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh -u '+inv['command']+' > /tmp/nmpc_matched_final_20260916.log 2>&1',
        '```', '',
        '分析命令：', '', '```bash',
        './srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh scripts/analyze_nmpc_smoothness.py '+str(folder)+' --baseline '+str(args.baseline),
        './srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh scripts/audit_matched_probe.py '+str(folder),
        '```', '',
        '## 匹配范围与限制', '',
        '读取实际16路推力方向、位置、功率、COM构建B；每次重规划读取实际质量和完整惯量，预测域传播燃料、质量和惯量缩放。',
        '实际执行器先提交力/力矩，后更新质量，因此当前步加速度使用更新前质量/惯量。初始PhysX质量只有干质量，首步后加入剩余燃料；本实验忠实匹配这个现有行为，没有修改执行器。',
        'COM平移采用PhysX半隐式步进；考虑角阻尼和陀螺项，旋转采用近似离散积分，不能宣称与PhysX逐位一致。一步残差见one_step_audit.json。',
        '燃料耗尽时动力学、预测wrench及KOZ制动能力均关断。模型匹配入口为--matched-executor，历史HCW入口保留用于复现，不表示所有旧默认配置已替换。',
        '本次对比同时改变动力学、参数、COM参考点、离散积分和热启动；不是仅改变n的单因素因果实验。', '',
        '## 原始数据与口径', '',
        'velocity.csv为世界坐标系COM速度；acceleration.csv为相邻速度差/0.05s，时间戳在区间中点；jerk.csv为相邻加速度差/0.05s。没有滤波。',
        'commands.csv为实际提交力、力矩和16路指令；executor.npz额外记录执行器实际处理的duty、质量、惯量和燃料；smoothness.npz也有世界系力。',
        'jerk统计的norm_rms是sqrt(mean(||j||²))；replay.json旧字段rms_jerk_m_s3是三个分量混合RMS，数值差sqrt(3)，不要混用。',
        '20Hz差分jerk只描述采样尺度的变化，不是零阶保持推力切换时的连续时间峰值jerk；未给定工程阈值，不能判定达到某个平滑性认证标准。',
        f'共同前段统计窗口为{window*dt:g}秒，避免只用长时间悬停稀释jerk。完整逐轴RMS/峰值及P95/P99见smoothness_metrics.json。', '',
        '## 实验结果', '', '| 指标 | 历史N40/K1 | 自由漂浮匹配 |','|---|---:|---:|']
    for label,key in [('执行步数','steps'),('最终位置误差m','final_position_error_m'),('最终速度m/s','final_relative_speed_m_s'),('最终姿态误差deg','final_attitude_error_deg'),('最长严格悬停s','longest_hover_hold_s'),('最小KOZ余量m','min_koz_clearance_m'),('最大接触力N','max_contact_force_N'),('求解调用P50 ms','solver_hot_p50_ms'),('求解调用P99 ms','solver_hot_p99_ms'),('50ms超期比例','solver_deadline_miss_fraction'),('完整成功','complete_success')]:
        report.append(f'| {label} | {baseline_result[key]} | {result[key]} |')
    for label,key,sub in [('加速度范数RMS m/s²','acceleration','norm_rms'),('jerk范数RMS m/s³','jerk','norm_rms'),('jerk范数峰值m/s³','jerk','norm_max')]:
        report.append(f'| {label} | {os[key][sub]:.8g} | {s[key][sub]:.8g} |')
    report+=['',f'正式运行停止原因：`{result["stopped_reason"]}`；failure=`{result["failure"]}`。',
        f'匹配组剩余燃料：{result["fuel_remaining_kg"]}kg，初始5kg。首次求解（含构建）：{result["solver_first_wall_s_including_build"]}s。',
        '严格悬停：位置误差<0.1m、速度<0.05m/s、姿态误差<5deg连续至少1秒；完整成功还要求全部步数执行且没有失败。SRB native success不是同一标准。',
        '位置/KOZ历史统计以actor原点为参考，新统计以COM为参考；新运行额外检查actor原点KOZ。速度两组均为COM，可直接按相同差分口径比较。KOZ未据航天器包络重新标定。',
        '同步仿真不注入计算延迟；solve_wall包括参数更新、求解和质量验证，非纯QP时间；不证明20Hz实时性或GPU PhysX。', '',
        '## 平滑性判断', '',
        f'全程jerk范数RMS变化 {(s["jerk"]["norm_rms"]/os["jerk"]["norm_rms"]-1)*100:.2f}%；占空比总变化量变化 {(s["duty_total_variation"]/os["duty_total_variation"]-1)*100:.2f}%。',
        f'共同前{window*dt:g}秒jerk RMS：历史{summary["historical_common_window"]["jerk"]["norm_rms"]:.6f}，匹配{summary["matched_common_window"]["jerk"]["norm_rms"]:.6f} m/s³。',
        '本次数据不支持“消除模型偏差即可显著改善平滑性”。匹配后仍有反向修正、推力饱和切换与加速度抖动；悬停后的平稳不代表接近过程平滑。',
        '当前duty_rate_per_s=20，在dt=0.05s下允许单步占空比跳变1，相当于没有比[0,1]幅值界更紧的变化率限制；位置/速度权重10/1，duty/delta-duty权重0.001/0.01，未加入jerk项。',
        '首步PhysX质量从6.1086979kg更新到11.1024475kg；当前实现的这个初始化不连续被忠实保留，会造成起始加速度变化，不能把它误归因于HCW。',
        '建议后续独立对比：收紧占空比变化率；增加实际世界系加速度差/jerk代价；设计可制动的速度参考及终端条件；单独修复并验证初始湿质量。此次没有同时修改这些控制目标。',
        '终端误差接近浮点数精度下限，只说明此确定性同步仿真收敛，不代表真实系统纳米级精度。', '',
        '## 调试记录与局限', '',
        'probe：48步独立16路小脉冲；debug_40_1：30步通过；full_40_1：248步因热启动相关QP失败停止（保留）。',
        '离线保存状态复现：旧warm失败而cold成功。修复后按当前实测质量/燃料重积分、重分块初值，原失败状态通过；debug_v2_300完成300步。',
        'full_v2_40_1为审查后主动中断的中间版本，不计正式结果；debug_v3_300在初始化时参数字段名错误，零执行步；已修正并增加异常落盘。debug_v3b_300为最终300步调试。',
        '审查修复：燃料耗尽安全约束、非单位目标姿态COM转换、静止参考轨迹的重复积分开销。最终实验采用修复版本。',
        '已知警告：headless GLFW/GLInterop；CasADi3.8未被acados官方测试及acados弃用提示。physics-only成功不证明相机或视频可用。',
        '仅单个初始条件、一个seed、一种N/K；不证明普适鲁棒性、非单位目标姿态大范围收敛、工程碰撞安全或控制已经足够平滑。', '',
        '![三轴曲线](axis_smoothness.png)', '![历史比较](smoothness_comparison.png)',
        '![推力指令与质量](commands_and_mass.png)','![预测和执行轨迹](predicted_and_executed.png)', '',
        '详细物理参数（包括完整惯量/B）见physics_audit.json、controller_params.json；原始预测见predictions.npz；运行环境见invocation.json。',
        '日志：/tmp/nmpc_matched_final_20260916.log；所有实验目录位于logs/nmpc_matched_smooth_20260916。']
    (folder/'report.md').write_text('\n'.join(report)+'\n')

if __name__=='__main__':
    main()
