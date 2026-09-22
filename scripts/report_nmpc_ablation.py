"""Write reproducible experiment report from recorded ablation metrics."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main(root):
    root=Path(root).resolve()
    rows=json.loads((root/'metrics.json').read_text())
    commands=[json.loads(line) for line in (root/'commands.jsonl').read_text().splitlines()]
    combo=json.loads((root/'combination_selection.json').read_text())
    lines=['# NMPC 占空比变化率与世界系 jerk 单因素／组合实验', '',
        '## 实验问题与共同设置', '',
        '已批准的8组：基线(rate20, jerk0)，rate4/rate2/rate1，jerk权重0.001/0.01/0.1（rate20），最后按预先规则选择组合。',
        '任务：真实SRB闭环接近/悬停。N=40、K=3、dt=0.05 s、预测2 s、每0.15 s重规划，保留原动作分块(3,3,3,3,2,2,2,2)×2。初始actor位置[-3,0.2,0.15] m，目标actor位置[-1.5,0,0] m；两者单位姿态，初始线速度/角速度0。控制与评价采用COM参考。seed=0。',
        'Kit设备cuda:1（本机Bus81），CUDA_VISIBLE_DEVICES未设置，CUDA_DEVICE_ORDER=PCI_BUS_ID；环境/PhysX张量cpu，physics-only，无渲染，无计算延迟注入；OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1。',
        '模型：n=0自由漂浮、实际RCS B矩阵/COM/全惯量、在线质量燃料演化；干质量6.108697891 kg、初始湿质量11.108697891 kg、燃料5 kg、16路0.1 N推进器。湿惯量按质量比例恢复；实际参数详见每组physics_audit.json/controller_params.json。旋转积分仍为近似。',
        '每组先30步debug再1200步/60秒正式尝试。debug失败则保留并标记该配置未完成，不能当作平滑性改进；成功标准：完成全部步骤，无求解/质量门失败或安全停止，同时误差<0.1 m、速度<0.05 m/s、姿态<5度连续至少1秒。KOZ半径0.5 m，记录最小余量/接触力。', '',
        '## 代价定义与计量口径', '',
        '加速度 a[k]=(v[k+1]-v[k])/dt；jerk j[k]=(a[k]-a[k-1])/dt，世界系COM。新增归一化残差 j/0.1；OCP运行项按acados标准1/2×dt×weight×残差平方累加，终端跟踪代价不变。weight=0不增加加速度记忆状态。',
        'jerk组首节点使用上一个实际执行区间的加速度，后续节点通过状态传递前一区间预测加速度；初次点火前取0。重规划仍每3步一次，jerk和变化率计算均用0.05秒。',
        '统计使用原始速度差分、不滤波：全程、固定前30秒、去前1秒，以及前30秒内去前1秒。RMS为三轴向量范数RMS。正式jerk统计仅比较两个已测区间；零初始加速度到首次点火的等效采样jerk另列。ZOH控制下这些数值不是连续时间峰值jerk。',
        '占空比总变差TV=Σk,i |u[k,i]-u[k-1,i]|，主表包括初始零指令；另存排除点火TV。大变化为单通道单步变化>0.1+1e-6，小变化计数阈值为1e-3；变化次数增加可能只是将大跳变拆成多个小变化。',
        '当前首动作块长3、执行K=3，边界最大增量仍是rate×0.05，因此较小rate在此分块下会产生更慢的有效爬升。所有组共享这个因素。',
        '求解时间包括参数更新、求解和独立质量检查；首次构建单列，其余报告热P50/P99。150ms预算对应K3重规划，50ms预算也保留于原始记录；同步仿真不证明实时调度可行。', '',
        '## 结果', '',
        '| 组 | 执行/请求 | 成功 | 前30秒jerk RMS | 全程jerk RMS | jerk峰值 | duty TV | 大变化次数 | 最终持续悬停起点s | 燃料kg |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return '—' if v is None else f'{v:.5g}'
    for r in rows:
        j=r['jerk_first30']['norm_rms']
        lines.append(f'| {r["case"]} | {r["steps"]}/{r["requested_steps"]} | {r["complete_success"]} | {fmt(j)}'+('（窗口未满）' if not r['first30_complete'] else '')+
            f' | {fmt(r["jerk_full"]["norm_rms"])} | {fmt(r["jerk_full"]["norm_max"])} | {fmt(r["duty_tv_with_ignition"])} | {r["channel_changes_over_010"]} | {fmt(r["final_hold_start_s"])} | {fmt(r["fuel_used_kg"])} |')
    lines+=['', '| 组 | 超调m | 最终位置误差m | KOZ余量m | 接触N | 求解P50/P99 ms | >150ms比例 | 停止/失败 |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        reason=r.get('attempt_failure') or r['failure'] or r['stopped_reason'] or '无'
        if isinstance(reason,dict):
            reason=f'step {reason.get("step")}: {reason.get("reason")}'
        lines.append(f'| {r["case"]} | {fmt(r["approach_direction_overshoot_m"])} | {fmt(r["final_position_error_m"])} | {fmt(r["min_koz_clearance_m"])} | {fmt(r["max_contact_force_N"])} | {fmt(r["solver_hot_p50_ms"])}/{fmt(r["solver_hot_p99_ms"])} | {fmt(r["solver_replan_deadline_miss_fraction"])} | {reason} |')
    lines+=['', '失败组全程指标仅覆盖停止前的记录，不能与60秒成功轨迹作为同长度结果直接排序。其“最终持续悬停起点”指直到记录结束的连续悬停，并不保证之后保持到60秒。前30秒是各组共同观测窗口。', '',
            '| 组 | 前30秒燃料kg | 首次点火等效采样jerk | 去前1秒jerk峰值 | >0.001占空比变化次数 |',
            '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["case"]} | {fmt(r.get("fuel_used_first30_kg"))} | {fmt(r.get("ignition_sampled_jerk_norm"))} | {fmt(r["jerk_after1"]["norm_max"])} | {r.get("channel_changes_over_001")} |')
    lines+=['', '## 组合选择', '', '选择在正式结果前已写入implementation plan：完整成功且安全，在最终持续悬停起点≤基线+5秒、燃料≤基线1.1倍的候选内，以前30秒jerk RMS最小分别选择两个因素；没有通过折中筛选者才退回完整成功/安全集合。无成功候选时仅选最弱水平作诊断，不作为推荐。', '',
            '```json',json.dumps(combo,ensure_ascii=False,indent=2),'```', '', '## 实际执行一致性', '',
            '| 组 | 最大变化率超限量(duty) | 实测加速度与规划边界误差 | 一步速度预测最大误差m/s |',
            '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["case"]} | {fmt(r["slew_violation_max"])} | {fmt(r["boundary_acceleration_error_max"])} | {fmt(r.get("predicted_next_velocity_error_max"))} |')
    lines+=['', '## 原始文件与图', '',
            '每组保存replay/executor/predictions NPZ、controller_params/physics_audit/smoothness_config JSON、三轴velocity/acceleration/jerk CSV和包含16路duty/机体系与世界系推力/机体系力矩的commands.csv。预测轨迹图为抽样2秒预测叠加实际轨迹，所有预测数据保存在predictions.npz。', '',
            '![指标对比](comparison.png)', '![变化率曲线](slew_curves.png)','![jerk代价曲线](jerk_curves.png)', '']
    for r in rows:
        rel=Path(r['folder']).relative_to(root)
        if r['steps']:
            lines.append(f'- {r["case"]}：[三轴曲线]({rel}/axis_curves.png) · [推进器指令]({rel}/commands.png) · [预测与执行轨迹]({rel}/trajectory.png) · [全部指标]({rel}/smoothness_metrics.json)')
        else:
            lines.append(f'- {r["case"]}：无正式轨迹，失败仍保留；实际执行步数未知，不能用debug轨迹替代。')
    lines+=['', '## 复现命令与日志', '', f'工作目录：`{ROOT}`；所有目录均为新建，重跑时必须使用新的输出目录。']
    for record in commands:
        lines+=['', f'### {record["stage"]}/{record["case"]}', '', '```sh',record['command'],'```',
                f'输出：`{record["output"]}`；日志：`{record["log"]}`。']
    lines+=['', '## 调试、警告和边界', '',
            '新增功能测试曾因缺失jerk_weight接口失败；实现后焦点测试18项通过，原有其余回归29项通过。报告错误处理测试3项先失败再通过，总计50项。独立控制器审查未发现阻断问题；第二次实验脚本审查指出失败记录处理、审计资格、缺失末尾悬停和报告ROOT问题，均已修复。真实debug与正式实验结果分别保留，不混入计时统计。',
            '最终artifact_verification.json核查8组debug及8组正式记录：初态/B矩阵/湿质量/模型/权重/seed一致，仅执行通过门控的规划前缀，占空比变化率无超限，重规划边界使用真实上一间隔加速度。运行验证：`./srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh scripts/verify_nmpc_ablation.py logs/nmpc_smoothness_ablation_20260917`。',
            '测试命令均使用wrapper和Isaac Python：`-m pytest -q --disable-warnings tests/test_nmpc_jerk_cost.py tests/test_nmpc_execution_prefix.py`；`-m pytest -q --disable-warnings tests/test_nmpc_core.py tests/test_nmpc_orbital.py tests/test_nmpc_matched_executor.py tests/test_nmpc_smoothness.py`；`-m pytest -q tests/test_nmpc_ablation_reporting.py`。',
            '已知环境警告：headless GLFW/GLInterop、CasADi3.8未被acados官方测试以及acados弃用提示。Kit退出码可能为0但Python运行失败，因此同时检查runtime_error.json和replay结果。',
            '这是单一初态、单seed、固定N/K和动作分块下的确定性对照；重复相同seed不构成独立随机样本。尚未验证不同初态、扰动、执行延迟、真实阀门PWM/最小脉宽或相机路径。',
            '没有修改默认rate20/s、jerk0，也没有用后处理滤波修饰控制输出或曲线。KOZ半径仍为实验默认，未按真实车辆包络重新标定。',
            '下游使用这些数据时应使用真实执行状态/指令，而不是将所有预测动作当作实际已执行动作。']
    findings=root/'findings.md'
    if findings.exists():
        lines[2:2]=[findings.read_text(), '']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main(sys.argv[1])
