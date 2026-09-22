"""Serial real SRB experiments for the approved slew versus jerk ablation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from run_nmpc_horizon_comparison import audit_run

ROOT = Path(__file__).resolve().parents[1]
CASES = [('baseline',20.,0.),('rate4',4.,0.),('rate2',2.,0.),('rate1',1.,0.),
         ('jerk001',20.,.001),('jerk01',20.,.01),('jerk1',20.,.1)]


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(root, case, stage, steps):
    name,rate,weight=case
    folder=root/stage/name
    logfile=root/f'{stage}_{name}.log'
    argv=['env','-u','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER=PCI_BUS_ID',
          'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','timeout','-k','15','600',
          str(ROOT/'srb_0.0.5_env.sh'),'--','/home/ubuntu/isaac-sim-4.5/python.sh',
          '-u','scripts/validate_nmpc_replay.py','replay','--plan',
          'logs/nmpc_validation_20260907/approach_hcw_plan','--output',str(folder),
          '--physics-only','--device','cuda:1','--closed-loop','--align-goal',
          '--matched-executor','--seed','0','--steps',str(steps),
          '--horizon-steps','40','--execute-steps','3',
          '--duty-rate',str(rate),'--jerk-weight',str(weight),'--jerk-scale','0.1']
    record={'case':name,'stage':stage,'cwd':str(ROOT),'argv':argv,
            'command':shlex.join(argv),'output':str(folder),'log':str(logfile)}
    # Existing output is never overwritten, including incomplete attempts.
    if folder.exists() or logfile.exists():
        raise FileExistsError(f'Use a new experiment directory: {folder}')
    with (root/'commands.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    print(f'START {stage} {name} rate={rate} jerk={weight}',flush=True)
    with logfile.open('w') as f:
        completed=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
    record['returncode']=completed.returncode
    try:
        r=audit_run(folder,40,3,steps)
        record['steps']=r['steps']
        record['success']=r['complete_success']
        record['failure']=r['failure'] or r['stopped_reason']
        record['debug_pass']=r['steps']==steps and record['failure'] is None and completed.returncode==0
    except (OSError,ValueError,AssertionError) as exc:
        record.update(debug_pass=False,success=False,failure=str(exc))
    write(root/f'{stage}_{name}_status.json',record)
    print(f'DONE {stage} {name}: steps={record.get("steps")} success={record["success"]} failure={record["failure"]}',flush=True)
    return record


def choose_combo(root):
    from analyze_nmpc_ablation import attempt_summary
    baseline=attempt_summary(root,'baseline')
    if not baseline['complete_success']:
        raise RuntimeError('No independently validated successful baseline for combination selection')
    choices={}
    selected=[]
    for factor,cases in [('rate',CASES[1:4]),('jerk',CASES[4:7])]:
        eligible=[]; safe=[]
        for case in cases:
            folder=root/'full'/case[0]
            if not (folder/'replay.json').exists():
                continue
            s=attempt_summary(root,case[0])
            if not s['complete_success'] or s['min_koz_clearance_m']<=0 or s['max_contact_force_N']>1e-4:
                continue
            entry=(s['jerk_first30']['norm_rms'],case,s)
            safe.append(entry)
            if (s['final_hold_start_s'] is not None and baseline['final_hold_start_s'] is not None
                    and s['final_hold_start_s']<=baseline['final_hold_start_s']+5
                    and s['fuel_used_kg']<=baseline['fuel_used_kg']*1.1):
                eligible.append(entry)
        pool=eligible or safe
        if pool:
            _,case,s=min(pool,key=lambda row:row[0])
            choices[factor]={'case':case[0],'tradeoff_filter_passed':bool(eligible),
                             'first30_jerk_rms':s['jerk_first30']['norm_rms']}
        else:
            case=cases[0]
            choices[factor]={'case':case[0],'diagnostic_only':True,'reason':'no complete safe candidate'}
        selected.append(case)
    combo=('combined',selected[0][1],selected[1][2])
    write(root/'combination_selection.json',{'rule':'lowest first30s jerk RMS; complete safe; final hold <= baseline+5s; fuel <= baseline*1.1; fallback disclosed',
                                           'choices':choices,'case':combo})
    return combo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--phase',choices=['single','combo'],default='single')
    args=parser.parse_args(); root=args.root.resolve()
    if args.phase=='single':
        root.mkdir(parents=True,exist_ok=False)
        write(root/'protocol.json',{'cases':CASES,'jerk_scale_m_s3':.1,'horizon':40,'execute':3,
              'dt':.05,'seed':0,'debug_steps':30,'full_steps':1200,
              'selection_plan':str(ROOT/'docs/superpowers/plans/2026-09-17-nmpc-smoothness-ablation.md')})
        cases=CASES
    else:
        cases=[choose_combo(root)]
    for case in cases:
        debug=run(root,case,'debug',30)
        if debug['debug_pass']:
            run(root,case,'full',1200)
        else:
            print(f'SKIP full {case[0]}: debug failure retained for investigation',flush=True)


if __name__=='__main__':
    main()
