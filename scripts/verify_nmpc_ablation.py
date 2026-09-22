"""Verify experiment artifacts without running or changing any controller."""
import json
from pathlib import Path
import sys
import numpy as np
from run_nmpc_horizon_comparison import audit_run
from analyze_nmpc_ablation import summarize


def main(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    combo=json.loads((root/'combination_selection.json').read_text())['case']
    cases=protocol['cases']+[combo]
    outcomes=[]; initial=None; B=None
    for name,rate,weight in cases:
        for stage,count in [('debug',30),('full',1200)]:
            folder=root/stage/name
            r=audit_run(folder,40,3,count,require_complete=stage=='debug')
            status=json.loads((root/f'{stage}_{name}_status.json').read_text())
            assert status['returncode']==0
            assert not (folder/'runtime_error.json').exists()
            p=json.loads((folder/'controller_params.json').read_text())
            c=json.loads((folder/'smoothness_config.json').read_text())
            a=json.loads((folder/'physics_audit.json').read_text())
            assert c['duty_rate_per_s']==rate and c['jerk_weight']==weight and c['jerk_scale_m_s3']==.1
            assert p['mean_motion_rad_s']==0 and a['mean_motion_srb_rad_s']==0
            assert p['duty_rate_per_s']==rate
            assert p['position_cost']==10 and p['velocity_cost']==1 and p['duty_rate_cost']==.01
            assert r['seed']==0 and r['physics_only'] and r['max_contact_force_N']==0
            assert r['min_koz_clearance_m']>0
            s=summarize(folder)
            assert s['slew_violation_max']<=1e-6
            assert s['boundary_acceleration_error_max']==0
            assert s['executor_duty_error_max']<=1e-6
            assert abs(s['initial_force_mass_kg']-11.108697891235352)<1e-7
            if initial is None:
                initial=r['initial_state']; B=a['actual_rcs_matrix']
            np.testing.assert_array_equal(r['initial_state'],initial)
            np.testing.assert_array_equal(a['actual_rcs_matrix'],B)
            if stage=='full':
                outcomes.append({'case':name,'steps':r['steps'],'complete_success':r['complete_success'],
                                 'first30_complete':s['first30_complete']})
    out={'passed':True,'formal_cases':len(cases),'debug_cases':len(cases),
         'invariants':['same initial state and actual B','wet mass','n=0','40/3','fixed cost and seed',
                       'executed accepted prefixes only','actual duty slew','actual boundary acceleration',
                       'positive KOZ and zero contacts'], 'outcomes':outcomes}
    (root/'artifact_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main(sys.argv[1])
