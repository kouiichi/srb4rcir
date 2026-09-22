"""Check discrete executor predictions against saved independent pulse data."""
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srb.nmpc.matched_executor import MatchedExecutor
from srb.nmpc.quaternion import quaternion_to_rotation

def audit(folder):
    folder=Path(folder)
    a=json.loads((folder/'physics_audit.json').read_text())
    d=np.load(folder/'replay.npz'); e=np.load(folder/'executor.npz')
    x=d['states'].copy()
    if 'state_position' not in a:
        x[:,:3]+=np.array([quaternion_to_rotation(q)@a['com_m'] for q in x[:,6:10]])
    model=MatchedExecutor(float(np.asarray(a['dry_mass_kg']).ravel()[0]),
        a['fuel_consumption_rate']*a['agent_rate']*np.asarray(a['thruster_power']),
        a['step_dt_s'],a['angular_damping'])
    errors=[]
    for k,u in enumerate(d['duties']):
        mass=float(e['mass_before'][k].ravel()[0]); J=e['inertia_before'][k].reshape(3,3)
        p=np.r_[mass,J.ravel(),np.linalg.inv(J).ravel(),np.asarray(a['actual_rcs_matrix']).ravel()]
        y=model.step(np.r_[x[k],mass,e['fuel_before'][k].ravel()[0]],u,p)
        truth=np.r_[x[k+1],e['mass_after'][k].ravel()[0],e['fuel_after'][k].ravel()[0]]
        errors.append(y-truth)
    errors=np.asarray(errors)
    report={}
    for name,sl in [('position_m',slice(0,3)),('velocity_m_s',slice(3,6)),('quaternion',slice(6,10)),('omega_rad_s',slice(10,13)),('mass_kg',slice(13,14)),('fuel_kg',slice(14,15))]:
        norms=np.linalg.norm(errors[:,sl],axis=1)
        report[name]={'rms':float(np.sqrt(np.mean(norms**2))),'max':float(max(norms))}
    (folder/'one_step_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    audit(sys.argv[1])
