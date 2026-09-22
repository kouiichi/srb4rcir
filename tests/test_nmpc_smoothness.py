"""Check raw sampled derivative conventions and metrics."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('smoothness',Path(__file__).resolve().parents[1]/'scripts/analyze_nmpc_smoothness.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_constant_acceleration_has_zero_jerk():
    dt=.05; t=np.arange(10)*dt
    v=t[:,None]*np.array([1.,2.,3.])[None,:]
    a,j,stats=module.smoothness(v,np.zeros((9,16)),dt)
    np.testing.assert_allclose(a,np.tile([1,2,3],(9,1)),atol=1e-12)
    np.testing.assert_allclose(j,0,atol=1e-12)
    np.testing.assert_allclose(stats['acceleration']['norm_rms'],np.sqrt(14))

def test_linear_acceleration_has_known_jerk():
    dt=.05; t=np.arange(10)*dt
    v=.5*t[:,None]**2*np.array([1.,-2.,3.])[None,:]
    _,j,_=module.smoothness(v,np.zeros((9,16)),dt)
    np.testing.assert_allclose(j,np.tile([1,-2,3],(8,1)),atol=1e-12)
