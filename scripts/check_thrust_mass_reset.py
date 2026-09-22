"""Real-PhysX regression for wet-mass initialization and indexed resets.

Called by validate_nmpc_replay.py --check-mass-reset with two environments.
Removing mass/inertia synchronization from init/reset must fail this check.
"""
import numpy as np


def check_mass_reset(env, constructor_properties):
    import torch
    term=env.action_manager._terms[env._thrust_action_term_key]
    view=env._robot.root_physx_view
    def array(x):
        return x.detach().cpu().numpy().copy()
    dry=array(env._robot.data.default_mass)
    dry_j=array(env._robot.data.default_inertia)
    wet=dry+term.cfg.fuel_capacity
    wet_j=dry_j*(wet/dry)
    np.testing.assert_allclose(constructor_properties[0],wet,rtol=2e-7,atol=1e-6,
        err_msg='Constructor must install wet mass before the first env.reset')
    np.testing.assert_allclose(constructor_properties[1],wet_j,rtol=2e-6,atol=1e-7)
    def full_check():
        np.testing.assert_allclose(array(view.get_masses()),wet,rtol=2e-7,atol=1e-6,
            err_msg='Fuel is full but PhysX still has dry/previous mass')
        np.testing.assert_allclose(array(view.get_inertias()),wet_j,rtol=2e-6,atol=1e-7,
            err_msg='Wet inertia must be restored from the immutable dry baseline')
        np.testing.assert_allclose(array(term.remaining_fuel),term.cfg.fuel_capacity)
    full_check()
    initial_mass=array(view.get_masses()).tolist()
    for _ in range(3):
        term.reset()
        full_check()
    term.record_execution=True
    action=torch.full((2,term.action_dim),.2,device=env.device)
    env.step(action)
    burned_mass=array(view.get_masses())
    burned_j=array(view.get_inertias())
    burned_fuel=array(term.remaining_fuel)
    assert np.all(burned_mass<wet)
    np.testing.assert_allclose(array(term.last_execution['mass_before']),wet,rtol=2e-7,atol=1e-6)
    np.testing.assert_allclose(burned_mass-dry,burned_fuel[:,None],rtol=2e-7,atol=1e-6)
    term.reset([0])
    np.testing.assert_allclose(array(view.get_masses())[0],wet[0],rtol=2e-7)
    np.testing.assert_allclose(array(view.get_inertias())[0],wet_j[0],rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(array(view.get_masses())[1],burned_mass[1])
    np.testing.assert_array_equal(array(view.get_inertias())[1],burned_j[1])
    np.testing.assert_array_equal(array(term.remaining_fuel)[1],burned_fuel[1])
    before_empty=array(view.get_masses())
    term.reset([])
    np.testing.assert_array_equal(array(view.get_masses()),before_empty)
    term.reset(torch.tensor([1],device=env.device))
    full_check()
    # Exercise the complete environment reset lifecycle on depleted values,
    # not just on properties already restored by the direct term resets.
    env.step(action)
    assert np.all(array(view.get_masses())<wet)
    assert np.all(array(term.remaining_fuel)<term.cfg.fuel_capacity)
    env.reset()
    full_check()
    return {'passed':True,'num_envs':2,'dry_mass_kg':dry.tolist(),'initial_wet_mass_kg':initial_mass,
            'mass_after_pulse_kg':burned_mass.tolist(),
            'checks':['constructor before first reset','initial env reset','repeated full reset','first force uses wet mass',
                      'partial list reset preserves other env','empty reset','tensor-index reset','env reset after burn']}
