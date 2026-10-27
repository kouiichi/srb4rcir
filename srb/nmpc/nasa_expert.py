"""Experimental acados NMPC using the explicitly configured actuator count.

The canonical 16-channel expert keeps its existing interface. This controller
uses the same costs, timing, move blocks and matched discrete executor, with
one nonnegative duty per explicitly configured nozzle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

from .acados_expert import AcadosExpert, _as_state
from .dynamics import casadi_quaternion_error_vector
from .quaternion import quaternion_to_rotation


@dataclass
class NasaExpertResult:
    predicted_state: np.ndarray
    predicted_duty: np.ndarray
    predicted_mass_fuel: np.ndarray
    status: int
    accepted: bool
    failure_reason: str | None
    constraint_margins: dict
    runtime_s: float
    shooting_state: np.ndarray


class NasaPairExpert:
    def __init__(self, params, executor, *, code_export_root):
        if params.mean_motion_rad_s != 0.:
            raise ValueError('NASA pair requires matched free-floating dynamics')
        if params.enable_fov:
            raise ValueError('original-model experiment has no added camera/FOV constraint')
        self.params, self.executor = params, executor
        self.nu = executor.num_thrusters
        self.code_export_root = Path(code_export_root)
        self.solver = None

    def _build(self):
        import casadi as ca
        AcadosExpert._ensure_acados_environment()
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        cfg, nu = self.params, self.nu
        nx, np_phys = 15+nu, 19+6*nu
        x, u, p = ca.SX.sym('x', nx), ca.SX.sym('u', nu), ca.SX.sym('p', 13+np_phys)
        extended = ca.vertcat(x[:13], x[13+nu:15+nu])
        following = self.executor.symbolic(extended, u, p[13:])
        model = AcadosModel()
        digest = hashlib.sha1(repr((cfg.signature(), self.executor.signature(), 'nasa_actual_nozzles_v1')).encode()).hexdigest()[:12]
        model.name = 'srb_nasa_' + digest
        model.x, model.u, model.p = x, u, p
        model.disc_dyn_expr = ca.vertcat(following[:13], u, following[13:15])
        error = ca.vertcat(x[:3]-p[:3], x[3:6]-p[3:6],
                           casadi_quaternion_error_vector(p[6:10], x[6:10]), x[10:13]-p[10:13])
        delta = u-x[13:13+nu]
        model.cost_y_expr = ca.vertcat(error, u, delta)
        model.cost_y_expr_e = error
        offset = x[:3]-ca.DM(cfg.koz_center_L_m)
        h = ca.dot(offset, offset)-cfg.koz_radius_m**2
        acceleration = (following[3:6]-x[3:6])/cfg.control_period_s
        cbf = (2*ca.dot(x[3:6],x[3:6])+2*ca.dot(offset,acceleration)
               +cfg.koz_k1_s*2*ca.dot(offset,x[3:6])+cfg.koz_k0_s2*h-cfg.koz_safety_margin_m2_s2)
        model.con_h_expr = ca.vertcat(h,cbf,delta)
        model.con_h_expr_0 = model.con_h_expr
        model.con_h_expr_e = h
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = cfg.horizon_steps
        ocp.solver_options.tf = cfg.horizon_seconds
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.qp_solver_cond_N = min(5,cfg.horizon_steps)
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.print_level = 0
        weights = np.repeat([cfg.position_cost,cfg.velocity_cost,cfg.attitude_cost,cfg.angular_velocity_cost],3)
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = np.diag(np.r_[weights, np.full(nu,cfg.duty_cost), np.full(nu,cfg.duty_rate_cost)])
        ocp.cost.W_e = np.diag(weights*cfg.terminal_cost_scale)
        ocp.cost.yref = np.zeros(12+2*nu)
        ocp.cost.yref_e = np.zeros(12)
        ocp.constraints.lbu = np.zeros(nu)
        ocp.constraints.ubu = np.ones(nu)
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.idxbx = np.r_[np.arange(3,6),np.arange(10,13)]
        ocp.constraints.lbx = -np.r_[np.full(3,cfg.velocity_limit_m_s),np.full(3,cfg.angular_velocity_limit_rad_s)]
        ocp.constraints.ubx = -ocp.constraints.lbx
        ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()
        ocp.constraints.lbx_e, ocp.constraints.ubx_e = ocp.constraints.lbx.copy(),ocp.constraints.ubx.copy()
        lower = np.r_[0.,0.,np.full(nu,-cfg.max_duty_delta)]
        upper = np.r_[1e10,1e10,np.full(nu,cfg.max_duty_delta)]
        ocp.constraints.lh,ocp.constraints.uh = lower,upper
        ocp.constraints.lh_0,ocp.constraints.uh_0 = lower.copy(),upper.copy()
        ocp.constraints.lh_e,ocp.constraints.uh_e = np.array([0.]),np.array([1e10])
        initial = np.zeros(nx); initial[6] = 1.;initial[-2] = self.executor.dry_mass
        ocp.constraints.x0 = initial
        ocp.parameter_values = np.r_[initial[:13],1.,np.eye(3).ravel(),np.eye(3).ravel(),np.zeros(6*nu)]
        export = self.code_export_root/model.name
        export.mkdir(parents=True,exist_ok=True)
        ocp.code_gen_options.code_export_directory = str(export)
        ocp.code_gen_options.json_file = str(export/(model.name+'_ocp.json'))
        self.solver = AcadosOcpSolver(ocp,verbose=False)

    def solve(self, state, physical, fuel, previous_duty, references, *, warm_duties=None):
        started = time.perf_counter()
        cfg,nu,N = self.params,self.nu,self.params.horizon_steps
        state,physical,previous,refs = (np.asarray(a,dtype=float) for a in (state,physical,previous_duty,references))
        if (state.shape!=(13,) or physical.shape!=(19+6*nu,) or previous.shape!=(nu,)
                or refs.shape!=(N+1,13) or not all(np.all(np.isfinite(a)) for a in (state,physical,previous,refs))):
            raise ValueError('invalid state, physical parameters, actual nozzle duties or reference shape/value')
        if not np.isfinite(fuel) or fuel<0 or physical[0]<=0 or not np.isclose(np.linalg.norm(state[6:10]),1.,atol=1e-6):
            raise ValueError('fuel/mass/quaternion invalid')
        # PhysX returns float32 quaternions; use the canonical expert's state
        # conversion before anchoring a double-precision rollout at node zero.
        state=_as_state(state)
        if self.solver is None:
            self._build()
        x0 = np.r_[state,physical[0],fuel]
        guess = np.tile(previous,(N,1)) if warm_duties is None else np.asarray(warm_duties).copy()
        if guess.shape!=(N,nu):
            raise ValueError('warm duties must match actual actuator count and horizon')
        rollout,guess = self.executor.warm_rollout(x0,guess,physical,cfg.action_block_lengths)
        solver = self.solver
        for k in range(N+1):
            prior = previous if k==0 else guess[k-1]
            solver.set(k,'x',np.r_[rollout[k,:13],prior,rollout[k,13:]])
            solver.set(k,'p',np.r_[refs[k],physical])
            if k<N:
                solver.set(k,'u',guess[k])
                bound = cfg.max_duty_delta if k in cfg.action_block_starts else 0.
                solver.constraints_set(k,'lh',np.r_[0.,0.,np.full(nu,-bound)])
                solver.constraints_set(k,'uh',np.r_[1e10,1e10,np.full(nu,bound)])
        initial = np.r_[state,previous,physical[0],fuel]
        solver.constraints_set(0,'lbx',initial)
        solver.constraints_set(0,'ubx',initial)
        status = int(solver.solve())
        aug = np.asarray([solver.get(k,'x') for k in range(N+1)])
        duties = np.asarray([solver.get(k,'u') for k in range(N)])
        predicted = aug[:,:13]
        mass_fuel = aug[:,-2:]
        finite = bool(np.all(np.isfinite(aug)) and np.all(np.isfinite(duties)))
        margins = {}
        accepted = status==0 and finite
        reason = None if accepted else f'acados status {status}; finite={finite}'
        if finite:
            independent = [x0]
            for duty in duties:
                independent.append(self.executor.step(independent[-1],duty,physical))
            independent = np.asarray(independent)
            shooting_state_defect = float(np.max(np.abs(independent[:,:13]-predicted)))
            shooting_mass_defect = float(np.max(np.abs(independent[:,13:]-mass_fuel)))
            if status==0:
                # Match the canonical expert: SQP-RTI shooting nodes have a
                # nonlinear defect; public predictions integrate executable
                # duties through the matched model before all safety checks.
                predicted = independent[:,:13]
                mass_fuel = independent[:,13:]
            differences = np.diff(np.vstack((previous,duties)),axis=0)
            blocked = [np.max(np.abs(differences[k])) for k in range(N) if k not in cfg.action_block_starts]
            offset = predicted[:,:3]-cfg.koz_center_L_m
            h = np.sum(offset**2,axis=1)-cfg.koz_radius_m**2
            velocity = predicted[:-1,3:6]
            acceleration = np.diff(independent[:,3:6],axis=0)/cfg.control_period_s
            cbf = (2*np.sum(velocity**2,axis=1)+2*np.sum(offset[:-1]*acceleration,axis=1)
                   +cfg.koz_k1_s*2*np.sum(offset[:-1]*velocity,axis=1)+cfg.koz_k0_s2*h[:-1]-cfg.koz_safety_margin_m2_s2)
            margins = {
                'duty_lower':float(duties.min()),'duty_upper':float((1.-duties).min()),
                'duty_rate':float(cfg.max_duty_delta-np.max(np.abs(differences))),
                'block_residual':float(max(blocked,default=0.)),
                'dynamics_residual':float(np.max(np.abs(independent[:,:13]-predicted))),
                'mass_fuel_residual':float(np.max(np.abs(independent[:,13:]-mass_fuel))),
                'shooting_state_defect':shooting_state_defect,
                'shooting_mass_fuel_defect':shooting_mass_defect,
                'koz_h':float(h.min()),'koz_ecbf':float(cbf.min()),
                'velocity':float(cfg.velocity_limit_m_s-np.max(np.abs(predicted[:,3:6]))),
                'angular_velocity':float(cfg.angular_velocity_limit_rad_s-np.max(np.abs(predicted[:,10:13]))),
                'quaternion_residual':float(np.max(np.abs(np.linalg.norm(predicted[:,6:10],axis=1)-1.))),
            }
            negative = [k for k in ('duty_lower','duty_upper','duty_rate','koz_h','koz_ecbf','velocity','angular_velocity') if margins[k]<-1e-6]
            residual = [k for k in ('block_residual','dynamics_residual','mass_fuel_residual') if margins[k]>1e-6]
            if margins['quaternion_residual']>1e-8:
                residual.append('quaternion_residual')
            if accepted and (negative or residual):
                accepted,reason = False,'quality gate: '+', '.join(negative+residual)
        return NasaExpertResult(predicted,duties,mass_fuel,status,accepted,reason,margins,time.perf_counter()-started,aug)
