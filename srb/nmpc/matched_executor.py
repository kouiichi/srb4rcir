"""Free-floating discrete executor model, audited against CPU PhysX.

Internal state is [COM state13, mass at force submission, remaining fuel].
Forces are accumulated using pre-setter inverse mass/inertia. Fuel and inertia
are then updated. COM position uses semi-implicit Euler, matching PhysX rather
than the historical continuous RK4 position update. No orbital terms exist.
"""
from dataclasses import dataclass
import numpy as np


def stationary_com_goal(actor_position, actor_quaternion, com):
    """One world-frame stationary goal shared by control and evaluation."""
    from .quaternion import quaternion_to_rotation
    q=np.asarray(actor_quaternion,dtype=float)
    q=q/np.linalg.norm(q)
    goal=np.zeros(13)
    goal[:3]=np.asarray(actor_position)+quaternion_to_rotation(q)@np.asarray(com)
    goal[6:10]=q
    return goal


@dataclass
class MatchedExecutor:
    dry_mass: float
    burn_per_duty: np.ndarray
    dt: float = .05
    angular_damping: float = .05

    def __post_init__(self):
        self.burn_per_duty = np.asarray(self.burn_per_duty, dtype=float)
        if (self.burn_per_duty.ndim != 1 or len(self.burn_per_duty) == 0
                or not np.all(np.isfinite(self.burn_per_duty)) or np.any(self.burn_per_duty < 0)):
            raise ValueError('burn coefficients must be a finite nonnegative nonempty vector')
        self.num_thrusters = len(self.burn_per_duty)
        self.remaining_fuel = 0.
        self._function = None

    def signature(self):
        return ('physx_free_v1', self.dry_mass, tuple(self.burn_per_duty), self.dt, self.angular_damping)

    def symbolic(self, state, u, physical):
        import casadi as ca
        from .dynamics import casadi_quaternion_to_rotation, casadi_omega_body_matrix
        x, mass, fuel = state[:13], state[13], state[14]
        R = casadi_quaternion_to_rotation(x[6:10])
        J = ca.reshape(physical[1:10],3,3).T * mass / physical[0]
        Ji = ca.reshape(physical[10:19],3,3).T * physical[0] / mass
        B = ca.reshape(physical[19:19+6*self.num_thrusters],self.num_thrusters,6).T
        enabled = ca.if_else(fuel > 0., 1., 0.)
        wrench = (B @ u) * enabled
        v = x[3:6] + self.dt * R @ wrench[:3] / mass
        w = x[10:13]
        # Implicit gyro correction: one Newton linearization of the free-body
        # Euler equation. Torque increment uses inertia at force submission.
        def skew(a):
            return ca.vertcat(ca.horzcat(0,-a[2],a[1]),ca.horzcat(a[2],0,-a[0]),ca.horzcat(-a[1],a[0],0))
        gyro = ca.solve(J + self.dt*(skew(w)@J-skew(J@w)), -self.dt*ca.cross(w,J@w))
        w_next_old = (w + gyro + self.dt*Ji@wrench[3:]) * max(0., 1-self.dt*self.angular_damping)
        # Exponential quaternion update in the old body coordinates.
        angle = ca.sqrt(ca.sumsqr(w_next_old) + 1e-24)
        half = .5*self.dt*angle
        q = ca.cos(half)*x[6:10] + ca.sin(half)/angle * (casadi_omega_body_matrix(w_next_old)@x[6:10])
        q = q/ca.sqrt(ca.sumsqr(q))
        R_next = casadi_quaternion_to_rotation(q)
        w_next = R_next.T @ R @ w_next_old
        fuel_next = ca.fmax(0., fuel - enabled*ca.dot(ca.DM(self.burn_per_duty),u))
        return ca.vertcat(x[:3]+self.dt*v,v,q,w_next,self.dry_mass+fuel_next,fuel_next)

    def function(self):
        if self._function is None:
            import casadi as ca
            x=ca.SX.sym('x',15); u=ca.SX.sym('u',self.num_thrusters)
            p=ca.SX.sym('p',19+6*self.num_thrusters)
            self._function=ca.Function('matched_step',[x,u,p],[self.symbolic(x,u,p)])
        return self._function

    def warm_rollout(self, state, duties, physical, block_lengths):
        """Dynamically consistent warm nodes from current measured parameters.

        Re-block the shifted guess only; never project the solved command.
        """
        duties=np.asarray(duties).copy()
        start=0
        for length in block_lengths:
            duties[start:start+length]=duties[start]
            start+=length
        states=[np.asarray(state)]
        for duty in duties:
            states.append(self.step(states[-1],duty,physical))
        return np.asarray(states),duties

    def step(self, state, u, physical, fuel=None):
        state=np.asarray(state,dtype=float)
        if len(state)==14:
            fuel=max(0.,state[13]-self.dry_mass) if fuel is None else fuel
            return np.asarray(self.function()(np.r_[state,fuel],u,physical)).ravel()[:14]
        return np.asarray(self.function()(state,u,physical)).ravel()
