import numpy as np
from settings import g, m_b, m_p, k_c
from DynamicSystem import DynamicSystem
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory
from KalmanFilter import KalmanFilter
from utils import steps_from_time, DotDict


class ILC:
    # ILC('x_0', x_0, 't_f', t_f)

  def __init__(self, dt, kf_d1d2_params, kf_dpn_params, x_0, t_f, t_h=None):
    # design params
    self.dt  = dt
    self.kf_d1d2_params = kf_d1d2_params
    self.kf_dpn_params = kf_dpn_params
    self.x_0 = x_0  # starting state

    # Here we want to set some convention to avoid missunderstandins later on.
    # 1. the state is [xb, xp, ub, up]^T
    # 2. the system can have as input either velocity u_des or the force F_p
    # I. SYSTEM DYNAMICS
    input_is_velocity = True
    self.sys = DynamicSystem(self.dt, input_is_velocity=input_is_velocity)
    # II. LIFTED STATE SPACE
    self.lss = LiftedStateSpace(sys=self.sys, x0=x_0)
    # IV. DESIRED INPUT OPTIMIZER
    self.quad_input_optim = OptimLss(self.lss)
    # V. KALMAN FILTERS
    # d1, d2
    s = DotDict(
      {
        'GK': np.eye(2, dtype='float'),
        'Gd0': np.array([0, 0], dtype='float').reshape(-1, 1),
        'GF': np.array([0, 0], dtype='float').reshape(-1, 1),
      }
    )
    self.kf_d1d2 = KalmanFilter(lss=s, **self.kf_d1d2_params)
    # dpn
    self.kf_dpn = KalmanFilter(lss=self.lss, **self.kf_dpn_params)

    self.t_f = t_f  # flying time of the ball
    self.t_h = t_h  # time that ball is in the hand
    # TODO: n_b/n_a = (t_h+t_f)/(t_h+t_e)
    if self.t_h is None:
      self.t_h = self.t_f/2

  def initILC(self, N_1, impact_timesteps):
    self.N_1 = len(impact_timesteps)  # time steps
    # init LSS
    self.lss.updateQuadrProgMatrixes(impact_timesteps)
    # init KFs
    self.resetILC()

  def resetILC(self):
    # reset KFs
    self.kf_d1d2.resetKF()
    self.kf_dpn.resetKF()

  def learnThrowStep(self, ub_0, u_ff_old=None, y_meas=None, d1_meas=0, d2_meas=0):
    # 1. Throw
    if u_ff_old is not None:  # we are calculating u_ff for the first time
      self.kf_d1d2.updateStep(0, np.array([d1_meas, d2_meas], dtype='float').reshape(-1, 1))  # estimate d1d2 disturbances
      self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc new ub_0
    # ub_0 = 0.5*self.g*( self.t_f - self.kf_d1d2.d(2))
    ub_0 = ub_0 - 0.3*0.5*g*self.kf_d1d2.d[1]  # move in oposite direction of error
    # ub_0 = ub_0 - 0.7*self.kf_d1d2.d(2) # move in oposite direction of error

    # new MinJerk
    y_des, v, a, j = get_min_jerk_trajectory(dt=self.dt, ta=0, tb=self.t_h/2, x_ta=self.x_0[0], x_tb=0, u_ta=self.x_0[2], u_tb=ub_0)
    # plotMinJerkTraj(y_des, v, a, j, self.dt, "MINJERK")

    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1))
    return y_des, u_ff_new, ub_0

  def learnPlateMotionStep(self, y_des, u_ff_old=None, y_meas=None):
    # 2. Plate Free Motion
    if u_ff_old is not None >2:  # the first time we call it like: learnPlateMotionStep(ilc)
      # estimate dpn disturbance
      self.kf_dpn.updateStep(u_ff_old, y_meas)
    # else:
    #   self.N_1 = steps_from_time(self.t_f, self.dt) - 1  # important since input cannot influence the first state
    #   self.resetILC([2] + [1]*self.N_1-2 + [2])

    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, y_des)
    return u_ff_new

  def learnWhole(self, ub_0, u_ff_old=None, y_meas=None, d1_meas=0, d2_meas=0):
    # 1. Throw
    if u_ff_old is not None:  # we are calculating u_ff for the first time
      self.kf_d1d2.updateStep(0, np.array([d1_meas, d2_meas], dtype='float').reshape(-1, 1))  # estimate d1d2 disturbances
      self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc new ub_0
    # ub_0 = 0.5*self.g*( self.t_f - self.kf_d1d2.d(2))
    ub_0 = ub_0 - 0.3*0.5*g*self.kf_d1d2.d[1]  # move in oposite direction of error
    # ub_0 = ub_0 - 0.7*self.kf_d1d2.d(2) # move in oposite direction of error

    # new MinJerk
    t0 = 0;           t1 = self.t_h/2; t2 = t1 + self.t_f; t3 = self.t_f + self.t_h
    x0 = self.x_0[0]; x1 = 0;          x2 = 0;             x3 = x0
    u0 = self.x_0[2]; u1 = ub_0;       u2 = -ub_0/8;       u3 = u0
    # a0 = None;        a1 = None;       a2 = None;          a3 = None
    y_des, _, _, _ = get_minjerk_trajectory(self.dt, ta=[t0, t1, t2], tb=[t1, t2, t3],
                                            x_ta=[x0, x1, x2], x_tb=[x1, x2, x3],
                                            u_ta=[u0, u1, u2], u_tb=[u1, u2, u3])

    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1), True)
    return y_des, u_ff_new, ub_0
