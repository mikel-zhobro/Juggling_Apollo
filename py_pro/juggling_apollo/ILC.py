import numpy as np
from settings import g, m_b, m_p, k_c
from DynamicSystem import DynamicSystem
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory
from KalmanFilter import KalmanFilter
from utils import steps_from_time, DotDict, plt
from JugglingPlanner import traj_nb_2_na_1

class ILC:
  def __init__(self, dt, kf_dpn_params, x_0, t_f, t_h=None):
    # design params
    self.dt  = dt
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
    self.kf_dpn = KalmanFilter(lss=self.lss, **kf_dpn_params)  # dpn estimator

    self.t_f = t_f  # flying time of the ball
    self.t_h = t_h  # time that ball is in the hand
    # TODO: n_b/n_a = (t_h+t_f)/(t_h+t_e)
    if self.t_h is None:
      self.t_h = self.t_f/2

  def initILC(self, N_1, impact_timesteps):
    self.N_1 = len(impact_timesteps)  # time steps
    self.lss.updateQuadrProgMatrixes(impact_timesteps)  # init LSS
    self.resetILC()  # init KFs

  def resetILC(self):
    # reset KFs
    self.kf_dpn.resetKF()

  def learnThrowStep(self, ub_throw, T_throw, x_catch=0.0, u_ff_old=None, y_meas=None, d1_meas=0, d2_meas=0):
    # 1. Throw
    if u_ff_old is not None:  # we are calculating u_ff for the first time
      self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc new ub_throw
    ub_throw = ub_throw - 0.3*0.5*g*d2_meas  # move in oposite direction of error

    # new MinJerk
    y_des, v, a, j = get_min_jerk_trajectory(dt=self.dt, ta=0, tb=T_throw, x_ta=self.x_0[0], x_tb=0.0, u_ta=self.x_0[2], u_tb=ub_throw)
    # plotMinJerkTraj(y_des, v, a, j, self.dt, "MINJERK")

    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1))
    return y_des, u_ff_new, ub_throw

  def learnEmptyStep(self, u_ff_old, y_meas, y_des):
    # 1. Throw
    self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1))
    return u_ff_new
  
  def learnEmptyStep(self, u_ff_old, y_meas, y_des):
    # 1. Throw
    self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1))
    return u_ff_new

  def learnWhole2(self, ub_throw, ub_catch, 
                  T_throw, T_hand, T_empty, z_catch,
                  u_ff_old=None, y_meas=None, 
                  d1_meas=0, d2_meas=0, d3_meas=0):
    # 1. Throw
    if u_ff_old is not None:  # we are calculating u_ff for the first time
      self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc new ub_0
    ub_throw = ub_throw - 0.3*g*d2_meas  # move in oposite direction of error

    # calc new catch velocity
    # ub_catch = -ub_0/4
    ub_catch = ub_catch - 0.3*0.5*g*d3_meas  # move in oposite direction of error
    # new MinJerk
    y_des = traj_nb_2_na_1(T_throw, T_hand, ub_catch, ub_throw, T_empty, z_catch, self.x_0[0:-1:2], self.dt, False)
    # calc desired input
    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1), True)
    return y_des, u_ff_new, ub_throw, 0

  def learnWhole(self, ub_0, t_catch=None, u_ff_old=None, y_meas=None, d1_meas=0, d2_meas=0, d3_meas=0):
    # 1. Throw
    if u_ff_old is not None:  # we are calculating u_ff for the first time
      self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # calc new ub_0
    ub_0 = ub_0 - 0.3*0.5*g*d2_meas  # move in oposite direction of error # TODO: vectorized

    # calc new catch velocity # TODO: vectorized
    no_t_catch = t_catch is None
    if no_t_catch:
      t_catch = self.t_h/2
      t_throw = self.t_h/2  # self.t_h + self.t_f - t_catch
    else:
      t_catch = min(self.t_h-0.01, float(t_catch - 0.03*d3_meas))  # move in oposite direction of error
      t_throw = float(self.t_h - t_catch)

    # new MinJerk
    t_end = self.t_f + self.t_h
    t0 = 0;           t1 = t_throw;    t2 = t1 + self.t_f/4;  t3 = t_end - t_catch;  t4 = t_end
    x0 = self.x_0[0]; x1 = 0;          x2 = -0.2;               x3 = 0;                x4 = x0
    u0 = self.x_0[2]; u1 = ub_0;       u2 = 0.0;          u3 = -ub_0/3;          u4 = u0
    # a0 = None;        a1 = None;       a2 = None;          a3 = None
    tt=[t0, t1, t2, t3, t4]
    xx=[x0, x1, x2, x3, x4]
    uu=[u0, u1, u2, u3, u4]
    smooth_acc = False
    y_des, v, a, j = get_minjerk_trajectory(self.dt, smooth_acc=smooth_acc,
                                            tt=tt,
                                            xx=xx,
                                            uu=uu)
    # title = "Min-Jerk trajectory with " +  ("" if smooth_acc else "non") +"-smoothed acceleration."
    # plotMinJerkTraj(y_des, v, a, j, self.dt, title, tt=tt[0:4], xx=xx[0:4], uu=uu[0:4])
    # plt.show()

    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1), True)

    if no_t_catch:
      return y_des, u_ff_new, ub_0
    else:
      return y_des, u_ff_new, ub_0, t_catch
