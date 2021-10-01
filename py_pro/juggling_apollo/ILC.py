import numpy as np
from settings import g, m_b, m_p, k_c
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory
from KalmanFilter import KalmanFilter
from utils import steps_from_time, DotDict, plt
from JugglingPlanner import traj_nb_2_na_1

class ILC:
  def __init__(self, dt, sys, kf_dpn_params, x_0):
    # design params
    self.dt  = dt
    self.x_0 = x_0  # starting state

    # II. LIFTED STATE SPACE
    self.lss = LiftedStateSpace(sys=sys, x0=x_0)
    # IV. DESIRED INPUT OPTIMIZER
    self.quad_input_optim = OptimLss(self.lss)
    # V. KALMAN FILTERS
    self.kf_dpn = KalmanFilter(lss=self.lss, **kf_dpn_params)  # dpn estimator

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

  def learnWhole(self, y_des, u_ff_old=None, y_meas=None):
    # 1. Throw
    if u_ff_old is not None:  # we are calculating u_ff for the first time
      self.kf_dpn.updateStep(u_ff_old, y_meas)  # estimate dpn disturbance

    # title = "Min-Jerk trajectory with " +  ("" if smooth_acc else "non") +"-smoothed acceleration."
    # plotMinJerkTraj(y_des, v, a, j, self.dt, title, tt=tt[0:4], xx=xx[0:4], uu=uu[0:4])
    # plt.show()


    u_ff_new = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1), True)

    return u_ff_new
