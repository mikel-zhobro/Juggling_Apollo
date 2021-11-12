import numpy as np
from settings import g
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter
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

  @property
  def d(self):
    return self.kf_dpn.d

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

  def learnWhole(self, y_des, u_ff_old=None, y_meas=None, verbose=False, lb=None, ub=None):
    # 1. Throw
    disturbance = self.kf_dpn.d
    if u_ff_old is not None:  # we are calculating u_ff for the first time(so just use the linear model for that)
      disturbance = self.kf_dpn.updateStep(u_ff_old.reshape(-1, 1), y_meas.reshape(-1, 1))  # estimate dpn disturbance

    u_ff_new = self.quad_input_optim.calcDesiredInput(disturbance, np.array(y_des[1:], dtype='float').reshape(-1, 1), verbose, lb, ub)

    return u_ff_new
