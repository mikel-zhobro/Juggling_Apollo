import numpy as np
from settings import g
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter


class ILC:
  def __init__(self, dt, sys, kf_dpn_params, x_0):
    # design params
    self.dt  = dt
    # starting state
    self.x_0 = x_0
    # components of ilc
    self.lss              = LiftedStateSpace(sys=sys, x0=x_0)
    self.quad_input_optim = OptimLss(self.lss)
    self.kf_dpn           = KalmanFilter(lss=self.lss, **kf_dpn_params)  # dpn estimator

  def initILC(self, N_1, impact_timesteps):
    self.N_1 = len(impact_timesteps)  # time steps
    self.lss.updateQuadrProgMatrixes(impact_timesteps)  # init LSS
    self.resetILC()  # init KFs

  def resetILC(self):
    self.kf_dpn.resetKF() # reset KFs

  @property
  def d(self):
    return self.kf_dpn.d

  def learnWhole(self, y_des, u_ff_old=None, y_meas=None, verbose=False, lb=None, ub=None):
    # 1. Throw
    disturbance = self.kf_dpn.d
    if u_ff_old is not None:  # we are calculating u_ff for the first time(so just use the linear model for that)
      disturbance = self.kf_dpn.updateStep(u_ff_old.reshape(-1, 1), y_meas.reshape(-1, 1))  # estimate dpn disturbance

    u_ff_new = self.quad_input_optim.calcDesiredInput(disturbance, np.array(y_des[1:], dtype='float').reshape(-1, 1), verbose, lb, ub)

    return u_ff_new
