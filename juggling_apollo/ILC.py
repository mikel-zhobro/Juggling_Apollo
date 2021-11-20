import numpy as np
from settings import g
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter


class ILC:
  def __init__(self, dt, sys, kf_dpn_params, x_0, impact_timesteps=None):
    
    self.FirstTime = True
    # design params
    self.dt  = dt
    # starting state
    self.x_0 = x_0
    # components of ilc
    self.lss              = LiftedStateSpace(sys=sys, x0=x_0)
    self.quad_input_optim = OptimLss(self.lss)
    self.kf_dpn           = KalmanFilter(lss=self.lss, **kf_dpn_params)  # dpn estimator
    # size of the trajectoty
    self.N = self.kf_dpn.N
    
    self._initILC(impact_timesteps)

  def _initILC(self, impact_timesteps=None):
    if impact_timesteps is not None:
      assert self.N == len(impact_timesteps)
    else:
      impact_timesteps = [False]*self.N
      
    self.lss.updateQuadrProgMatrixes(impact_timesteps)  # init LSS
    self.resetILC()  # init KFs

  def resetILC(self, d=None, P=None):
    self.kf_dpn.resetKF() # reset KFs

  @property
  def d(self):
    return self.kf_dpn.d

  @property
  def P(self):
    return self.kf_dpn.P


  def ff_from_lin_model(self, y_des):
    self.FirstTime = False
    u_ff = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, y_des.reshape(-1, 1))
    return u_ff

  def learnWhole(self, y_des, u_ff_old, y_meas, verbose=False, lb=None, ub=None):
    # 1. Throw
    if self.FirstTime:
      self.FirstTime = False
      disturbance = self.kf_dpn.d # we are calculating u_ff for the first time(so just use the linear model for that, i.e. use the initial d=0)
    else:
      disturbance = self.kf_dpn.updateStep(u_ff_old.reshape(-1, 1), y_meas.reshape(-1, 1))  # estimate dpn disturbance

    u_ff_new = self.quad_input_optim.calcDesiredInput(disturbance, y_des.reshape(-1, 1), verbose, lb, ub)

    return u_ff_new
