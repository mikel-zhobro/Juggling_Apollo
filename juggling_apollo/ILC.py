import numpy as np
from settings import g
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter


class ILC:
  def __init__(self, sys, kf_dpn_params, x_0, freq_domain=False, **kwargs):
    # kwargs can be {impact_timesteps: [True/False]*N} for timedomain 
    # and must be {T:Float} for freqdomain
    self.FirstTime = True
    # starting state
    self.x_0 = x_0
    # components of ilc
    self.lss              = LiftedStateSpace(sys=sys, x0=x_0)
    self.quad_input_optim = OptimLss(self.lss)
    self.kf_dpn           = KalmanFilter(lss=self.lss, **kf_dpn_params)  # dpn estimator
    # size of the trajectoty
    self.N = self.kf_dpn.N
    self._initILC(freq_domain, **kwargs)

  def _initILC(self, freq_domain, **kwargs):
    self.lss.updateQuadrProgMatrixes(N=self.N, freq_domain=freq_domain, **kwargs)  # init LSS
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
