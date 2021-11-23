import numpy as np
import utils
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter


class ILC:
  def __init__(self, sys, y_des, freq_domain=False, Nf=None, lss_params={}, optim_params={}, kf_dpn_params={}):
    # kwargs can be {impact_timesteps: [True/False]*N} for timedomain
    # and must be {T:Float} for freqdomain
    self.freq_domain = freq_domain
    self.dt = sys.dt
    self.T = utils.time_from_step(y_des.size, self.dt)  # periode
    if not freq_domain:
      self.y_des = y_des
      self.N = y_des.size
    else:
      assert Nf is not None, "Please choose Nf for the fourier transform. Allowed values are ( Nf <= 0.5*self.T/self.sys.dt )"
      assert Nf <= 0.5*self.T/self.dt, "Make sure that Nf{} is small enough{} to satisfy the Nyquist criterium.".format(Nf, 0.5*self.T/self.dt)
      self.N = Nf
      self.y_des = np.fft.fft(y_des)[:self.N]  #TODO

    assert kf_dpn_params['d0'].size * sys.S.shape[1] == self.N, "Size of disturbance ({}) and that of N ({}) don't match.".format(kf_dpn_params['d0'].size * sys.S.shape[1], self.N)


    self.FirstTime = True
    # components of ilc
    self.lss              = LiftedStateSpace(sys=sys, T=self.T, N=self.N, freq_domain=freq_domain, **lss_params)
    self.quad_input_optim = OptimLss(self.lss, **optim_params)
    self.kf_dpn           = KalmanFilter(lss=self.lss, **kf_dpn_params)  # dpn estimator
    # size of the trajectoty
    self.N = self.kf_dpn.N
    self.resetILC()

  def resetILC(self, d=None, P=None):
    self.kf_dpn.resetKF(d, P) # reset KFs

  @property
  def d(self):
    return self.kf_dpn.d

  @property
  def P(self):
    return self.kf_dpn.P


  def ff_from_lin_model(self, y_des):
    self.FirstTime = False
    u_ff = self.quad_input_optim.calcDesiredInput(self.kf_dpn.d, self.y_des.reshape(-1, 1))

    if self.freq_domain:
      u_ff = np.fft.ifft(u_ff, utils.steps_from_time(self.T, self.dt)) #TODO
    return u_ff

  def learnWhole(self, y_des, u_ff_old, y_meas, verbose=False, lb=None, ub=None):
    # 1. Throw
    if self.FirstTime:
      self.FirstTime = False
      return self.ff_from_lin_model(y_des)

    disturbance = self.kf_dpn.updateStep(u_ff_old.reshape(-1, 1), y_meas.reshape(-1, 1))  # estimate dpn disturbance
    u_ff_new = self.quad_input_optim.calcDesiredInput(disturbance, y_des.reshape(-1, 1), verbose, lb, ub)
    if self.freq_domain:
      u_ff_new = np.fft.ifft(u_ff_new, utils.steps_from_time(self.T, self.dt)) #TODO
    return u_ff_new


  def fourier_series_coeff(f, Nf, complex=False):
      """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

      Given a periodic, function f(t) with period T, this function returns the
      coefficients a0, {a1,a2,...},{b1,b2,...} such that:

      f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

      Parameters
      ----------
      f : the periodic function values
      Nf : the function will return the first N + 1 Fourier coeff.

      Returns
      -------
      a0 : float
      a,b : numpy float arrays describing respectively the cosine and sine coeff.
      """
      # In order for the nyquist theorem to be satisfied N_t > 2 N_f where N_t=f.size = T/dt
      y = np.fft.rfft(f, norm=None) / f.size * 2.0
      if complex:
          return y[:Nf]
      return y[0].real, y[1:].real[0:Nf], -y[1:].imag[0:Nf]

  def series_real_coeff(a0, a, b, t, T):
      """calculates the Fourier series with period T at times t,
      from the real coeff. a0,a,b"""
      tmp = np.ones_like(t) * a0 / 2.
      for k, (ak, bk) in enumerate(zip(a, b)):
          tmp += ak * np.cos(2 * np.pi * (k + 1) * t / T) + bk * np.sin(2 * np.pi * (k + 1) * t / T)
      return tmp
