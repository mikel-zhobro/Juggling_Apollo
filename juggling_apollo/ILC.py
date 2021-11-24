import numpy as np
from scipy import optimize
import utils
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter


class ILC:
  def __init__(self, sys, y_des, freq_domain=False, Nf=None, lss_params={}, optim_params={}, kf_dpn_params={}):
    # kwargs can be {impact_timesteps: [True/False]*N} for timedomain
    # and must be {T:Float} for freqdomain
    
    assert kf_dpn_params['d0'].size == (self if freq_domain else y_des.size), "Size of disturbance ({}) and that of N ({}) don't match.".format(kf_dpn_params['d0'].size, self.N)
    # N: traj length if timedomain, freq traje length otherwise
    
    self.timeDomain = not freq_domain
    self.firstTime = True
    
    # Time domain
    self.uff_t = None
    self.dt = sys.dt
    self.Nt = y_des.size
    self.T = utils.time_from_step(self.Nt, self.dt)  # periode
    self.y_des_t = y_des.reshape(-1,1)
    
    # Freq domain
    self.uff_f   = None
    self.Nf      = None
    self.y_des_f = None
    
    if freq_domain:
      assert Nf is not None and Nf < int(0.5*self.Nt), "Make sure that Nf{} is small enough{} to satisfy the Nyquist criterium.".format(Nf, int(0.5*self.Nt))
      self.Nf = None
      self.y_des_f = fourier_series_coeff(y_des, self.N, complex=True).reshape(-1,1)

    # components of ilc
    self._u_ff = None
    self.lss              = LiftedStateSpace(sys=sys, T=self.T, N=self.N, freq_domain=freq_domain, **lss_params)
    self.kf_dpn           = KalmanFilter(lss=self.lss, freqDomain=freq_domain, **kf_dpn_params)  # dpn estimator
    self.quad_input_optim = OptimLss(self.lss, **optim_params)

    # size of the trajectoty
    self.resetILC()

  def resetILC(self, d=None, P=None):
    self.kf_dpn.resetKF(d, P) # reset KFs

  @property
  def y_des(self):
    return self.y_des_t if self.timeDomain else self.y_des_f

  @property
  def N(self):
    return self.Nt if self.timeDomain else self.Nf

  @property
  def d(self):
    return self.kf_dpn.d

  @property
  def P(self):
    return self.kf_dpn.P

  def update_y_des(self, y_des):
    assert y_des.size == self.Nt, "Make sure the new y_des has the same size as the input ILC was initialized with."
    self.y_des_t = y_des.reshape(-1,1)
    if not self.timeDomain:
      self.y_des_f = fourier_series_coeff(y_des, self.N, complex=True).reshape(-1,1)

  def transf_uff(self, uff):
    """ Transforms uff to the right format for robot usage.

    Args:
        uff ([np.array(N{t/f},1)]): feedForward input in timeDomain or freqDomain according to how ILC was initialized

    Returns:
        [np.array(Nt,1)]: copy of feedForward input in timeDomain
    """
    return uff.copy() if self.timeDomain else series_real_coeff(uff, t=np.arange(0, self.Nt)*self.dt, T=self.T)

  def init_uff_from_lin_model(self, verbose=False, lb=None, ub=None):
    if not self.timeDomain:
      assert lb is None and ub is None, "No constraint optimization possible in freqDomain."

    # update uff
    self._u_ff = self.quad_input_optim.calcDesiredInput(np.zeros_like(self.kf_dpn.d), self.y_des.reshape(-1, 1), verbose=verbose, lb=lb, ub=ub)
    return self.transf_uff(self._u_ff)

  def updateStep(self, y_meas, y_des=None, verbose=False, lb=None, ub=None):
    """ Updates learned feedforward input and disturbance according to (self._u_ff, y_meas) tuple. 

    Args:
        y_meas ([np.array(Nt, 1)]): measured output for the previously calculated u_ff
        y_des ([np.array(Nt, 1)], optional): A new desired desired trajectory in time domain. Defaults to None.
        verbose (bool, optional): whether to print out verbose info
        lb, ub ([type], optional): Whether to inforce upper and lower bounds onthe input. Only valid for timeDomain usage.

    Returns:
        [np.array(Nt, 1)]: new feed forward input
    """
    if not self.timeDomain:
      assert lb is None and ub is None, "No constraint optimization possible in freqDomain."

    if y_des is not None:
      self.update_y_des(y_des)

    if self._u_ff is None:
      return self.init_uff_from_lin_model(verbose, lb, ub)

    # estimate dpn disturbance
    disturbance = self.kf_dpn.updateStep(self._u_ff.reshape(-1, 1), y_meas.reshape(-1, 1))
    # update uff
    self._u_ff = self.quad_input_optim.calcDesiredInput(disturbance, self.y_des, verbose=verbose, lb=lb, ub=ub)

    return self.transf_uff(self._u_ff)


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
    y = np.fft.rfft(f, norm=None, axis=0) / f.size * 2.0
    if complex:
        return y[:Nf]
    return y[0].real, y[1:].real[0:Nf], -y[1:].imag[0:Nf]

def series_real_coeff(y, t, T):
    """calculates the Fourier series with period T at times t,
    from the real coeff. a0,a,b"""
    tmp = np.ones_like(t) * y[0].real / 2.
    for k, (ak, bk) in enumerate(zip(y[1:].real, -y[1:].imag)):
        tmp += ak * np.cos(2 * np.pi * (k + 1) * t / T) + bk * np.sin(2 * np.pi * (k + 1) * t / T)
    return tmp