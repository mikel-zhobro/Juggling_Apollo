#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ILC.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   In this file we define the class that puts the ILC components (LSS, KF) together
             and defines the update steps(OptimLSS).
             The method performing the update for time-domain ILC is ILC.updateStep(..),
             the methods performing the update for freq-domain ILC are ILC.updateStep2(..) and ILC.updateStep3(..).
'''

import numpy as np
import utils
from LiftedStateSpace import LiftedStateSpace
from OptimLss import OptimLss
from KalmanFilter import KalmanFilter


class ILC:
  def __init__(self, sys, y_des, freq_domain=False, Nf=None, kf_dpn_params={}):
    """_summary_

    Args:
        sys (DynamicSystem): The estimated state space equations of the plant.
        y_des (np.array(Nt, 1)): The desired output trajectory for which we want to learn the feedforward input.
        freq_domain (bool, optional): Whether we are doing time- or freq-domain ILC. Defaults to False.
        Nf (int, optional): The number of fourier coefficients to be used in case of freq-domain ILC.
        kf_dpn_params (dict, optional): Dictionary with parameters for the Kalman Filter {M:, d0:, P0:, epsilon0:, epsilon_decrease_rate:}
    """

    # Time domain
    self.timeDomain = not freq_domain
    self.dt = sys.dt
    self.Nt = y_des.size
    self.T = utils.time_from_step(self.Nt, self.dt)  # periode

    # Freq domain
    self.Nf = Nf # nr of fourier coeffizients to use for the frequency-domain ILC.
    assert not freq_domain or (Nf is not None and Nf < int(0.5*self.Nt)), "Make sure that Nf{} is small enough{} to satisfy the Nyquist criterium.".format(Nf, int(0.5*self.Nt))
    assert kf_dpn_params['d0'].size == self.N, "Size of disturbance ({}) and that of N ({}) don't match.".format(kf_dpn_params['d0'].size, self.N)

    # Components of ILC
    self.y_des = None  # keeps the time domain desired trajectory
    self.c_s_ = None   # keeps the Fourier coefficients for the cos() and sin() terms of Fourier series
    self._u_ff = None  # keeps the initial feedforward signal (Fourier coeficients in FreqDomain)
    self._delta_u_ff = None  # the actual
    self.y_des_feedback = 0.  # keeps the feedback part that has to be substracted from the desired traj (Fourier Coef in FreqDomin)
    self.lss              = LiftedStateSpace(sys=sys, N=self.N, T=self.T, freq_domain=freq_domain)
    self.kf_dpn           = KalmanFilter(lss=self.lss, freqDomain=freq_domain, **kf_dpn_params)  # disturbance estimator
    self.quad_input_optim = OptimLss(self.lss)

    # Init
    self.update_y_des(y_des)
    self.resetILC()

  def resetILC(self, d=None, P=None):
    self.kf_dpn.resetKF(d, P) # resets the Kalman Filter

  @property
  def N(self):
    return self.Nt if self.timeDomain else self.Nf  # length of the learned signal / nr of fourier coefs

  @property
  def d(self):
    return self.kf_dpn.d  # learned disturbance

  @property
  def P(self):
    return self.kf_dpn.P  # covariance for the learned disturbance

  def update_y_des(self, y_des):
    """ Function to update the time-domain desired trajectory.

    Args:
        y_des ([np.array(Nt, 1)]): The new time-domain desired trajectory
    """
    assert y_des.size == self.Nt, "Make sure the new y_des has the same size as the input ILC was initialized with."
    self.y_des = y_des.copy().reshape(-1,1)

    if self.lss.sys.with_feedback:
      self.y_des_feedback = self.lss.GF_feedback.dot(self.y_des)

  def transf_uff(self, uff):
    """ Transforms uff to time format for robot usage.

    Args:
        uff ([np.array(N{t/f},1)]): feedForward input in timeDomain or freqDomain according to how ILC was initialized

    Returns:
        [np.array(Nt,1)]: copy of feedForward input in timeDomain
    """
    return uff.copy() if self.timeDomain else series_real_coeff(uff, t=np.arange(0, self.Nt)*self.dt, T=self.T).reshape(-1,1)

  def get_delta_y(self, y_meas):
    """ Computes the input deviation trajectory in the right format.
    Args:
        y_meas ([np.array(Nt, 1)]): Measured time-domain output for which to calculate delta_y

    Returns:
        [np.array(Nt(Nf), 1)]: delta_y in the right format(time- or freq-domain)
    """
    return self.y_des-y_meas if self.timeDomain else fourier_series_coeff(self.y_des-y_meas-self.y_des_feedback, self.N).reshape(-1,1)

  def init_uff_from_lin_model(self, lb=None, ub=None, verbose=False):
    """Initialize the feeedforward input for the first iteration

    Args:
        lb ([float], optional): lower and upper bounds when calculating the feedforward input(used only for the time-domain ILC)
        verbose (bool, optional): Whether to print verbose information.

    Returns:
        [np.array(Nt(Nf), 1)]: Feedforward input in either time-domain or freq-domain format
    """
    if not self.timeDomain:
      assert lb is None and ub is None, "No constraint optimization possible in freqDomain."

    # update uff
    if self.timeDomain:
      self._u_ff = self.quad_input_optim.calcDesiredInput(np.zeros_like(self.kf_dpn.d), self.y_des.reshape(-1, 1), print_norm=verbose, lb=lb, ub=ub)
    else:
      self._u_ff = np.linalg.pinv(self.lss.GF).dot(self.get_delta_y(0))
    return self.transf_uff(self._u_ff)

  # Different ILC
  def updateStep(self, y_meas, y_des=None, verbose=False, lb=None, ub=None):
    """ Updates learned feedforward input and disturbance according to (self._u_ff, y_meas) tuple.

    Args:
        y_meas ([np.array(Nt, 1)]): measured output for the previously calculated u_ff
        y_des ([np.array(Nt, 1)], optional): A new desired desired trajectory in time domain.
                                             If we intend to slightly change the desired trajectory during learning.
        verbose (bool, optional): Whether to print out verbose info
        lb, ub ([type], optional): Whether to inforce upper and lower bounds onthe input. Only valid for timeDomain usage.

    Returns:
        [np.array(Nt, 1)]: new feed forward input
    """
    if not self.timeDomain:
      assert lb is None and ub is None, "No constraint optimization possible in freqDomain."

    if y_des is not None:
      self.update_y_des(y_des)

    if self._u_ff is None:
      return self.init_uff_from_lin_model(lb, ub, verbose)

    # estimate first delta_u as 0 and use the above uff computed from the lin_model
    if self._delta_u_ff is None:
      self._delta_u_ff = self._u_ff.copy()*0.

    disturbance = self.kf_dpn.updateStep(self._delta_u_ff, self.get_delta_y(y_meas))
    # update uff
    self._delta_u_ff = self.quad_input_optim.calcDesiredInput(disturbance, self.get_delta_y(y_meas)*0., print_norm=verbose, lb=lb, ub=ub)
    # self._u_ff = self._delta_u_ff

    return self.transf_uff(self._u_ff - self._delta_u_ff)

  def updateStep2(self, y_meas, y_des=None, verbose=False, lb=None, ub=None):
      """ Updates learned feedforward input and disturbance according to (self._u_ff, y_meas) tuple.

      Args:
          y_meas ([np.array(Nt, 1)]): measured output for the previously calculated u_ff
          y_des ([np.array(Nt, 1)], optional): A new desired desired trajectory in time domain.
                                               If we intend to slightly change the desired trajectory during learning.
          verbose (bool, optional): Whether to print out verbose info
          lb, ub ([type], optional): Whether to inforce upper and lower bounds onthe input. Only valid for timeDomain usage.

      Returns:
          [np.array(Nt, 1)]: new feed forward input
      """
      delta_y = self.get_delta_y(y_meas)
      print(np.linalg.norm(delta_y))

      if not self.timeDomain:
        assert lb is None and ub is None, "No constraint optimization possible in freqDomain."

      if y_des is not None:
        self.update_y_des(y_des)

      if self._u_ff is None:
        return self.init_uff_from_lin_model(lb, ub, verbose)

      # update uff
      self._u_ff += 0.363 *np.linspace(1.0, 0.2, self._u_ff.size).reshape(self._u_ff.shape)*(np.linalg.pinv(self.lss.GF).dot(delta_y))

      return self.transf_uff(self._u_ff)

  def updateStep3(self, y_meas, y_des=None, verbose=False, lb=None, ub=None):
      """ Updates learned feedforward input and disturbance according to (self._u_ff, y_meas) tuple.

      Args:
          y_meas ([np.array(Nt, 1)]): measured output for the previously calculated u_ff
          y_des ([np.array(Nt, 1)], optional): A new desired desired trajectory in time domain.
                                               If we intend to slightly change the desired trajectory during learning.
          verbose (bool, optional): Whether to print out verbose info
          lb, ub ([type], optional): Whether to inforce upper and lower bounds onthe input. Only valid for timeDomain usage.

      Returns:
          [np.array(Nt, 1)]: new feed forward input
      """
      delta_y = self.get_delta_y(y_meas)
      print(np.linalg.norm(delta_y))

      if not self.timeDomain:
        assert lb is None and ub is None, "No constraint optimization possible in freqDomain."

      if y_des is not None:
        self.update_y_des(y_des)

      if self._u_ff is None:
        return self.init_uff_from_lin_model(lb, ub, verbose)

      if self.c_s_ is None:
        cks = delta_y.real[0:self.Nf]
        sks = -delta_y.imag[0:self.Nf]
        self.c_s_ = np.hstack((cks, sks))

      # update uff
      def T(k):
        a = 17.
        K = a/4.
        tmp = k* 2*np.pi/self.T
        A = np.linalg.inv(
          # np.array([[tmp,  0,    0,    -1.],
          #           [0 ,   tmp,  a*K,   a],
          #           [0 ,   -1.,   tmp,   0],
          #           [a*K,   a,    0,   tmp]]).reshape(4,4))
          np.array([[0,    -1.,    tmp,    0.],
                    [a*K,   a,  0.,   tmp],
                    [-tmp ,  0.,  0,    -1.],
                    [0.,   -tmp,  a*K,   a]]).reshape(4,4))
        t_ret = np.eye(2)*0.0

        t_ret[0,0] = A[0,1]
        t_ret[0,1] = A[0,3]
        t_ret[0,1] = A[2,1]
        t_ret[1,1] = A[2,3]
        return t_ret

      aks = delta_y.real[0:self.Nf]
      bks = -delta_y.imag[0:self.Nf]
      a_b_ = np.hstack((aks, bks))

      for i, ab in enumerate(a_b_):
        k = i+1
        self.c_s_[i] -= 0.163 * (T(k).dot(ab.reshape(2,1))).squeeze()
      print(self.c_s_)
      # return self.transf_uff(self._u_ff)
      return series_real_coeff3(self.c_s_[:,0], self.c_s_[:,1], t=np.arange(0, self.Nt)*self.dt, T=self.T).reshape(-1,1)



# Helper functions to calculate frequency domain components
def fourier_series_coeff(f, Nf, complex=False):
  return fourier_series_coeff2(f, Nf+1, True)[1:]


def series_real_coeff(y, t, T):
    """calculates the Fourier series with period T at times t,
    from the real coeff. a0,a,b"""
    tmp = np.zeros_like(t)
    for k, (ck, sk) in enumerate(zip(y.real, -y.imag)):
        tmp += ck * np.cos(2 * np.pi * (k + 1) * t / T) + sk * np.sin(2 * np.pi * (k + 1) * t / T)
    return tmp


# Unchanged implementation that do and undo eachother
def fourier_series_coeff2(f, Nf, complex=True):
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

def series_real_coeff2(y, t, T):
    """calculates the Fourier series with period T at times t,
    from the real coeff. a0,a,b"""
    tmp = np.ones_like(t) * y[0].real / 2.
    for k, (ak, bk) in enumerate(zip(y[1:].real, -y[1:].imag)):
        tmp += ak * np.cos(2 * np.pi * (k + 1) * t / T) + bk * np.sin(2 * np.pi * (k + 1) * t / T)
    return tmp

def series_real_coeff3(a, b, t, T):
    """calculates the Fourier series with period T at times t,
    from the real coeff. a0,a,b"""
    tmp = np.zeros_like(t)
    for k, (ak, bk) in enumerate(zip(a, b)):
        tmp += ak * np.cos(2 * np.pi * (k + 1) * t / T) + bk * np.sin(2 * np.pi * (k + 1) * t / T)
    return tmp



if __name__ == "__main__":
  import matplotlib.pyplot as plt
  x = np.linspace(0,3,400)
  fx = (x-1.5)**6 - 1.5**6  # second order polynomial

  F = fourier_series_coeff2(fx, 40, complex=True)
  fxx= series_real_coeff2(F, x , T=3.)


  plt.plot(x, fx)
  plt.plot(x, fxx)
  # plt.plot(x, np.fft.irfft(F)*40 /2 )
  plt.show()