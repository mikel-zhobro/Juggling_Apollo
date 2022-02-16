#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   DynamicSystem.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   In this file we implement a class to describe the dynamic system.
             That is done by saving the state space equation in the form
             x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
             y_k = Cd*x_k + n_k
             where Ad, Bd, Cd are the standard ss matrixes,
             S describes how we model the disturbance (how it enters the plant)
             and c is some constant(e.g. gravity).
'''

import numpy as np
from settings import m_b, m_p, g, k_c, alpha
from abc import ABCMeta, abstractmethod

class DynamicSystem:
  """An abstract base-class implementation for dynamic systems.
  """
  __metaclass__ = ABCMeta
  def __init__(self, dt, x0, freq_domain=False, **kwargs):
    # TimeDomain
    self.dt = dt           # time step
    self.x0 = x0         # initial state (xb0, xp0, ub0, up0)

    self.Ad = None
    self.Bd = None
    self.Cd = None         # [ny, nx]
    self.S = None          # [nx, ndup]
    self.c = None          # constants from gravity ~ dt, g, mp mb

    # FreqDomain
    self.Hu = None  # C(sI - A)^-1 B
    self.Hd = None  # C(sI - A)^-1 Bd

    # Feedback Sys
    self.B_feedback = None

    if freq_domain:
      try:
        self.initTransferFunction(**kwargs)
      except:
        assert False, "No freq domain equations present."
    else:
      try:
        self.initDynSys(dt, **kwargs)
      except:
        assert False, "No time domain equations present."

  @abstractmethod
  def initDynSys(self, dt, **kwargs):
    assert False, "The 'initDynSys' abstract method is not implemented for the used subclass."

  @property
  def with_feedback(self):
    return self.B_feedback is not None

  def initTransferFunction(self, **kwargs):
    assert False, "The frequence domain TransferFunction is not implemented for this dynamical system."

class BallAndPlateDynSys(DynamicSystem):
  """ Dynamic system for the simulated ball and plate system.
  """
  def __init__(self, dt, x0, input_is_velocity=True):
    DynamicSystem.__init__(self, dt=dt, x0=x0, input_is_velocity=input_is_velocity)

  def initDynSys(self, dt, input_is_velocity=True):
    if input_is_velocity:
      self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)
    else:
      self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesForceControl(dt)

  def getSystemMarixesVelocityControl(self, dt, contact_impact=False):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k

    if contact_impact:
      mbp = 1/(m_p+m_b)
    else:
      mbp = 0

    dt_2 = 0.5 * dt
    Ad = np.array([[1, 0, dt-dt_2*m_p*mbp, dt_2*m_p*mbp*(1 - dt*k_c)],
                   [0, 1, dt_2*m_b*mbp,    dt_2*(2 - dt*k_c - m_b*mbp*(1 - dt*k_c))],
                   [0, 0, 1-m_p*mbp,       m_p*mbp*(1 - dt*k_c)],
                   [0, 0, m_b*mbp,         1 - dt*k_c - m_b*mbp*(1 - dt*k_c)]], dtype='float')

    Bd = np.array([[dt_2*dt*mbp*m_p*k_c],
                   [k_c*dt_2*dt*(1-m_b*mbp)],
                   [dt*mbp*m_p*k_c],
                   [dt*k_c*(1-m_b*mbp)]], dtype='float')

    c = np.array([[-dt_2*dt*g*(1-m_p*mbp)],
                  [-dt_2*dt*g*m_b*mbp],
                  [-dt*g*(1-m_p*mbp)],
                  [-dt*g*m_b*mbp]], dtype='float')

    Cd = np.array([0, 1, 0, 0], dtype='float').reshape(1, -1)

    # S = np.array([[-dt_2/m_b],
    #               [dt_2/m_p],
    #               [-1/m_b],
    #               [1/m_p]], dtype='float')
    S = np.array([0, 1, 0, 0], dtype='float').reshape(-1, 1)
    return Ad, Bd, Cd, S, c

  def getSystemMarixesForceControl(self, dt, contact_impact=False):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k

    if contact_impact:
      mbp = 1/(m_p+m_b)
    else:
      mbp = 0
    dt_2 = 0.5 * dt
    # xb xp ub up
    Ad = np.array([[1, 0, dt-dt_2*m_p*mbp,   dt_2*m_p*mbp],
                   [0, 1, dt_2*m_b*mbp,      dt_2*(2 - m_b*mbp)],
                   [0, 0, 1-m_p*mbp,         m_p*mbp],
                   [0, 0, m_b*mbp,           1-m_b*mbp]], dtype='float')

    Bd = np.array([[dt_2*dt*mbp],
                   [1/m_p*dt_2*dt*(1-m_b*mbp)],
                   [dt*mbp],
                   [dt/m_p*(1-m_b*mbp)]], dtype='float')

    c = np.array([[-dt_2*dt*g*(1-m_p*mbp)],
                  [-dt_2*dt*g*m_b*mbp],
                  [-dt*g*(1-m_p*mbp)],
                  [-dt*g*m_b*mbp]], dtype='float')

    Cd = np.array([0, 1, 0, 0], dtype='float').reshape(1, -1)

    # S = np.array([[-dt_2/m_b],
    #               [dt_2/m_p],
    #               [-1/m_b],
    #               [1/m_p]], dtype='float')

    S = np.array([0, 1, 0, 0], dtype='float').reshape(-1, 1)
    return Ad, Bd, Cd, S, c

class ApolloDynSys(DynamicSystem):
  """ Estimated velocity controlled dynamic system for Apollo.
  """
  def __init__(self, dt, x0, alpha_=alpha, freq_domain=False):
    self.alpha = alpha_
    DynamicSystem.__init__(self, dt, x0=x0, freq_domain=freq_domain)

  def initDynSys(self, dt):
    self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)

  def getSystemMarixesVelocityControl(self, dt):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k
    Ad = np.array([1.0, dt*(1-self.alpha*dt/2),
                   0.0, 1-dt*self.alpha], dtype='float').reshape(2,2)
    Bd = np.array([self.alpha*dt**2/2,
                   dt*self.alpha], dtype='float').reshape(2,1)
    Cd = np.array([1.0, 0.0], dtype='float').reshape(1,2)
    # S = np.array([dt**2/2, dt], dtype='float').reshape(2,1) # torque level
    # S = np.array([0.0, 1.0], dtype='float').reshape(2,1)
    S = np.array([dt/2., 1.], dtype='float').reshape(2,1) # velocity level
    c = np.array([0.0, 0.0], dtype='float').reshape(2,1)

    return Ad, Bd, Cd, S, c

  def initTransferFunction(self):
    # X(s) = Hu(s) * U(s) + Hd(s) * S * D(s)
    self.S = np.array([0.0, 1.0], dtype='float').reshape(2,1)

    self.Hu = lambda s: self.alpha /(s*(s+self.alpha))
    self.Hd = lambda s: 1.0 /(s*(s+self.alpha))


class ApolloDynSysWithFeedback(DynamicSystem):
  """ Estimated velocity controlled dynamic system with feedback for Apollo.
  """
  def __init__(self, dt, x0, alpha_=alpha, K=None, freq_domain=False):
    self.alpha = alpha_
    self.K = self.alpha /4. if K is None else K
    DynamicSystem.__init__(self, dt, x0=x0, freq_domain=freq_domain)

  def initDynSys(self, dt):
    self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)

  def getSystemMarixesVelocityControl(self, dt):
    # State is: [yk, yk-1, dk-1, uk-1]
    # x_k = A*x_k-1 + B*u_k-1 + Bd*d_k + Bn*n_k + c
    # y_k = C*x_k
    adt = self.alpha*dt
    adtdt_2 = adt*dt /2.0

    A = np.array([1. + self.K*adtdt_2, dt - adtdt_2,
                  self.K*adt, 1. - adt], dtype='float').reshape(2,2)

    B =  self.K* np.array([adtdt_2, adt], dtype='float').reshape(2,1)
    Bd = np.array([0.5*dt, 1.], dtype='float').reshape(2,1)

    C = np.array([1., 0.], dtype='float').reshape(1,2)
    c = np.array([0., 0.], dtype='float').reshape(2,1)
    Bn = np.array([1.0], dtype='float').reshape(1,1)

    self.B_feedback = -B / self.K
    return A, B, C, Bd, c

  def initTransferFunction(self):
    # x[k+1] = A*x[k] + B*u[k] + S*d[k]
    # y = C*x
    # |
    # X(z) = C(zI -A)^-1*B * U(z) + C(zI -A)^-1*Bd * D(z) <-- z transform
    # |
    # X(kw) = cu_k * U(kw) +  cd_k D(kw)  <-- discrete fourier transformation
    self.Hu         = lambda s: self.alpha / (s**2 + self.alpha*s + self.alpha*self.K)
    self.Hd         = lambda s: 1.0 / (s**2 + self.alpha*s + self.alpha*self.K)
    self.H_feedback = lambda s: self.K*self.alpha / (s**2 + self.alpha*s + self.alpha*self.K)