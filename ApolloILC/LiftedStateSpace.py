#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   LiftedStateSpace.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   unrolls the state space equations and creates the LSS mappings for the ILC
'''

import numpy as np
from math import e


class LiftedStateSpace:
  # Unroll the state space equations of the dynamic system to build the mappings on iteration level.
    #    x[1:] = Fu + Kdu_p + d0 + F_feedback*ydes,
    #    y[1:] = Gx,
  def __init__(self, sys, N, T=None, freq_domain=False):
    self.sys = sys  # dynamic system
    self.N = N      # length of traj (Nf if fre_domain)
    self.T = T      # time length of traj

    # Time domain
    self.x0  = np.array(sys.x0).reshape(-1, 1) # initial state
    self.GF  = None          # G * F  <- u
    self.GK  = None          # G * k  <- d
    self.Gd0 = None          # G * d0 <- x0
    self.GF_feedback = None  # G * F_feedback <- y_des

    # Freq Domain(u, d here are fourier coefficients)
    self.freqDomain = freq_domain
    self.updateQuadrProgMatrixes(self.freqDomain)

  def updateQuadrProgMatrixes(self, freqDomain):
    self.freqDomain = freqDomain
    if freqDomain:
      self.updateQuadrProgMatrixesFreqDomain()
    else:
      self.updateQuadrProgMatrixesTimeDomain()

  def updateQuadrProgMatrixesFreqDomain(self):
    """ Updates the matrixes describing the liftes state space equations
    Args:
        T ([type]): periode of the periodic output
        Nf ([type]): number of samples in freq domain
                     Nyquist Crit: fs >= 2*f_max = 2*f0*Nf <=> dt <= 1/(2*f0*Nf)  <=> Nf <= 1/(2*f0*dt)= T/(2*dt)
    """
    assert self.N <= 0.5*self.T/self.sys.dt, "Make sure that Nf{} is small enough{} to satisfy the Nyquist criterium.".format(self.N, 0.5*self.T/self.sys.dt)
    w0 = 2*np.pi/self.T
    self.Gd0 = 0.0
    self.GF = np.diag([self.sys.Hu(complex(0.0,k*w0)) for k in range(1, self.N+1)])  # SISO only
    self.GK = np.diag([self.sys.Hd(complex(0.0,k*w0)) for k in range(1, self.N+1)])
    if self.sys.with_feedback:
      self.GF_feedback = np.diag([self.sys.H_feedback(complex(0.0,k*w0)) for k in range(1, self.N+1)])

  def updateQuadrProgMatrixesTimeDomain(self):
    """ Updates the lifted state space matrixes G, GF, GK, Gd0
    """
    N    = self.N
    nx   = self.sys.Ad.shape[1]
    ny   = self.sys.Cd.shape[0]
    nu   = self.sys.Bd.shape[1]
    ndup = self.sys.S.shape[1]

    # calculate I, A_1, A_2*A_1, .., A_N-1*A_N-2*..*A_1
    A_power_holder    = [None] * N
    A_power_holder[0] = np.eye(nx, dtype='float')
    for i in range(N-1):
        A_power_holder[i+1] = self.sys.Ad.dot(A_power_holder[i])

    # Create lifted-space matrixes F, K, G, M:
    #    x[1:] = F u + K d + d0 + F_feedback ydes,
    #    y[1:] = Gx,
    # where the constant part
    #    d0 = L*x0_N-1 + M*c0_N-1

    # F = [B0          0        0  .. 0
    #      A1B0        B1       0  .. 0
    #      A2A1B0      A1B1     B2 .. 0
    #        ..         ..         ..
    #      AN-1..A1B0  AN-2..A1B1  .. B0N-1]
    F = np.zeros((nx*N, nu*N), dtype='float')
    # F = [B_feedback0          0                 0             ..  0
    #      A1B_feedback0        B_feedback1       0             ..  0
    #      A2A1B_feedback0      A1B_feedback1     B_feedback2   ..  0
    #        ..         ..         ..
    #      AN-1..A1B_feedback  AN-2..A1B1  .. B0N-1]
    F_feedback = np.zeros((nx*N, ny*N), dtype='float')
    # ---------- uncomment if dup is disturbance on dPN -----------
    # K = [S          0       0 .. 0
    #      A1S        S       0 .. 0
    #      A2A1S      A1S     S .. 0
    #        ..       ..        ..
    #      AN-1..A1S AN-2..A1S  .. S]
    K = np.zeros((nx*N, ndup*N), dtype='float')
    # -------------------------------------------------------------
    # G = [C  0  .  .  0
    #      0  C  0  .  0
    #      .  .  .  .  .
    #      0  0  0  .  C]
    G = np.zeros((ny*N, nx*N), dtype='float')
    # M = [I         0      0 .. 0
    #      A1        I      0 .. 0
    #      A2A1      A1     I .. 0
    #       ..       ..       ..
    #      AN-1..A1 AN-2..A1  .. I]
    M = np.zeros((nx*N, nx*N), dtype='float')
    # L = [A0 0     ..        0
    #      0  A1A0  ..        0
    #      ..  ..   ..
    #      0   0    ..   AN-1AN-2..A0]
    L = np.zeros((nx*N, nx*N), dtype='float')

    A_0 = self.sys.Ad
    for ll in range(N):
      G[ll*ny:(ll+1)*ny, ll*nx:(ll+1)*nx]  = self.sys.Cd
      L[ll*nx:(ll+1)*nx, ll*nx:(ll+1)*nx]       = A_power_holder[ll]*A_0
      for m in range(ll+1):
        M[ll*nx:(ll+1)*nx, m*nx:(m+1)*nx]       = A_power_holder[ll-m]
        F[ll*nx:(ll+1)*nx, m*nu:(m+1)*nu]       = A_power_holder[ll-m].dot(self.sys.Bd)  # F_lm
        K[ll*nx:(ll+1)*nx, m*ndup:(m+1)*ndup]   = A_power_holder[ll-m].dot(self.sys.S)
        if self.sys.with_feedback:
          F_feedback[ll*nx:(ll+1)*nx, m*ny:(m+1)*ny] = A_power_holder[ll-m].dot(self.sys.B_feedback)  # F_lm

    # Create d0 = L*x0_N-1 + M*c0_N-1
    c_vec = np.vstack([self.sys.c for _ in range(self.N)])
    d0    = L.dot(np.tile(self.x0, [N, 1])) + M.dot(c_vec)

    # Prepare matrixes needed for the quadratic problem and KF
    self.GF  = G.dot(F)
    self.GK  = G.dot(K)
    self.Gd0 = G.dot(d0)
    if self.sys.with_feedback:
      self.GF_feedback = G.dot(F_feedback)