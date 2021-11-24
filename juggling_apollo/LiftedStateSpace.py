import numpy as np
from math import e

class LiftedStateSpace:
  # Constructor
  def __init__(self, sys, T, N, freq_domain, **kwargs):
    self.sys = sys
    self.N = N  # length of traj
    self.T = T  # time length of traj

    # Time domain
    self.timeDomain = False
    self.x0  = np.array(sys.x0).reshape(-1, 1)
    self.G   = None       # [ny*N, nx*N]
    self.GF  = None      # G * F
    self.GK  = None      # G * k
    self.Gd0 = None      # G * d0

    # Freq Domain
    self.freqDomain = freq_domain
    self.updateQuadrProgMatrixes(self.freqDomain)

  def updateQuadrProgMatrixes(self, freqDomain, **kwargs):
    self.freqDomain = freqDomain
    if freqDomain:
      self.updateQuadrProgMatrixesFreqDomain(**kwargs)
    else:
      self.updateQuadrProgMatrixesTimeDomain(**kwargs)

  def updateQuadrProgMatrixesFreqDomain(self, **kwargs):
    """[summary]
    Args:
        T ([type]): periode of the periodic output
        Nf ([type]): number of samples in freq domain
                     Nyquist Crit: fs >= 2*f_max = 2*f0*Nf <=> dt <= 1/(2*f0*Nf)  <=> Nf <= 1/(2*f0*dt)= T/(2*dt)
    """
    assert self.N <= 0.5*self.T/self.sys.dt, "Make sure that Nf{} is small enough{} to satisfy the Nyquist criterium.".format(self.N, 0.5*self.T/self.sys.dt)
    w0 = 2*np.pi/self.T
    self.Gd0 = 0.0
    self.GF = np.diag([self.sys.Hu(complex(0.0,-k*w0)).squeeze() for k in range(0, self.N)])  # SISO only
    self.GK = np.diag([self.sys.Hd(complex(0.0,-k*w0)).squeeze() for k in range(0, self.N)])

  def updateQuadrProgMatrixesTimeDomain(self, impact_timesteps=None, **kwargs):
    """ Updates the lifted state space matrixes G, GF, GK, Gd0

    Args:
        N[(int)]: nr of steps
        impact_timesteps ([tuple]): ith element is 1/True if there is an impact at timestep i
    """
    # impact_timesteps{t} = False if no impact, = True if impact for the timesteps 0 -> N-1
    # sizes
    if impact_timesteps is not None:
      assert self.N == len(impact_timesteps)
    else:
      impact_timesteps = [False]*self.N

    N    = self.N
    nx   = self.sys.Ad.shape[1]
    ny   = self.sys.Cd.shape[0]
    nu   = self.sys.Bd.shape[1]
    ndup = self.sys.S.shape[1]

    # calculate I, A_1, A_2*A_1, .., A_N-1*A_N-2*..*A_1
    A_power_holder    = [None] * N
    A_power_holder[0] = np.eye(nx, dtype='float')
    for i in range(N-1):
        A_power_holder[i+1] = self.get_Ad(impact_timesteps[i+1]).dot(A_power_holder[i])

    # Create lifted-space matrixes F, K, G, M:
    #    x[1:] = Fu + Kdu_p + d0,
    #    y[1:] = Gx,
    # where the constant part
    #    d0 = L*x0_N-1 + M*c0_N-1

    # F = [B0          0        0  .. 0
    #      A1B0        B1       0  .. 0
    #      A2A1B0      A1B1     B2 .. 0
    #        ..         ..         ..
    #      AN-1..A1B0  AN-2..A1B1  .. B0N-1]
    F = np.zeros((nx*N, nu*N), dtype='float')
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
    self.G = np.zeros((ny*N, nx*N), dtype='float')
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

    A_0 = self.get_Ad(impact_timesteps[0])
    for ll in range(N):
      self.G[ll*ny:(ll+1)*ny, ll*nx:(ll+1)*nx]  = self.sys.Cd
      L[ll*nx:(ll+1)*nx, ll*nx:(ll+1)*nx]       = A_power_holder[ll]*A_0
      for m in range(ll+1):
        M[ll*nx:(ll+1)*nx, m*nx:(m+1)*nx]       = A_power_holder[ll-m]
        F[ll*nx:(ll+1)*nx, m*nu:(m+1)*nu]       = A_power_holder[ll-m].dot(self.get_Bd(impact_timesteps[m]))  # F_lm
        K[ll*nx:(ll+1)*nx, m*ndup:(m+1)*ndup]   = A_power_holder[ll-m].dot(self.sys.S)

    # Create d0 = L*x0_N-1 + M*c0_N-1
    c_vec = np.vstack([self.get_c(impact) for impact in impact_timesteps])
    d0    = L.dot(np.tile(self.x0, [N, 1])) + M.dot(c_vec)

    # Prepare matrixes needed for the quadratic problem and KF
    self.GF  = self.G.dot(F)
    self.GK  = self.G.dot(K)
    self.Gd0 = self.G.dot(d0)

  def get_Ad(self, impact):
    return self.sys.Ad_impact if impact else self.sys.Ad

  def get_Bd(self, impact):
    return self.sys.Bd_impact if impact else self.sys.Bd

  def get_c(self, impact):
    return self.sys.c_impact if impact else self.sys.c
