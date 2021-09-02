import numpy as np


class LiftedStateSpace:
  # Constructor
  def __init__(self, sys):
    self.sys = sys
    self.G = None       # [ny*N, nx*N]
    self.GF = None      # G * F
    self.GK = None      # G * k
    self.Gd0 = None     # G * d0
    # GFTGF_1 x*# ((G*F)^T * (G*F))^-1
    # F       # [nx*N, nu*N]
    # K       # [nN, ndup*N]
    # d0      # [nx*N, 1]

  def updateQuadrProgMatrixes(self, set_of_impact_timesteps):
    # set_of_impact_timesteps{t} = 1 if no impact, = 2 if impact for the timesteps 0 -> N-1
    # sizes
    N = len(set_of_impact_timesteps)    # nr of steps
    nx = self.sys.Ad.shape[1]
    ny = self.sys.Cd.shape[0]
    nu = self.sys.Bd.shape[1]
    ndup = self.sys.S.shape[1]

    # calculate I, A_1, A_2*A_1, .., A_N-1*A_N-2*..*A_1
    A_power_holder = [None] * N
    A_power_holder[0] = np.eye(nx)
    for i in range(N-1):
        A_power_holder[i+1] = self.get_Ad(set_of_impact_timesteps(i+1)) * A_power_holder[i]

    # Create lifted-space matrixes F, K, G, M:
    #    x = Fu + Kdu_p + d0,
    #    y = Gx,
    # where the constant part
    #    d0 = L*x0_N-1 + M*c0_N-1

    # F = [B0          0        0  .. 0
    #      A1B0        B1       0  .. 0
    #      A2A1B0      A1B1     B2 .. 0
    #        ..         ..         ..
    #      AN-1..A1B0  AN-2..A1B1  .. B0N-1]
    F = np.zeros((nx*N, nu*N))
    # ---------- uncomment if dup is disturbance on dPN -----------
    # K = [S          0       0 .. 0
    #      A1S        S       0 .. 0
    #      A2A1S      A1S     S .. 0
    #        ..       ..        ..
    #      AN-1..A1S AN-2..A1S  .. S]
    K = np.zeros((nx*N, ndup*N))
    # -------------------------------------------------------------
    # G = [Cd 0  .. .. 0
    #      0  Cd 0  .. 0
    #      .. .. .. .. ..
    #      0  0  0  .. Cd]
    self.G = np.zeros((ny*N, nx*N))
    # M = [I         0      0 .. 0
    #      A1        I      0 .. 0
    #      A2A1      A1     I .. 0
    #       ..       ..       ..
    #      AN-1..A1 AN-2..A1  .. I]
    M = np.zeros((nx*N, nx*N))
    # L = [A0 0     ..        0
    #      0  A1A0  ..        0
    #      ..  ..   ..
    #      0   0    ..   AN-1AN-2..A0]
    L = np.zeros((nx*N, nx*N))

    A_0 = self.get_Ad(set_of_impact_timesteps(1))

    for ll in range(N):
      self.G[(ll-1)*ny+1:ll*ny, (ll-1)*nx+1:ll*nx] = self.sys.Cd
      L[(ll-1)*nx+1:ll*nx, (ll-1)*nx+1:ll*nx] = A_power_holder[ll]*A_0
      for m in range(ll):
        M[(ll-1)*nx+1:ll*nx, (m-1)*nx+1:m*nx] = A_power_holder[ll-m+1]
        F[(ll-1)*nx+1:ll*nx, (m-1)*nu+1:m*nu] = A_power_holder[ll-m+1] * self.get_Bd(set_of_impact_timesteps(m))  # F_lm
        K[(ll-1)*nx+1:ll*nx, (m-1)*ndup+1:m*ndup] = A_power_holder[ll-m+1] * self.sys.S

    # Create d0 = L*x0_N-1 + M*c0_N-1
    # c_vec = transpose(cell2mat(arrayfun(@(ii){transpose(self.get_c(ii))}, set_of_impact_timesteps)))
    c_vec = np.vstack([self.get_c(impact) for impact in set_of_impact_timesteps])
    d0 = L * np.tile(self.sys.x0, [N, 1]) + M * c_vec

    # Prepare matrixes needed for the quadratic problem and KF
    self.GF = self.G * F
    self.GK = self.G * K
    self.Gd0 = self.G * d0

  def get_Ad(self, impact):
    return self.sys.Ad_impact if impact else self.sys.Ad

  def get_Bd(self, impact):
    return self.sys.Bd_impact if impact else self.sys.Bd

  def get_c(self, impact):
    return self.sys.c_impact if impact else self.sys.c
