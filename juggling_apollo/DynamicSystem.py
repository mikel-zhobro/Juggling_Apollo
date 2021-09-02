import numpy as np


class DynamicSystem:

  def __init__(self, m_b, m_p, k_c, g, dt, input_is_velocity):
    self.m_b = m_b  # mass of ball
    self.m_p = m_p  # mass of plate
    self.k_c = k_c  # force coefficient
    self.g = g      # gravitational acceleration constant
    self.dt = dt    # time step

    self.Ad = None
    self.Ad_impact = None  # [nx, nx]
    self.Bd = None
    self.Bd_impact = None  # [nx, nu]
    self.Cd = None         # [ny, nx]
    self.S = None          # [nx, ndup]
    self.x0 = None         # initial state (xb0, xp0, ub0, up0)
    self.c = None
    self.c_impact = None   # constants from gravity ~ dt, g, mp mb

    self.initDynSys(input_is_velocity)

  def initDynSys(self, input_is_velocity, dt):
    if input_is_velocity:
      self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)
      self.Ad_impact, self.Bd_impact, self.Cd, self.S, self.c_impact = self.getSystemMarixesVelocityControl(dt, True)
    else:
      self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesForceControl(dt)
      self.Ad_impact, self.Bd_impact, self.Cd, self.S, self.c_impact = self.getSystemMarixesForceControl(dt, True)

  def getSystemMarixesVelocityControl(self, dt, contact_impact=False):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k

    if contact_impact:
      mbp = 1/(self.m_p+self.m_b)
    else:
      mbp = 0

    dt_2 = 0.5 * dt
    Ad = np.array([[1, 0, dt-dt_2*self.m_p*mbp, dt_2*self.m_p*mbp*(1 - dt*self.k_c)],
                   [0, 1, dt_2*self.m_b*mbp,    dt_2*(2 - dt*self.k_c - self.m_b*mbp*(1 - dt*self.k_c))],
                   [0, 0, 1-self.m_p*mbp,       self.m_p*mbp*(1 - dt*self.k_c)],
                   [0, 0, self.m_b*mbp,         1 - dt*self.k_c - self.m_b*mbp*(1 - dt*self.k_c)]])

    Bd = np.array([[dt_2*dt*mbp*self.m_p*self.k_c],
                   [self.k_c*dt_2*dt*(1-self.m_b*mbp)],
                   [dt*mbp*self.m_p*self.k_c],
                   [dt*self.k_c*(1-self.m_b*mbp)]])

    c = np.array([[-dt_2*dt*self.g*(1-self.m_p*mbp)],
                  [-dt_2*dt*self.g*self.m_b*mbp],
                  [-dt*self.g*(1-self.m_p*mbp)],
                  [-dt*self.g*self.m_b*mbp]])

    Cd = np.array([0, 1, 0, 0])

    # S = np.array([[-dt_2/self.m_b],
    #               [dt_2/self.m_p],
    #               [-1/self.m_b],
    #               [1/self.m_p]])
    S = np.array([0, 1, 0, 0]).T
    return Ad, Bd, Cd, S, c

  def getSystemMarixesForceControl(self, dt, contact_impact):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k

    if contact_impact:
      mbp = 1/(self.m_p+self.m_b)
    else:
      mbp = 0
    dt_2 = 0.5 * dt
    # xb xp ub up
    Ad = np.array([[1, 0, dt-dt_2*self.m_p*mbp,   dt_2*self.m_p*mbp],
                   [0, 1, dt_2*self.m_b*mbp,      dt_2*(2 - self.m_b*mbp)],
                   [0, 0, 1-self.m_p*mbp,         self.m_p*mbp],
                   [0, 0, self.m_b*mbp,           1-self.m_b*mbp]])

    Bd = np.array([[dt_2*dt*mbp],
                   [1/self.m_p*dt_2*dt*(1-self.m_b*mbp)],
                   [dt*mbp],
                   [dt/self.m_p*(1-self.m_b*mbp)]])

    c = np.array([[-dt_2*dt*self.g*(1-self.m_p*mbp)],
                  [-dt_2*dt*self.g*self.m_b*mbp],
                  [-dt*self.g*(1-self.m_p*mbp)],
                  [-dt*self.g*self.m_b*mbp]])

    Cd = np.array([0, 1, 0, 0])

    # S = np.array([[-dt_2/self.m_b],
    #               [dt_2/self.m_p],
    #               [-1/self.m_b],
    #               [1/self.m_p]])

    S = np.array([0, 1, 0, 0]).T
    return Ad, Bd, Cd, S, c
