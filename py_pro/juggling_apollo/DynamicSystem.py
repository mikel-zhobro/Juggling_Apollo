import numpy as np
from settings import m_b, m_p, g, k_c


class DynamicSystem:
  def __init__(self, dt, input_is_velocity):
    self.dt = dt    # time step

    self.Ad = None
    self.Ad_impact = None  # [nx, nx]
    self.Bd = None
    self.Bd_impact = None  # [nx, nu]
    self.Cd = None         # [ny, nx]
    self.S = None          # [nx, ndup]
    self.c = None
    self.c_impact = None   # constants from gravity ~ dt, g, mp mb
    self.x0 = None         # initial state (xb0, xp0, ub0, up0)
    self.initDynSys(input_is_velocity, dt)
    pass

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
