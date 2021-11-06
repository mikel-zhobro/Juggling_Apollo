import numpy as np
from settings import m_b, m_p, g, k_c, alpha


class DynamicSystem:
  """A abstract class implementing a dynamic system.
  """
  def __init__(self, dt, input_is_velocity):
    self.dt = dt    # time step
    self.Ad = None
    self.Bd = None
    self._Ad_impact = None  # [nx, nx]
    self.Bd_impact = None  # [nx, nu]
    self.c_impact = None   # constants from gravity ~ dt, g, mp mb
    self.Cd = None         # [ny, nx]
    self.S = None          # [nx, ndup]
    self.c = None
    self.x0 = None         # initial state (xb0, xp0, ub0, up0)

    self.initDynSys(dt, input_is_velocity)

  def initDynSys(self, dt, input_is_velocity):
    assert False, "The 'initDynSys' abstract method is not implemented for the used subclass."

  @property
  def Ad_impact(self):
    assert self._Ad_impact is not None, "Impact dynamics are not implemented."
    return self._Ad_impact

class BallAndPlateDynSys(DynamicSystem):

  def initDynSys(self, dt, input_is_velocity):
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


class ApolloDynSys(DynamicSystem):
  def __init__(self, dt, input_is_velocity=True, alpha_=alpha):
    self.alpha = alpha_
    DynamicSystem.__init__(self, dt, input_is_velocity)

  def initDynSys(self, dt, input_is_velocity=True):
    assert input_is_velocity, "For apollo only dynamic system with velocity input is provided."
    self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)

  def getSystemMarixesVelocityControl(self, dt):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k
    Ad = np.array([1.0, dt*(1-self.alpha*dt/2),
                   0.0, 1-dt*self.alpha], dtype='float').reshape(2,2)
    Bd = np.array([self.alpha*dt**2/2,
                   dt*self.alpha], dtype='float').reshape(2,1)
    Cd = np.array([1.0, 0.0], dtype='float').reshape(1,2)
    S = np.array([1.0, 0.0], dtype='float').reshape(2,1)
    c = np.array([0.0, 0.0], dtype='float').reshape(2,1)

    return Ad, Bd, Cd, S, c
  
  
class ApolloDynSysIdeal(DynamicSystem):
  def __init__(self, dt, input_is_velocity=True):
    DynamicSystem.__init__(self, dt, input_is_velocity)

  def initDynSys(self, dt, input_is_velocity=True):
    assert input_is_velocity, "For apollo only dynamic system with velocity input is provided."
    self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)

  def getSystemMarixesVelocityControl(self, dt):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k
    Ad = np.array([1.0], dtype='float').reshape(1,1)
    Bd = np.array([dt], dtype='float').reshape(1,1)
    Cd = np.array([1.0], dtype='float').reshape(1,1)
    S = np.array([1.0], dtype='float').reshape(1,1)
    c = np.array([0.0], dtype='float').reshape(1,1)

    return Ad, Bd, Cd, S, c
  
  
class ApolloDynSys2(DynamicSystem):
  def __init__(self, dt, alpha_=alpha, input_is_velocity=True):
    self.alpha = alpha_
    DynamicSystem.__init__(self, dt, input_is_velocity)

  def initDynSys(self, dt, input_is_velocity=True):
    assert input_is_velocity, "For apollo only dynamic system with velocity input is provided."
    self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)

  def getSystemMarixesVelocityControl(self, dt):
    # x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
    # y_k = Cd*x_k
    Ad = np.array([1.0,  dt,         dt**2/2,
                   0.0,  1.0,        dt,
                   0.0, -self.alpha, 0.0], dtype='float').reshape(3,3)
    Bd = np.array([0.0,
                   0.0,
                   self.alpha], dtype='float').reshape(3,1)
    Cd = np.array([0.0, 1.0, 0.0], dtype='float').reshape(1,3)
    S = np.array([1.0, 0.0, 0.0], dtype='float').reshape(3,1)
    c = np.array([0.0, 0.0, 0.0], dtype='float').reshape(3,1)

    return Ad, Bd, Cd, S, c