import numpy as np
from settings import m_b, m_p, g, k_c, alpha
from abc import ABCMeta, abstractmethod

class DynamicSystem:
  """A abstract class implementing a dynamic system.
  """
  __metaclass__ = ABCMeta
  def __init__(self, dt, x0, freq_domain=False, **kwargs):
    # TimeDomain
    self.dt = dt           # time step
    self.x0 = x0         # initial state (xb0, xp0, ub0, up0)

    self.Ad = None
    self.Bd = None
    self._Ad_impact = None # [nx, nx]
    self.Bd_impact = None  # [nx, nu]
    self.c_impact = None   # constants from gravity ~ dt, g, mp mb
    self.Cd = None         # [ny, nx]
    self.S = None          # [nx, ndup]
    self.c = None

    # FreqDomain
    self.Hu = None  # C(sI - A)^-1 B
    self.Hd = None  # C(sI - A)^-1 Bd

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
  def Ad_impact(self):
    assert self._Ad_impact is not None, "Impact dynamics are not implemented."
    return self._Ad_impact

  def initTransferFunction(self, **kwargs):
    assert False, "The frequence domain TransferFunction is not implemented for this dynamical system."

class BallAndPlateDynSys(DynamicSystem):
  def __init__(self, dt, x0, input_is_velocity=True):
    DynamicSystem.__init__(self, dt=dt, x0=x0, input_is_velocity=input_is_velocity)

  def initDynSys(self, dt, input_is_velocity=True):
    if input_is_velocity:
      self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)
      self._Ad_impact, self.Bd_impact, self.Cd, self.S, self.c_impact = self.getSystemMarixesVelocityControl(dt, True)
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
  def __init__(self, dt, x0, alpha_=alpha):
    self.alpha = alpha_
    DynamicSystem.__init__(self, dt=dt, x0=x0)

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
    S = np.array([1.0, 0.0], dtype='float').reshape(2,1)
    c = np.array([0.0, 0.0], dtype='float').reshape(2,1)

    return Ad, Bd, Cd, S, c


class ApolloDynSysIdeal(DynamicSystem):
  def __init__(self, dt, x0):
    DynamicSystem.__init__(self, dt, x0=x0)

  def initDynSys(self, dt):
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
    S = np.array([0.0, 1.0], dtype='float').reshape(2,1)
    c = np.array([0.0, 0.0], dtype='float').reshape(2,1)

    return Ad, Bd, Cd, S, c

  def initTransferFunction(self):
    # x_dot = A*x + B*u + S*d
    # y = C*x
    # |
    # X(s) = C(sI -A)^-1*B * U(s) + C(sI -A)^-1*S * D(s)
    A = np.array([0.0,  1.0,
                  0.0,  -self.alpha], dtype='float').reshape(2,2)
    B = np.array([0.0,  self.alpha], dtype='float').reshape(2,1)
    C = np.array([1.0, 0.0], dtype='float').reshape(1,2)
    self.S = np.array([0.0, 1.0], dtype='float').reshape(2,1)

    # self.Hu = lambda s: C.dot(np.linalg.pinv(s*np.eye(2) - A )).dot(B).squeeze()
    # self.Hd = lambda s: C.dot(np.linalg.pinv(s*np.eye(2) - A )).dot(self.S).squeeze()

    self.Hu = lambda s: self.alpha /(s*(s+self.alpha))
    self.Hd = lambda s: 1.0 /(s*(s+self.alpha))
    
class ApolloDynSysWithFeedback(DynamicSystem): #TODO: update Lifted Space to allow delay
  def __init__(self, dt, x0, alpha_=alpha, K=None, freq_domain=False):
    self.alpha = alpha_
    self.K = 2**-2 * self.alpha if K is None else K
    DynamicSystem.__init__(self, dt, x0=x0, freq_domain=freq_domain)

  def initDynSys(self, dt):
    self.Ad, self.Bd, self.Cd, self.S, self.c = self.getSystemMarixesVelocityControl(dt)

  def getSystemMarixesVelocityControl(self, dt):
    # x_k = A*x_k-1 + B*u_k-1 + Bd*d_k + Bn*n_k + c
    # y_k = C*x_k
    adt = self.alpha*dt
    tmp = 0.5*adt*dt
    a =  self.K*tmp + adt - 2.0 # 2.0 - self.alpha*dt + tmp*self.K
    b =  self.K*tmp - adt + 1.0

    A = np.array([-a, -b,
                  1., 0.], dtype='float').reshape(2,2)

    B = tmp/self.K* np.array([1., 1.,
                              0., 0.], dtype='float').reshape(2,2)

    C = np.array([1.0, 0.0], dtype='float').reshape(1,2)
    Bd = self.alpha**-1 * B
    Bn = np.array([0.0, 1.0], dtype='float').reshape(2,1)
    c = np.array([0.0, 0.0], dtype='float').reshape(2,1)

    return A, B, C, Bd, c

  def initTransferFunction(self):
    # x[k+1] = A*x[k] + B*u[k] + S*d[k]
    # y = C*x
    # |
    # X(z) = C(zI -A)^-1*B * U(z) + C(zI -A)^-1*Bd * D(z) <-- z transform
    # |
    # X(kw) = cu_k * U(kw) +  cd_k D(kw)  <-- discrete fourier transformation
    pass