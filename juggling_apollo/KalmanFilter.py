import numpy as np


class KalmanFilter:
  def __init__(self, lss, M, d0, P0, epsilon0, freqDomain=False, epsilon_decrease_rate=0.9):
    
    self.timeDomain = not freqDomain
    # size of the state
    self.N = d0.size

    # pointer to the lifted space (can be changed from outside)
    self.lss = lss
    self.epsilon_decrease_rate = epsilon_decrease_rate
    
    imag = 1j if freqDomain else 0.0
    # initial values
    self.M = np.diag((1+imag)*M)            # covariance of noise on the measurment
    self.d0 = d0+0*imag          # initial disturbance value
    self.P0 = np.diag((1+imag)*P0)          # initial disturbance covariance
    self.epsilon0 = epsilon0

    # current values
    self._d = None
    self._P = None
    self.epsilon = None  # di= di_0 + ni with ni~N(0,Omega) <- Omega = epsilon*eye(N)
    self.Ident = (1+imag)*np.eye(self.d0.size)

  @property
  def d(self):
    if self._d is None:
      raise ValueError("The Kalman Filter is not yet reseted(d is not initialized))")
    else:
      return self._d

  @property
  def P(self):
    if self._P is None:
      raise ValueError("The Kalman Filter is not yet reseted(P is not initialized))")
    else:
      return self._P

  def resetKF(self, d=None, P=None):
      self._d = self.d0 if d is None else d
      # self._P = self.P0
      if P is None:
        P = self.P0
      self._P = self.lss.GK.dot(P).dot(self.lss.GK.T.conj()) if self.timeDomain else P
      self.epsilon = self.epsilon0

  def updateStep(self, u, y_meas):
    # In this case
    # d0 ~ N(P0)
    # d = d + n_d                        with n_d ~ N(eps*I)
    # y = Fu + Gd0  + ((( GKd ))) + n_y  with n_y ~ N(0, M)
    P1_0 = self._P + self.Ident * self.epsilon
    Theta = self.lss.GK.dot(P1_0).dot(self.lss.GK.T.conj()) + self.M
    K = P1_0.dot(self.lss.GK.T.conj()).dot(np.linalg.inv(Theta))

    # Weight
    if self.timeDomain:
      N = y_meas.size
      # D = np.diag(np.linspace(0,1.0,N))
      D = np.diag(np.log10(np.linspace(3.0,10.0,N)))
      # import matplotlib.pyplot as plt
      # plt.plot(np.log10(np.linspace(1.0,10.0,N)))
      # plt.show()
      K = D.dot(K)


    self._P = (self.Ident - K.dot(self.lss.GK)).dot(P1_0)
    self._d = self._d + K.dot(y_meas - self.lss.Gd0 - self.lss.GK.dot(self._d) - self.lss.GF.dot(u))

    # update epsilon
    self.epsilon *= self.epsilon_decrease_rate
    return self._d
