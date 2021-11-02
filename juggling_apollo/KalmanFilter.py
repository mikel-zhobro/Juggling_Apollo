import numpy as np


class KalmanFilter:
  def __init__(self, lss, M, d0, P0, epsilon0, epsilon_decrease_rate=0.9):
    # initial values
    self.d0 = d0
    self.P0 = P0
    self.epsilon0 = epsilon0

    # pointer to the lifted space (can be changed from outside)
    self.lss = lss
    self.M = M
    self.epsilon_decrease_rate = epsilon_decrease_rate

    # current values
    self._d = None
    self._P = None
    self.epsilon = None  # di= di_0 + ni with ni~N(0,Omega) <- Omega = epsilon*eye(N)
    self.Ident = np.eye(self.d0.size)

  @property
  def d(self):
    if self._d is None:
      raise ValueError("The Kalman Filter is not yet reseted(d is not initialized))")
    else:
      return self._d

  def resetKF(self):
      self._d = self.d0
      self._P = self.P0
      self.epsilon = self.epsilon0

  def updateStep(self, u, y_meas):
    # In this case
    # d0 ~ N(P0)
    # d = d + n_d                        with n_d ~ N(eps*I)
    # y = Fu + Gd0  + ((( GKd ))) + n_y  with n_y ~ N(0, M)
    P1_0 = self._P + self.Ident * self.epsilon
    Theta = self.lss.GK.dot(P1_0).dot(self.lss.GK.T) + self.M
    K = P1_0.dot(self.lss.GK.T).dot(np.linalg.inv(Theta))
    self._P = (self.Ident - K.dot(self.lss.GK)).dot(P1_0)
    self._d = self._d + K.dot(y_meas - self.lss.Gd0 - self.lss.GK.dot(self._d) - self.lss.GF.dot(u))

    # update epsilon
    self.epsilon *= self.epsilon_decrease_rate
    return self._d
