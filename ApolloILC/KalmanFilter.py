import numpy as np


class KalmanFilter:
  def __init__(self, lss, M, d0, P0, epsilon0, freqDomain=False, epsilon_decrease_rate=0.9):
    ## Prediction model
    # d_{j+1} = d_{j} + n_d_{j} with d_{0}~N(0,P0) and n_d_{j}~N(0,Dj) where Dj = epsilon_{j}*eye(N)
    ## Measurment model
    # y_{j} = Fu_{j} + Gd0  + ((( GKd_{j} ))) + n_y  with n_y ~ N(0, M)

    # Size of the state
    self.N = d0.size

    # In case the KF works in freq-domain its components should have an imaginary part, thats what we prepare here.
    self.timeDomain = not freqDomain
    imag = 1j if freqDomain else 0.0
    self.Ident = (1+imag)*np.eye(self.N)

    # Pointer to the lifted space (can be changed from outside)
    self.lss = lss
    self.epsilon_decrease_rate = epsilon_decrease_rate

    # Covariance of the noise on the measurment
    self.M = np.diag((1+imag)*M)
    # self.M = self.lss.GK.dot(np.diag((1+imag)*M)).dot(self.lss.GK.T.conj())  # to use if we want to initialize the measurement-noise covariance in disturbance level.

    # Initial values
    self.d0 = d0+0*imag                     # initial disturbance value
    self.P0 = np.diag((1+imag)*P0)          # initial disturbance covariance
    self.D0 = epsilon0*self.Ident           # covariance of noise on the disturbance

    # Current values
    self._dj = None
    self._Pj = None
    self._Dj = None

  @property
  def d(self):
    if self._dj is None:
      raise ValueError("The Kalman Filter is not yet reseted(d is not initialized))")
    else:
      return self._dj

  @property
  def P(self):
    if self._Pj is None:
      raise ValueError("The Kalman Filter is not yet reseted(P is not initialized))")
    else:
      return self._Pj

  def resetKF(self, d=None, P=None):
      self._dj = self.d0 if d is None else d
      # self._Pj = self.P0
      if P is None:
        P = self.P0
      self._Pj = self.lss.GK.dot(P).dot(self.lss.GK.T.conj()) if False and self.timeDomain else P               # TODO: Is this needed?
      self._Dj = self.lss.GK.dot(self.D0).dot(self.lss.GK.T.conj()) if False and self.timeDomain else self.D0   # TODO: Is this needed?

  def updateStep(self, u, y_meas):
    # In this case
    # d0 ~ N(P0)
    # d = d + n_d                        with n_d ~ N(eps*I)
    # y = Fu + Gd0  + ((( GKd ))) + n_y  with n_y ~ N(0, M)
    P1_0 = self._Pj + self._Dj
    Theta = self.lss.GK.dot(P1_0).dot(self.lss.GK.T.conj()) + self.M
    K = P1_0.dot(self.lss.GK.T.conj()).dot(np.linalg.pinv(Theta))

    # Weight (can be used to scale the Kalman gain. Here we force the KF to make very little changesif to make KF less agressive in the beginning of the disturbance trajectory)
    if False and self.timeDomain:
      N = y_meas.size
      # D = np.diag(np.linspace(0,1.0,N))
      D = np.diag(np.log10(np.linspace(3.0,10.0,N)))
      # import matplotlib.pyplot as plt
      # plt.plot(np.log10(np.linspace(1.0,10.0,N)))
      # plt.show()
      K = D.dot(K)

    # update estimated state and its covariance
    self._dj = self._dj + K.dot(y_meas - self.lss.Gd0 - self.lss.GK.dot(self._dj) - self.lss.GF.dot(u))
    self._Pj = (self.Ident - K.dot(self.lss.GK)).dot(P1_0)

    # update epsilon
    self._Dj *= self.epsilon_decrease_rate
    return self._dj
