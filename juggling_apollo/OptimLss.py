import numpy as np
from utils import plt

class OptimLss:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(self, d, y_des, print_norm=False):
    # Solves  GFu = ydes -(GKd + Gd0)
    # as optimization 
    # u_des = argmin_u |Weight[ydes - (GKd + Gd0) - GFu]|_2 + sigma*|u|_2 + mu*|(I-I_1) u|_2
    # Which corresponds in setting the first gradient to 0, i.e. solving
    # (GF^T*W*GF + sigma*I + mu*(I-I_1)) * u = GF^T*W * (ydes - (GKd + Gd0) - GFu])  <=> Au = b
    N = y_des.shape[0]
    N_impo = N//30
    Weight = np.eye(N)
    # Weight[N-N_impo:, N-N_impo:] *= 30.0
    
    P = (self.lss.GF.T).dot(Weight).dot(self.lss.GF)    # close to the desired output
    Q = 0.0000001*np.eye(N)                             # possibly small inputs
    S = 0.000001*(np.eye(N)-np.eye(N, k=1))                # slow changes
    A = P + Q + S
    b = self.lss.GF.T.dot(Weight).dot(y_des - self.lss.GK.dot(d) - self.lss.Gd0)
    
    # u_des = np.linalg.lstsq(A, b, rcond=None)[0]
    u_des = np.linalg.inv(A).dot(b)
    # u_des = np.linalg.lstsq((self.lss.GF.T).dot(self.lss.GF) + 0.0000001*np.eye(self.lss.GF.shape[1]), 
    #                          self.lss.GF.T.dot(y_des - self.lss.GK.dot(d) - self.lss.Gd0), rcond=None)[0]
    # u_des = np.linalg.lstsq(self.lss.GF, y_des - self.lss.GK.dot(d) - self.lss.Gd0, rcond=None)[0]
    
    if print_norm:
      y = self.lss.GF.dot(u_des) + self.lss.Gd0  + self.lss.GK.dot(d) # predicted y
      plt.plot(y_des, label='desired', color='b')
      plt.plot(y, label='optimized', color='b', linestyle="--")
      plt.legend()
      plt.show()
      print("The norm of the optimization error is:", np.linalg.norm(y - y_des))
    return u_des
