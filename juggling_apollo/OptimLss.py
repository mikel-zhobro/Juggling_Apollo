import numpy as np


class OptimLss:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(self, d, y_des, print_norm=False):
    # Solves  GFu = ydes -(GKd + Gd0)
    # as optimization u_des = argmin_u||[ydes - (GKd + Gd0)] - GFu||_2 +0.00000001|u|_2
    # u_des = np.linalg.lstsq((self.lss.GF.T).dot(self.lss.GF) + 0.0000000*np.eye(self.lss.GF.shape[1]), 
    #                          self.lss.GF.T.dot(y_des - self.lss.GK.dot(d) - self.lss.Gd0), rcond=None)[0]
    u_des = np.linalg.lstsq(self.lss.GF, y_des - self.lss.GK.dot(d) - self.lss.Gd0, rcond=None)[0]
    
    if print_norm:
      y = self.lss.GF.dot(u_des) + self.lss.Gd0  # predicted y
      print("The norm of the optimization error is:", np.linalg.norm(y - y_des))
    return u_des
