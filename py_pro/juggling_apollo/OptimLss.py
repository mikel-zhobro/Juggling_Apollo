import numpy as np


class OptimLss:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(self, dup, y_des, print_norm=False):
    # Solving the problem u_des = argmin_u||GFu + GKdu_p + Gd0 - ydes||_2 +0.00000001|u|_2
    u_des = np.linalg.lstsq((self.lss.GF.T).dot(self.lss.GF) + 0.00000001*np.eye(self.lss.GF.shape[1]), 
                            -self.lss.GF.T.dot(self.lss.GK.dot(dup) + self.lss.Gd0 - y_des), rcond=None)[0]
    if print_norm:
      y = self.lss.GF.dot(u_des) + self.lss.Gd0  # predicted y
      print("The norm of the optimization error is:", np.linalg.norm(y - y_des))
    return u_des
