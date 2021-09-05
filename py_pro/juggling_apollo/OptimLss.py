#!/usr/bin/env python3
import numpy as np


class OptimLss:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(self, dup, y_des, print_norm=False):
    # TODO: Add penalty on the input
                       # transpose(obj.lss.GF) * obj.lss.GF, -transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des));
    u_des = np.linalg.lstsq((self.lss.GF.T).dot(self.lss.GF), -self.lss.GF.T.dot(self.lss.GK.dot(dup) + self.lss.Gd0 - y_des), rcond=None)[0]
    if print_norm:
      print("The norm of the optimization error is:", np.linalg.norm((self.lss.GF.T).dot(self.lss.GF).dot(u_des) + self.lss.GF.T.dot((self.lss.GK.dot(dup) + self.lss.Gd0 - y_des))))
    return u_des
