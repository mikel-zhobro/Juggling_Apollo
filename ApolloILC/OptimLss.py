#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   OptimLss.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   the optimization problem to compute feedforward input for the next iteration
'''

import numpy as np
from utils import plt


class OptimLss:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(self, d, delta_y_des, print_norm=False, lb=None, ub=None):
    # TODO: for delta way, no need for delta_ydes
    # Solves  GFdelta_u = delta_ydes -(GKd + Gd0)
    # as optimization
    # delta_u = argmin_u |Weight[delta_ydes - (GKd + Gd0) - GFdelta_u]|_2 + sigma*|delta_u|_2 + mu*|(I-I_1) delta_u|_2
    # Which corresponds in setting the first gradient to 0, i.e. solving
    # (GF^T*W*GF + sigma*I + mu*(I-I_1)) * delta_u = GF^T* W * (delta_ydes - (GKd + Gd0))  <=> Au = b
    N_y = delta_y_des.shape[0]
    N_u = self.lss.GF.shape[1]

    N_impo = N_y//4
    Weight = np.eye(N_y)
    # Weight[N_y-N_impo:, N_y-N_impo:] *= 10.0

    if lb is None and ub is None:
      P = (self.lss.GF.T).dot(Weight).dot(self.lss.GF)   # close to the desired output N_u x N_u
      Q = 0.00001*np.eye(N_u)                             # possibly small inputs  N_u x N_u
      S = 0.0000001*(np.eye(N_u)-np.eye(N_u, k=1))        # slow changes  N_u x N_u
      A = P + Q + S
      b = self.lss.GF.T.dot(Weight).dot(delta_y_des - self.lss.GK.dot(d) - self.lss.Gd0)
      delta_u_des = np.linalg.inv(A).dot(b)
    else:
      from scipy.optimize import minimize
      def loss(x):
        y = self.lss.GF.dot(x.reshape(-1, 1)) + self.lss.Gd0  + self.lss.GK.dot(d) # predicted y
        return np.linalg.norm(y - delta_y_des)

      def jacob(x):
        return (self.lss.GF.T).dot(self.lss.GF.dot(x))

      def hessian(x):
        return (self.lss.GF.T).dot(self.lss.GF)


      x0 = np.clip(self.calcDesiredInput(d, delta_y_des), lb, ub)

      delta_u_des = minimize(loss, x0, method='trust-krylov', jac=jacob, hess=hessian,
                    bounds=[(lb, ub) for i in range(N)]).x.reshape(-1,1)

      print("Change after constraints", np.linalg.norm(x0 - delta_u_des))

    if print_norm:
      y = self.lss.GF.dot(delta_u_des) + self.lss.Gd0  + self.lss.GK.dot(d) # predicted y
      plt.plot(delta_y_des, label='desired', color='b')
      plt.plot(y, label='optimized', color='b', linestyle="--")
      plt.legend()
      plt.show()
      print("The norm of the optimization error is:", np.linalg.norm(y - delta_y_des))
    return delta_u_des
