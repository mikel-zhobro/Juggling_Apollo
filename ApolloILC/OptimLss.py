#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   OptimLss.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   The optimization problem to compute feedforward input for the next iteration.
'''

import numpy as np
from utils import plt


class OptimLss:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(self, d, delta_y_des, print_norm=False, lb=None, ub=None):
    """ The LSS dictates what we want to optimize on:
              delta_y = GF delta_u + GK d + G d0 + GF_feedback ydes
        min_{delta_u} ||GF delta_u + GK d + G d0 + GF_feedback ydes||_2
        after adding a scaling matrix(weigt), penalty input's amplitude(sigma), and loss for fast changes of input(mu)
        we can write the minimization problem in its general form:
              delta_u_des = argmin_u ||Weight[(GKd + Gd0 - delta_y_des) + GFdelta_u]||_2 + sigma*||delta_u||_2 + mu*||(I-I_1) delta_u||_2
        Which corresponds in setting the first gradient to 0, i.e. solving
              (GF^T*W*GF + sigma*I + mu*(I-I_1)) * delta_u = GF^T* W * (delta_y_des - GKd - Gd0)  <=> Au = b

    Args:
        d (np.array(N,1)): disturbance trajectory
        delta_y_des (np.array(N,1)): deviation(is non-zero only the first time) trajectory
        print_norm (bool, optional): verbose . Defaults to False.
        lb, ub (float, optional): Lower and upper bounds for the input.

    Returns:
        np.array(N,1): input deviation trajectory (from the initial input trajectory)
    """
    N_y = delta_y_des.shape[0]
    N_u = self.lss.GF.shape[1]

    N_impo = N_y//4
    Weight = np.eye(N_y)
    # Weight[N_y-N_impo:, N_y-N_impo:] *= 10.0

    if lb is None and ub is None:
      P = (self.lss.GF.T).dot(Weight).dot(self.lss.GF)   # close to the desired output N_u x N_u
      Q = 0.00001*np.eye(N_u)                            # possibly small inputs  N_u x N_u
      S = 0.0000001*(np.eye(N_u)-np.eye(N_u, k=1))       # slow changes  N_u x N_u
      A = P + Q + S
      b = self.lss.GF.T.dot(Weight).dot(delta_y_des - self.lss.GK.dot(d) - self.lss.Gd0)
      delta_u_des = np.linalg.inv(A).dot(b)
    else:
      # here we solve the least squared error optimization problem with inequality constraines
      # we use the unconstrained solution as initial solution
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
                    bounds=[(lb, ub) for i in range(N_u)]).x.reshape(-1,1)

      print("Change after constraints", np.linalg.norm(x0 - delta_u_des))

    if print_norm:
      y = self.lss.GF.dot(delta_u_des) + self.lss.Gd0  + self.lss.GK.dot(d) # predicted y
      plt.plot(delta_y_des, label='desired', color='b')
      plt.plot(y, label='optimized', color='b', linestyle="--")
      plt.legend()
      plt.show()
      print("The norm of the optimization error is:", np.linalg.norm(y - delta_y_des))
    return delta_u_des
