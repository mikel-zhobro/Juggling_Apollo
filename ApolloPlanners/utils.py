#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   None
'''

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from ApolloILC.utils import *
from examples.utils import plot_A
from ApolloILC.settings import g
from ApolloKinematics import utilities

def plan_ball_trajectory(hb, d1=0, d2=0):
    ub_0 = np.sqrt(2*g*(hb - d1))  # velocity of ball at throw point
    Tb = 2*ub_0/g + d2  # flying time of the ball
    return Tb, ub_0

def flyTime2HeightAndVelocity(Tfly):
  ub_0 = g*Tfly/2
  Hb = ub_0**2 /(2*g)
  return Hb, ub_0

def calc(tau, dwell_ration, E, slower=1.0):
  # times
  T_hand = tau * dwell_ration  # time the ball spends on the hand
  T_empty = tau - T_hand       # time the hand is free
  T_fly = 2*tau - T_hand       # time the ball is in the air

  # positions
  z_throw = 0.0
  z_catch = z_throw + E

  # ball height and velocity
  H, ub_throw = flyTime2HeightAndVelocity(T_fly)

  return T_hand*slower, T_empty*slower, ub_throw/slower, H, z_catch