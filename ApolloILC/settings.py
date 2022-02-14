#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   settings.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   contains physical parameters used in ILC
'''

global g, dt, m_b, m_p, k_c


# Constants
g = 9.80665     # [m/s^2] gravitational acceleration constant
dt = 0.004      # [s] discretization time step size

# Params
m_b = 0.1       # [kg] mass of ball
m_p = 10.0      # [kg] mass of plate
k_c = 10.0      # [1/s] force coefficient
ABS = 1e-5

# Apollo
alpha = 10