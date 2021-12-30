from random import expovariate
import numpy as np
import time
import matplotlib.pyplot as plt


import __add_path__
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys, ApolloDynSysWithFeedback
from apollo_interface.Apollo_It import ApolloInterface
from kinematics.ApolloKinematics import ApolloArmKinematics
from kinematics import utilities
from Planners import SiteSwapPlanner, OneBallThrowPlanner
from utils import plot_A, save, save_all, colors, line_types, print_info, plot_info

np.set_printoptions(precision=4, suppress=True)

# PARAMS
print("juggling_apollo")
########################################################################################################
########################################################################################################
########################################################################################################
FREQ_DOMAIN=False
NF=6

SAVING = True
UB = 6.5

CARTESIAN_ERROR = False
NOISE=0.0

ILC_it = 22                                # number of ILC iteration
end_repeat = 0  if not FREQ_DOMAIN else 0 # repeat the last position value this many time

# Learnable Joints
N_joints = 7
learnable_joints = [0,1,2,3,4,5,6]
non_learnable_joints = list(set(range(N_joints)) - set(learnable_joints))

# ILC Params
n_ms  = np.ones((N_joints,))* 3e-3;   # covariance of noise on the measurment
n_ms[:] = 3e-3
# n_ms[:] = 1e-4
n_ds  = [1e-3]*N_joints               # initial disturbance covariance
ep_s  = [1e-3]*N_joints               # covariance of noise on the disturbance
alpha = np.ones((N_joints,)) * 18.0
alpha[:] = 17.
syss  = [ApolloDynSys(dt, x0=np.zeros((2,1)), alpha_=a, freq_domain=FREQ_DOMAIN) for a in alpha]

# Cartesian Error propogation params
damp            = 1e-12
mu              = 0.35

# Home Configuration
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.41],
                   [-1.0, 0.0, 0.0, -0.69],
                   [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
T_home = np.array([[0.0, -1.0, 0.0,  0.47],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.52],
                   [-1.0, 0.0, 0.0, -0.7],
                   [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
T_dhtcp_tcp = np.eye(4)
T_dhtcp_tcp[:3,:3] = T_home[:3,:3].T
########################################################################################################
########################################################################################################
########################################################################################################

########################################################################################################
# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=NOISE)  ## kinematics with noise (used for its (wrong)IK calculations)
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)               ## kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)

# C) PLANNINGs
if False:
    for z in np.arange(-0.4, -0.2, 0.05):
        for y in np.arange(0.8, 1., 0.05):
            for x in np.arange(0.3, 0.5, 0.05):

                T_home = np.array([[0.0, -1.0, 0.0,  x],  # uppword orientation(cup is up)
                                [0.0,  0.0, 1.0,     y],
                                [-1.0, 0.0, 0.0,     z],
                                [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
                raw_input("Press Enter to continue...")
                try:
                    qh = rArmKinematics.IK(T_home)
                    rArmInterface.go_to_home_position(qh, 100)
                    print(x,y,z)
                except:
                    print("Not vaild home")
                    print(x,y,z)


# C) Rotate around shoulder-wrist axis
if True:
    T_home = np.array([[0.0, -1.0, 0.0,  0.3],  # uppword orientation(cup is up)
                    [0.0,  0.0, 1.0,  0.9],
                    [-1.0, 0.0, 0.0, -0.5],
                    [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
    _, _, _, _, _, solu, feasible_set = rArmKinematics.IK(T_home, for_seqik=True)
    startt = feasible_set.b; endd = feasible_set.a
    rArmInterface.go_to_home_position(solu(feasible_set.b), 2000)

    raw_input("Press Enter to continue...")
    for i in range(4):
        for psi in np.linspace(startt, endd, 200):
            qh = solu(psi)
            rArmInterface.go_to_home_position(qh, it_time=10, eps=3., wait=1, zero_speed= (psi==endd), verbose=False)
            
        tmp = endd
        endd =startt
        startt = tmp

# q_traj_des, T_traj = OneBallThrowPlanner.plan2(dt, T_home, kinematics=rArmKinematics, verbose=True)


# N = len(q_traj_des)
# q_start = q_traj_des[0]
# T_home = T_traj[0]

########################################################################################################

    """
    (0.3, 0.8999999999999999, -0.5000000000000001)
    (0.4, 0.8999999999999999, -0.5000000000000001)
    (0.3, 0.8999999999999999, -0.4000000000000001)
    (0.4, 0.8999999999999999, -0.4000000000000001)
    (0.3, 0.9, -0.30000000000000004)
    (0.4, 0.9, -0.30000000000000004)
    (0.3, 1.0, -0.30000000000000004)
    """