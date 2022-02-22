import matplotlib.pyplot as plt
import numpy as np
import time
import os

import __add_path__
from ApolloILC.settings import dt
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics
from utils import save, plot_info

np.set_printoptions(precision=4, suppress=True)


print("juggling_apollo")
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

# 0. Create Apollo objects
rArmKinematics = ApolloArmKinematics(r_arm=True)  ## noise noisifies the forward dynamics only
q_start = rArmKinematics.IK(T_home)
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

N_1 = 4300
T_FULL=N_1*dt-0.00001



# C. LEARN BY ITERATING
# Learn Throw
ILC_it = 1  # number of ILC iteration

# a. System Trajectories
joints_q_vec   = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_vq_vec  = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_aq_vec  = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# c. Measurments
joint_torque_vec     = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')

# ILC Vectors Init
u_ff = [None] * N_joints
y_meas = np.zeros((N_1, N_joints), dtype='float')

####################################################################################################################################
q_traj, q_v_traj, q_a_traj, F_N_vec, u_vec, _ = rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL, u=np.zeros((N_1,7,1)), joint_home_config=q_start, repetitions=1)

# Collect Data
joints_q_vec[0]       = q_traj
joints_vq_vec[0]      = q_v_traj
joints_aq_vec[0]      = q_a_traj
joint_torque_vec[0]   = F_N_vec


plot_info(dt, joint_torque_vec=joint_torque_vec,
          v=False, p=False, dp=False, e_xyz=False, e=False, torque=True)


# Saving Results
dir_name = "/home/apollo/Desktop/Investigation/TorqueTest/{}".format(time.strftime("%Y_%m_%d"))
if not os.path.exists(dir_name):
  os.makedirs(dir_name)

filename = "examplesReal/dataReal/TorqueTest/torque_" + time.strftime("%Y_%m_%d-%H_%M_%S.txt")
fname = dir_name + "/torque_" + time.strftime("%H_%M_%S")

plot_info(0.004, joint_torque_vec=joint_torque_vec,
        v=False, p=False, dp=False, e_xyz=False, e=False, torque=True, fname=fname)

plt.show()