import numpy as np
import time

import __add_path__
from juggling_apollo.utils import steps_from_time, plt
from juggling_apollo.JugglingPlanner import calc
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys2
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, save, colors, line_types, plot_info, load


np.set_printoptions(precision=4, suppress=True)


print("juggling_apollo")
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

# 0. Create Apollo objects
rArmKinematics = ApolloArmKinematics(r_arm=True)  ## noise noisifies the forward dynamics only
q_start = rArmKinematics.IK_best(T_home)
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

N_1 = 2300
T_FULL=N_1*dt-0.00001



# C. LEARN BY ITERATING
# Learn Throw
ILC_it = 1  # number of ILC iteration

# a. System Trajectories
joints_q_vec   = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')
joints_vq_vec  = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')
joints_aq_vec  = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')
# c. Measurments
joint_torque_vec     = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')

# ILC Vectors Init
u_ff = [None] * N_joints
y_meas = np.zeros((N_1, N_joints), dtype='float')

####################################################################################################################################
q_traj, q_v_traj, q_a_traj, F_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=np.zeros((N_1,7,1)), joint_home_config=q_start, repetitions=1)

# Collect Data
joints_q_vec[0]       = q_traj
joints_vq_vec[0]      = q_v_traj
joints_aq_vec[0]      = q_a_traj
joint_torque_vec[0]   = F_N_vec


plot_info(dt, joint_torque_vec=joint_torque_vec,
          v=False, p=False, dp=False, e_xyz=False, e=False, torque=True)


SAVING = True
if SAVING:
  # Saving Results
  filename = "examplesReal/dataReal/TorqueTest/torque_" + time.strftime("%Y_%m_%d-%H_%M_%S.txt")
  save(filename,
       T_home=T_home,                                                                    # Home
       joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
       joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
       dt=dt
       )  # ILC parameters

  with open('examplesReal/dataReal/TorqueTest/list_files.txt', 'a') as f:
    f.write(filename + "\n")

