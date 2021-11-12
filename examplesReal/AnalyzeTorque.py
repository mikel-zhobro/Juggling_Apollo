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

N_1 = 5300
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
####################################################################################################################################
###############################################################################################################################


with open('examplesReal/dataReal/TorqueTest/list_files.txt') as topo_file:
    for filename in topo_file:
        filename = filename.strip()  # The comma to suppress the extra new line char
        
        ld = load(filename)

        plot_info(0.004, joint_torque_vec=ld.joint_torque_vec,
                v=False, p=False, dp=False, e_xyz=False, e=False, torque=True)