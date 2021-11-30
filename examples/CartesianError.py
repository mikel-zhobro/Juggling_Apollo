import numpy as np

import __add_path__
import configs
from juggling_apollo.utils import steps_from_time, plt
from juggling_apollo.JugglingPlanner import calc
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys2
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, save, colors, line_types, print_info, plot_info

np.set_printoptions(precision=4, suppress=True)

FREQ_DOMAIN=False
end_repeat = 0   # repeat the last position value this many time
SAVING = False
UB = 0.87

# Learnable Joints
learnable_joints = [0,1,2,3,4,5,6]




# ILC Params
n_ms  = [3e-5]*7    # covariance of noise on the measurment
n_ds  = [6e-3]*7    # initial disturbance covariance
ep_s  = [1e-4]*7    # covariance of noise on the disturbance
n_ds  = [6e-2]*7
alpha = [16.0]*7
syss  = [ApolloDynSys2(dt, x0=np.zeros((2,1)), alpha_=a, freq_domain=FREQ_DOMAIN) for a in alpha]

# Cartesian Error propogation params
# rArmKinematics_nn:  kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
# rArmKinematics:     kinematics with noise (used for its (wrong)IK calculations)
damp            = 1e-12
mu              = 5e-2
CARTESIAN_ERROR = False

print("juggling_apollo")
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')


# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=0.0)  ## noise noisifies the forward dynamics only
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)  ## noise noisifies the forward dynamics only


# A. COMPUTE TRAJECTORIES IN CARTESIAN AND JOINT SPACE
N, delta_xyz_traj_des, thetas, mj = configs.get_minjerk_config(dt, end_repeat)
xyz_traj_des = delta_xyz_traj_des + T_home[:3, -1]
N_1 = N-1

q_traj_des, q_start, psi_params = rArmKinematics.seqIK(delta_xyz_traj_des, thetas, T_home)  # [N, 7]
q_traj_des_nn, q_start_nn, _    = rArmKinematics_nn.seqIK(delta_xyz_traj_des, thetas, T_home)  # [N, 7]


q_traj_des_i       = q_traj_des.copy()
q_start_i          = q_start.copy()
delta_q_traj_des_i = q_traj_des_i[1:] - q_start_i


if CARTESIAN_ERROR:
for i in range(N):
    J_invj          = np.linalg.pinv(rArmKinematics.J(q_traj[i+1])[:3,:])
    q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(d_xyz[i].reshape(3, 1))
# Update desired trajectory
q_start_i = q_traj_des_i[0]
delta_q_traj_des_i = q_traj_des_i - q_start_i