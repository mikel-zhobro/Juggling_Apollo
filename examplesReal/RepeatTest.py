import numpy as np
import time
import matplotlib.pyplot as plt

import __add_path__
from ApolloILC.settings import dt
from ApolloILC.ILC import ILC
from ApolloILC.DynamicSystem import ApolloDynSys, ApolloDynSysWithFeedback
from ApolloPlanners import SiteSwapPlanner, OneBallThrowPlanner
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics
from ApolloKinematics import utilities
from utils import plot_A, save, load, save_all, colors, line_types, print_info, plot_info


np.set_printoptions(precision=4, suppress=True)
# PARAMS
print("juggling_apollo")
########################################################################################################
########################################################################################################
########################################################################################################
ILC_it = 12                                # number of ILC iteration
FREQ_DOMAIN=False
SAVING = True
end_repeat = 0  if not FREQ_DOMAIN else 0 # repeat the last position value this many time
FILENAME="/home/apollo/Desktop/Investigation/2021_12_15/15_03_14/one_throw_joint_[0, 1, 2, 3, 4, 5, 6]_alpha_[ 17.  17.  17.  17.  17.  17.  17.]_eps_0.001_time_domain_cart_err_off.data"


# Load data
########################################################################################################
ld = load(FILENAME)
T_traj = ld.T_traj
T_home = T_traj[0]

q_traj_des = np.concatenate((ld.q_traj_des_vec[-1], ld.q_traj_des_vec[0:1,0]))  # only needed for old backups
q_start = q_traj_des[0]
N = len(q_traj_des)

N_joints = 7
learnable_joints = ld.learnable_joints

u_ff = ld.u_ff_vec[-1]
########################################################################################################


# Create Apollo objects
########################################################################################################
# A) INTERFACE: create rArmInterface and go to home position
rArmInterface = ApolloInterface(r_arm=True)
# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)               ## kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
########################################################################################################


# C. LEARN BY ITERATING
########################################################################################################
T_FULL = N*dt - 0.002
N_1 = N-1
# Data collection
q_traj_des_vec        = np.zeros([ILC_it, N, N_joints, 1], dtype='float')
# a. Measurments
joints_q_vec          = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_vq_vec         = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_aq_vec         = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joint_torque_vec      = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# b. ILC Trajectories
u_ff_vec              = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# c. Trajectory errors
cartesian_error_norms = np.zeros([ILC_it, 6, 1], dtype='float')  # (x,y,z,nx,ny,nz)
joint_error_norms     = np.zeros([ILC_it, N_joints, 1], dtype='float')
joints_d_vec          = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
d_xyz_rpy_vec         = np.zeros([ILC_it, N, 6], dtype='float')
########################################################################################################



# D. Main Loop
########################################################################################################
every_N = 1

for j in range(ILC_it):
  # Main Simulation
  q_traj, q_v_traj, q_a_traj, F_N_vec, _, q0 = rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start, repetitions=3 if FREQ_DOMAIN else 1, it=j)
  # q_extra, qv_extra, q_des_extra, qv_des_extra = rArmInterface.measure_extras(dt, 2.3)
  q_traj   = np.average(  q_traj[0:], axis=0)
  q_v_traj = np.average(q_v_traj[0:], axis=0)
  q_a_traj = np.average(q_a_traj[0:], axis=0)
  F_N_vec  = np.average( F_N_vec[0:], axis=0)

  # Collect Data
  cartesian_traj_i = rArmKinematics_nn.seqFK(q_traj)
  delta = np.array([utilities.errorForJacobianInverse(T_i=rArmKinematics_nn.FK(q0), T_goal=T_traj[0])]+
                   [utilities.errorForJacobianInverse(T_i=cartesian_traj_i[i], T_goal=T_traj[i+1]) for i in range(N_1)]).reshape(-1, 6)
  d_xyz = delta[:, :3]  # measured cartesian error: calculated using the noise-less FK

  # a. Meas
  joints_q_vec[j]       = q_traj
  joints_vq_vec[j]      = q_v_traj
  joints_aq_vec[j]      = q_a_traj
  joint_torque_vec[j]   = F_N_vec
  # b. ILC
  u_ff_vec[j]           = u_ff
  q_traj_des_vec[j]     = q_traj_des
  # c. Errors
  d_xyz_rpy_vec[j]      = delta   # actual cartesian errors
  joints_d_vec[j]       = q_traj_des[1:] - q_traj         # actual joint space error
  joint_error_norms[j]  = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T
  cartesian_error_norms[j]  = np.linalg.norm(delta, axis=0, keepdims=True).T

  print_info(j, learnable_joints, joints_d_vec, d_xyz)

  if False and (j ==ILC_it-1 or j==0):
    plot_A([180./np.pi*q_extra, 180./np.pi*q_des_extra], labels=["observed", "desired"])
    plt.suptitle("{}. Joint angles it({})".format("freq" if FREQ_DOMAIN else "time", j))
    plt.savefig("/home/apollo/Desktop/Investigation/{}_Joint_angles_it{}.png".format("Freq" if FREQ_DOMAIN else "Time", j))
    plot_A([180./np.pi*qv_extra, 180./np.pi*qv_des_extra], labels=["observed", "desired"])
    plt.suptitle("{}. Joint velocities it({})".format("freq" if FREQ_DOMAIN else "time", j))
    # plt.savefig("/home/apollo/Desktop/Investigation/{}_Joint_velocities_it{}.png".format("Freq" if FREQ_DOMAIN else "Time", j))
    plt.show()
########################################################################################################


if SAVING:
  # Saving Results
  freq = 'freq_domain' if FREQ_DOMAIN else "time_domain"
  filename = "one_throw_joint_{}_{}".format(learnable_joints, freq)
  save_all(filename, special='REPEATABILITY_TEST_WITH_0_VEL_END',
       dt=dt,
       q_start=q_start, T_home=T_home,                                                   # Home
       T_traj=T_traj, q_traj_des_vec=q_traj_des_vec,                                     # Desired Trajectories
       joints_q_vec=joints_q_vec, joints_vq_vec=u_ff.reshape(1,-1,N_joints,1),                           # Joint Informations
       joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
       u_ff_vec=joints_vq_vec,                                                           # Learned Trajectories (uff and disturbance)
       d_xyz_rpy_vec=d_xyz_rpy_vec, joints_d_vec=joints_d_vec,                                   # Progress Measurments
       joint_error_norms=joint_error_norms, cartesian_error_norms=cartesian_error_norms,
       learnable_joints=learnable_joints)  # ILC parameters



# Run Simulation with several repetition
if False:
  if end_repeat!=0:
    rArmInterface.apollo_run_one_iteration2(dt=dt, T=end_repeat*dt, u=u_ff[:end_repeat], joint_home_config=q_start, repetitions=1, it=j)
    rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL-end_repeat*dt, u=u_ff[end_repeat:], repetitions=5, it=j)
  else:
    rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start, repetitions=5, it=j)
