#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Throw_Learn.py
@Time    :   2022/02/21
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   A template for learning a simple catch and throw trick
'''


import numpy as np
import matplotlib.pyplot as plt

import __add_path__
from ApolloILC.settings import dt
from ApolloILC.ILC import ILC
from ApolloILC.DynamicSystem import ApolloDynSys, ApolloDynSysWithFeedback
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics
from ApolloKinematics import utilities
from ApolloPlanners import SiteSwapPlanner, OneBallThrowPlanner
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
T_home = np.array([[0.0, -1.0, 0.0,  0.3],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.9],
                   [-1.0, 0.0, 0.0, -0.5],
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
q_traj_des, T_traj = OneBallThrowPlanner.plan(dt, kinematics=rArmKinematics, verbose=False)


N = len(q_traj_des)
q_start = q_traj_des[0]
T_home = T_traj[0]

########################################################################################################


# INIT ILC
# ILC Works on differences(ie delta)
q_traj_des_i       = q_traj_des.copy()
q_start_i          = q_start.copy()
# q_start_i = q_traj_des_i[1]; #q_start_i[-1,0] = np.pi/4.
delta_q_traj_des_i = q_traj_des_i[1:] - q_traj_des_i[1]


# B. Initialize ILC
Nf = NF if FREQ_DOMAIN else N-1
def kf_params(n_m=0.02, epsilon=1e-5, n_d=0.06, d0=None, P0=None):
  P0 = n_d*np.ones(Nf, dtype='float') if P0 is None else P0
  d0 = np.zeros((Nf, 1), dtype='float') if d0 is None else d0
  kf_dpn_params = {
    'M': n_m*np.ones(Nf, dtype='float'),   # covariance of noise on the measurment
    'P0': P0,                              # initial disturbance covariance
    'd0': d0,                              # initial disturbance value
    'epsilon0': epsilon,                   # covariance of noise on the disturbance
    'epsilon_decrease_rate': 0.95          # the decreasing factor of noise on the disturbance
  }
  return kf_dpn_params

my_ilcs = [
  ILC(sys=syss[i], y_des=delta_q_traj_des_i[:, i], kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i]), freq_domain=FREQ_DOMAIN, Nf=Nf)                # make sure to make up for the initial state during learning
  for i in range(N_joints)]


# C. LEARN BY ITERATING
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
disturbanc_vec        = np.zeros([ILC_it, Nf, N_joints], dtype='float')
# c. Trajectory errors
cartesian_error_norms = np.zeros([ILC_it, 6, 1], dtype='float')  # (x,y,z,nx,ny,nz)
joint_error_norms     = np.zeros([ILC_it, N_joints, 1], dtype='float')
joints_d_vec          = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
d_xyz_rpy_vec         = np.zeros([ILC_it, N, 6], dtype='float')



# D. Main Loop
every_N = 2

# Use linear model to compute first input
u_ff   = np.zeros([N-1, N_joints, 1], dtype='float')
for i in learnable_joints:
  u_ff[:,i] = my_ilcs[i].init_uff_from_lin_model()

for j in range(ILC_it):
  # Limit Input
  # u_ff = np.clip(u_ff,-UB,UB)

  if False:
    plot_A([u_ff])
    plt.show()
  # Main Simulation
  # q_traj, q_v_traj, q_a_traj, F_N_vec, _, q0 = rArmInterface.apollo_run_one_iteration_with_feedback(dt=dt, T=T_FULL, u=u_ff, thetas_des=q_traj_des_i, joint_home_config=q_start_i, repetitions=3 if FREQ_DOMAIN else 1, it=j)
  q_traj, q_v_traj, q_a_traj, F_N_vec, _, q0 = rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start_i, repetitions=3 if FREQ_DOMAIN else 1, it=j)
  q_extra, qv_extra, q_des_extra, qv_des_extra = rArmInterface.measure_extras(dt, 2.3)
  q_traj   = np.average(  q_traj[0:], axis=0)
  q_v_traj = np.average(q_v_traj[0:], axis=0)
  q_a_traj = np.average(q_a_traj[0:], axis=0)
  F_N_vec  = np.average( F_N_vec[0:], axis=0)

  delta_y_meas = q_traj - q_traj[0]  #  np.average(q0[0,], axis=0) #np.average(q0[0:], axis=0, weights=[3,2,1,1]) # calc delta of executed traj
  q_traj = q_start_i + delta_y_meas   # rebase executed traj as if it would start from the exact home position

  # Update feed-forward signal
  for i in learnable_joints:
    if FREQ_DOMAIN:
      u_ff[:,i] = my_ilcs[i].updateStep2(y_meas=delta_y_meas[:,i],  y_des=delta_q_traj_des_i[:, i],
                                        #  lb=-UB,ub=UB
                                        verbose=False)
    else:
      u_ff[:,i] = my_ilcs[i].updateStep(y_meas=delta_y_meas[:,i],  y_des=delta_q_traj_des_i[:, i],
                                        #  lb=-UB,ub=UB
                                        verbose=False)

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
  disturbanc_vec[j]     = np.squeeze([ilc.d for ilc in my_ilcs]).T  # learned joint space disturbances
  q_traj_des_vec[j]     = q_traj_des_i
  # c. Errors
  d_xyz_rpy_vec[j]      = delta   # actual cartesian errors
  joints_d_vec[j]       = delta_q_traj_des_i-delta_y_meas         # actual joint space error
  joint_error_norms[j]  = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T / N_1
  cartesian_error_norms[j]  = np.linalg.norm(delta, axis=0, keepdims=True).T / N_1


  # For the next iteration
  if CARTESIAN_ERROR:
    for i in range(0, N):
      qi = q0 if i==0 else q_traj[i-1]
      di = delta[i].reshape(6,1)
      # J_invj          = np.linalg.pinv(rArmKinematics.J(qi)[:,:])
      Ji = rArmKinematics.J(qi)
      J_invj          = Ji.T.dot(np.linalg.inv(Ji.dot(Ji.T) + damp*np.eye(6)))
      q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(di[:])
    # Update desired trajectory
    q_start_i = q_traj_des_i[0]
    delta_q_traj_des_i = q_traj_des_i[1:] - q_start_i
    mu = 0.95*mu

  print_info(j, learnable_joints, joints_d_vec, d_xyz)


# After Main Loop has finished
if False:
  plot_info(dt, learnable_joints,
            joints_q_vec=joints_q_vec, q_traj_des=q_traj_des_i[1:],
            u_ff_vec=u_ff_vec, q_v_traj=q_v_traj, cartesian_error_norms = cartesian_error_norms,
            disturbanc_vec=disturbanc_vec, d_xyz_rpy_vec=d_xyz_rpy_vec, joint_error_norms=joint_error_norms,
            v=True, p=False, dp=False, e_xyz=False, e=True, N=min(4, ILC_it-1))

if SAVING:
  # Saving Results
  freq = 'freq_domain' if FREQ_DOMAIN else "time_domain"
  cartesian_err = 'cart_err_on' if CARTESIAN_ERROR else "cart_err_off"
  filename = "one_throw_joint_{}_alpha_{}_eps_{}_{}_{}".format(learnable_joints, alpha, ep_s[0], freq, cartesian_err)
  save_all(filename, kinematics=rArmKinematics,
       dt=dt,
       q_start=q_start_i, T_home=T_home,                                                 # Home
       T_traj=T_traj, q_traj_des_vec=q_traj_des_vec,                                     # Desired Trajectories
       joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
       joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
       disturbanc_vec=disturbanc_vec, u_ff_vec=u_ff_vec,                                 # Learned Trajectories (uff and disturbance)
       d_xyz_rpy_vec=d_xyz_rpy_vec, joints_d_vec=joints_d_vec,                                   # Progress Measurments
       joint_error_norms=joint_error_norms, cartesian_error_norms=cartesian_error_norms,
    #    pattern=pattern, h=h, r_dwell=r_dwell, throw_height=throw_height,                 # SiteSwap Params
    #    swing_size=swing_size, w=w, slower=slower, rep=rep,
       ilc_learned_params = [(ilc.d, ilc.P) for ilc in my_ilcs],
       learnable_joints=learnable_joints, alpha=alpha, n_ms=n_ms, n_ds=n_ds, ep_s=ep_s)  # ILC parameters



# Run Simulation with several repetition
if False:
  if end_repeat!=0:
    rArmInterface.apollo_run_one_iteration2(dt=dt, T=end_repeat*dt, u=u_ff[:end_repeat], joint_home_config=q_start_i, repetitions=1, it=j)
    rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL-end_repeat*dt, u=u_ff[end_repeat:], repetitions=5, it=j)
  else:
    rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start_i, repetitions=5, it=j)
