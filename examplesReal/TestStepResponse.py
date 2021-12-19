import numpy as np
import time
import os

import __add_path__
from juggling_apollo.settings import dt
from apollo_interface.Apollo_It import ApolloInterface
from kinematics.ApolloKinematics import ApolloArmKinematics
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
rArmInterface = ApolloInterface(r_arm=True)


# C. LEARN BY ITERATING
# Learn Throw
ILC_it = 1  # number of ILC iteration
N_1 = 1000; T_FULL=N_1*dt
N_step = 150; N_start = (N_1-N_step)//2



step_value = -0.6
try_out_joints = [
  [[0],[1],[2],[3],[4],[5],[6]],
  [list(range(7))],
  ]

for step_value in [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]:
  q_v_traj_vec = np.zeros((1, N_1, N_joints, 1))
  uff_collect = np.zeros((1, N_1, N_joints, 1))

  for learnable_joints_l in try_out_joints:
    for learnable_joints in learnable_joints_l:

      # Prepare input
      u_ff = np.zeros((N_1,7,1))
      for i in learnable_joints:
        u_ff[N_start:N_start+N_step,i,0] = -step_value if i !=1 else step_value


      q_traj, q_v_traj, q_a_traj, F_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start, repetitions=1)

      for i in learnable_joints:
        uff_collect[0,:,i] = (180.0 / np.pi) * u_ff[:,i]
        q_v_traj_vec[0,:,i] = (180.0 / np.pi) * q_v_traj[1:,i]

    dir_name = "/home/apollo/Desktop/Investigation/Step_Response/{}".format(time.strftime("%Y_%m_%d"))
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    plot_info(dt, learnable_joints=[l for ll in learnable_joints_l for l in ll],
              u_ff_vec=uff_collect, q_v_traj=q_v_traj_vec[0],
              v=True, p=False, dp=False, e_xyz=False, e=False, torque=False, N=1,
              fname=dir_name + "/v_{}_joint_{}".format(str((180.0 / np.pi) * abs(step_value)).replace('.', ''), learnable_joints_l))


