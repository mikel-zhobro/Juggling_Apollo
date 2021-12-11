import numpy as np
import time

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


step_value = -1.0
try_out_joints = [0,1,2,3,4,5,6]
for i in try_out_joints:
  u_ff = np.zeros((N_1,7,1))
  if i==7:
    u_ff[N_start:N_start+N_step,:,0] = -step_value
  else:
    u_ff[N_start:N_start+N_step,i,0] = -step_value if i !=1 else step_value

  ####################################################################################################################################
  q_traj, q_v_traj, q_a_traj, F_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start, repetitions=1)


  learnable_joints = [i] if i<7 else list(range(7))
  plot_info(dt, learnable_joints=learnable_joints,
            u_ff_vec=u_ff[np.newaxis, :], q_v_traj=q_v_traj[1:],
            v=True, p=False, dp=False, e_xyz=False, e=False, torque=False, N=1,
            fname="/home/apollo/Desktop/Investigation/v_{}_joint_{}_{}".format(str(abs(step_value)).replace('.', ''), learnable_joints, time.strftime("%Y_%m_%d-%H_%M_%S")))


  SAVING = True
  if SAVING:
    # Saving Results
    filename = "examplesReal/dataReal/StepResponseTest/step_val_{}_joints_{}_".format(step_value, i if i<7 else 'all') + time.strftime("%Y_%m_%d-%H_%M_%S.txt")
    save(filename,
         u_ff=u_ff, learnable_joints = learnable_joints,
         T_home=T_home,                                                                    # Home
         q_v_traj=q_v_traj,                                                      # Joint Informations
         dt=dt
        )  # ILC parameters

    with open('examplesReal/dataReal/StepResponseTest/list_files.txt', 'a') as f:
      f.write(filename + "\n")

