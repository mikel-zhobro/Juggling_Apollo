import numpy as np

import __add_path__
import configs
from juggling_apollo.settings import dt
from kinematics.ApolloKinematics import ApolloArmKinematics

np.set_printoptions(precision=4, suppress=True)

print("juggling_apollo")
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

# A. KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=0.1)  ## noise noisifies the forward dynamics only
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)  ## noise noisifies the forward dynamics only


# B. COMPUTE TRAJECTORIES IN CARTESIAN AND JOINT SPACE
N, delta_xyz_traj_des, thetas, mj = configs.get_minjerk_config(dt, 0)
xyz_traj_des = delta_xyz_traj_des + T_home[:3, -1]
N_1 = N-1

q_traj_des, _, _ = rArmKinematics.seqIK(delta_xyz_traj_des, thetas, T_home)  # [N, 7]
q_traj_des_i = q_traj_des.copy()

# Cartesian Error propogation params
# rArmKinematics_nn:  kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
# rArmKinematics:     kinematics with noise (used for its (wrong)IK calculations)
damp            = 1e-12
mu              = 1.
CARTESIAN_ERROR = False
for j in range(20):
    d_xyz_traj = xyz_traj_des - rArmKinematics_nn.seqFK(q_traj_des_i)[:, :3, -1]
    for i in range(N):
        J_invj          = np.linalg.pinv(rArmKinematics.J(q_traj_des_i[i])[:3,:])
        q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(d_xyz_traj[i].reshape(3, 1))

    print('{:3}. {}'.format(j, np.linalg.norm(d_xyz_traj)))