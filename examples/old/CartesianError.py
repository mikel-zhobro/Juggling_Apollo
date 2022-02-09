import matplotlib.pyplot as plt
import numpy as np

import __add_path__
import configs
from ApolloILC.settings import dt
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics
from ApolloKinematics import utilities


np.set_printoptions(precision=4, suppress=True)

print("juggling_apollo")
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

# A. KINEMATICS: create rArmInterface and go to home position
# kinematics with noise (used for its (wrong)IK calculations)
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=0.1)
# kinematics without noise (used to calculate measurments, plays the role of a localization system)
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)


# B. COMPUTE MINJERK TRAJECTORY IN CARTESIAN SPACE
N, delta_xyz_traj_des, thetas, mj = configs.get_minjerk_config(dt, 0)
xyz_traj_des = delta_xyz_traj_des + T_home[:3, -1]
N_1 = N-1


# COMPUTE TRAJECTORIES IN JOINT SPACE
cartesian_traj_des, q_traj_des, _, _ = rArmKinematics.seqIK(delta_xyz_traj_des, thetas, T_home)  # [N, 7]
cartesian_traj_des, q_traj_des_nn, _, _ = rArmKinematics_nn.seqIK(delta_xyz_traj_des, thetas, T_home)  # [N, 7]

des_traj = rArmKinematics_nn.seqFK(q_traj_des_nn)

# Cartesian Error propogation params
damp            = 1e-12
mu              = 0.422

q_traj_des_i = q_traj_des.copy()
for j in range(60):
    traj_i =  rArmKinematics_nn.seqFK(q_traj_des_i)
    delta = np.array([utilities.errorForJacobianInverse(T_i=traj_i[i], T_goal=des_traj[i]) for i in range(N)])
    for i in range(N):
        J_invj          = np.linalg.pinv(rArmKinematics.J(q_traj_des_i[i]))
        q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(delta[i].reshape(-1,1))
        # q_traj_des_i[i] = q_traj_des_i[i] - mu* J_invj.dot(delta[i,:3].reshape(3, 1))  # only xyz

    if j%4==0:
        plt.plot(np.arange(N), delta[:, 0], label='x')
        plt.plot(np.arange(N), delta[:, 1], label='y')
        plt.plot(np.arange(N), delta[:, 2], label='z')
        plt.plot(np.arange(N), delta[:, 3], label='nx')
        plt.plot(np.arange(N), delta[:, 4], label='ny')
        plt.plot(np.arange(N), delta[:, 5], label='nz')
        plt.legend()
        plt.show()
    print('{:3}. {}, {}'.format(j, np.linalg.norm(delta[:,:3]), np.linalg.norm(delta[:,3:])))

