import numpy as np

import __add_path__
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics


np.set_printoptions(precision=4, suppress=True)

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
rArmKinematics = ApolloArmKinematics(r_arm=True)
q_start = rArmKinematics.IK_best(T_home)


# Go big or go home
rArmInterface.go_to_home_position(q_start)