import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

import __add_path__
from juggling_apollo.settings import dt
from utils import plot_A
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics

np.set_printoptions(precision=4, suppress=True)


error_list = []
label_list = []


# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True)
T_home = np.eye(4, dtype='float')
T_home[:3, -1] = [0.32, 0.81, -0.49]
T_home[:3, :3] = np.array([[0.0, -1.0, 0.0],  # uppword orientation(cup is up)
                           [0.0,  0.0, 1.0],
                           [-1.0, 0.0, 0.0]], dtype='float')

for j in range(1):
    with open('examples/data/AllJoints/list_files_all.txt') as topo_file:
        for filename in topo_file:
            filename = filename.strip()  # The comma to suppress the extra new line char

            # Data loading
            with open(filename, 'rb') as f:
                # Home
                q_start          = np.load(f)
                T_home           = np.load(f)
                # Desired Trajectories
                xyz_traj         = np.load(f)
                q_traj_des       = np.load(f)
                u_arr            = np.load(f)
                # a. System Trajectories
                joints_q_vec     = np.load(f)
                joints_vq_vec    = np.load(f)
                joints_aq_vec    = np.load(f)
                xyz_vec          = np.load(f)
                # b. ILC Trajectories
                disturbanc_vec   = np.load(f)
                joints_d_vec     = np.load(f)
                u_ff_vec         = np.load(f)
                # c. Measurments
                torque_vec       = np.load(f)
                # d. Trajectory error norms
                error_norms      = np.load(f)
                # Params
                ## a. Minjerk
                tt             = np.load(f)
                xx             = np.load(f)
                uu             = np.load(f)

                ## b. ILC
                learnable_joints = np.load(f)
                alpha            = np.load(f)
                n_ms             = np.load(f)
                n_ds             = np.load(f)
                ep_s             = np.load(f)

            
            if alpha != 16.0:
                continue
            
            title = filename.split('/')[-1].split("2021")[0][7:]
            print("-----------------")
            print(title)
            print("-----------------")
            N_ILC = joints_q_vec.shape[0]
            N_1 = joints_q_vec.shape[1]
            
            plot_A([u_arr, joints_vq_vec[-1, 1:]], learnable_joints, fill_between=[np.max(u_ff_vec, axis=0)[1:], np.min(u_ff_vec, axis=0)[1:]],
                labels=["desired", "real"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
            plt.suptitle("Angle Velocities")
            plt.show(block=True)
            rArmInterface.apollo_run_one_iteration(dt=dt, T=N_1*dt-0.00001, u=u_arr, joint_home_config=q_start, repetitions=10, it=j)