# %%
import numpy as np
import matplotlib.pyplot as plt

import __add_path__
from juggling_apollo.settings import dt
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, load

np.set_printoptions(precision=4, suppress=True)


# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True)

for j in range(1):
    with open('data/AllJoints3/list_files_all.txt') as topo_file:
        for filename in topo_file:
            filename = filename.strip()  # The comma to suppress the extra new line char
            
            
            ld = load(filename)
            
            title = filename.split('/')[-1].split("2021")[0][7:]
            print("-----------------")
            print(title)
            print("-----------------")
            N_ILC = ld.joints_q_vec.shape[0]
            N_1   = ld.joints_q_vec.shape[1]
            
            plot_A([ld.u_arr, ld.joints_vq_vec[-1, 1:]], ld.learnable_joints, 
                    fill_between=[np.max(ld.u_ff_vec, axis=0)[1:], np.min(ld.u_ff_vec, axis=0)[1:]],
                    labels=["desired", "real"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
            plt.suptitle("Angle Velocities")
            plt.show(block=True)
            
            
            # Run Simulation
            rArmInterface.apollo_run_one_iteration(dt=dt, T=N_1*dt-0.00001, u=ld.u_arr, joint_home_config=ld.q_start, repetitions=10, it=j)