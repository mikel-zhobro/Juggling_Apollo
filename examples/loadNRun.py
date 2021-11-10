# %%
import numpy as np
import matplotlib.pyplot as plt

import __add_path__
from juggling_apollo.settings import dt
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal
from juggling_apollo.MinJerk import plotMJ
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, load, colors

np.set_printoptions(precision=4, suppress=True)


# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True)


filename = ""
ld = load(filename)


N_ILC = ld.joints_q_vec.shape[0]
N_1   = ld.joints_q_vec.shape[1]

# Cartesian trajectories
if True:
    ls = ['x', 'y', 'z']
    fig, axs = plt.subplots(3,1, figsize=(16,11))
    for ii in range(3):
        axs[ii].plot(ld.xyz_traj_des[:, ii] + ld.T_home[ii, -1] - ld.xyz_vec[-1][:, ii], c=colors[ii], linestyle='--', label=ls[ii]+'')
        # axs[ii].plot(ld.xyz_traj_des[:, ii], c=colors[ii], label=ls[ii]+'')
        # axs[ii].plot(-ld.T_home[ii, -1] + ld.xyz_vec[-1][:, ii], c=colors[ii], label=ls[ii]+'', linestyle='--')

        axs[ii].legend(loc=1)
        axs[ii].grid(True)
        axs[ii].set_ylabel(ls[ii]+" error [m]")
    axs[ii].set_xlabel("t [s]")
    
    plt.suptitle(r"Cartesian Space errors")
    plt.show()
# Joing angle trajectories
if True:
    plot_A([ld.q_traj_des, ld.joints_q_vec[-1], ld.joints_q_vec[-2], ld.joints_q_vec[0]], ld.learnable_joints, ["des", "it="+str(N_ILC-1), "it="+str(N_ILC-2), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
    plt.suptitle("Angle Positions")
    plt.show(block=False)
if True:
    plot_A(lines_list=[ld.u_ff_vec[-1, 1:], ld.joints_vq_vec[-1, 1:]], indexes_list=ld.learnable_joints, 
            fill_between=[np.max(ld.u_ff_vec, axis=0)[1:], np.min(ld.u_ff_vec, axis=0)[1:]],
            labels=["desired", "real"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
    plt.suptitle("Angle Velocities")
    plt.show(block=True)


# Run Simulation
rArmInterface.apollo_run_one_iteration(dt=dt, T=N_1*dt-0.00001, u=0.0*ld.u_arr, joint_home_config=ld.q_start, repetitions=10)