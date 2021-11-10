# %%
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

import __add_path__
from juggling_apollo.settings import dt


np.set_printoptions(precision=4, suppress=True)


colors = ["r", 'g', 'b', 'k', 'c', 'm', 'y']
line_types = ["-", "--", ":", '-.']

def plot_A(lines_list, indexes_list, labels, dt=1, xlabel="", ylabel=""):
  assert len(lines_list) == len(labels), "Please use same number of lines and labels"
  N = len(lines_list)
  M = len(indexes_list)
  if M >= 3:
    a = M//3 + (1 if M%3 !=0 else 0)
    b = 3
  else:
    a = 1
    b = M
  timesteps = dt*np.arange(lines_list[0].shape[0])
  fig, axs = plt.subplots(a,b, figsize=(12,8))
  axs = np.array(axs)
  for iii, ix in enumerate(indexes_list):
    for i in range(N):
    #   axs.flatten()[iii].plot(timesteps, lines_list[i][:, ix].squeeze(), color=colors[i], linestyle=line_types[i], label=r"$\theta_{}$ {}".format(ix, labels[i]))
      axs.flatten()[iii].plot(lines_list[i][:, ix].squeeze(), label=r"$\theta_{}$ {}".format(ix, labels[i]))
    # axs.flatten()[iii].legend(loc=1)
    axs.flatten()[iii].grid(True)
  axs.flatten()[iii].legend(loc=1)
#   axs.flatten()[iii].legend(loc='center left', bbox_to_anchor=(1, 0.5))
  fig.text(0.5, 0.04, xlabel, ha='center')
  fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')




error_list = []
label_list = []


for j in range(1):
    # fig = plt.figure(figsize=(12,8))
        
    # with open('data/SingleJoints/list_files.txt') as topo_file:
    with open('data/AllJoints/list_files_all.txt') as topo_file:
        for filename in topo_file:
            filename = filename.strip()  # The comma to suppress the extra new line char
            if False and filename.split('/')[-1][6] != str(j):
                continue
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

            title = filename.split('/')[-1].split("2021")[0][7:]
            N_ILC = joints_q_vec.shape[0]
            
            
            if False and error_norms[:,j][-1] > 0.50:
                continue
            # print(title)
            # plt.plot(error_norms[:,j], label=r"$\alpha={}, \epsilon={}$".format(alpha, ep_s[j]))
            
            
            # Joint Velocity trajectories
            if False:
                plot_A([u_arr, joints_vq_vec[-1, 1:]], learnable_joints, ["des", "it="+str(N_ILC)], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
                plt.suptitle("Angle Velocities")
                plt.show(block=True)

            # Errors
            if alpha>12.0:
                error_list.append(error_norms)
                label_list.append(r"$\alpha={}, \epsilon={}$".format(alpha, ep_s[j]))
            else:
                continue
            if False:
                plot_A([error_norms], learnable_joints, ["des"], dt=1.0, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
                plt.suptitle("Error Norms__" + r"$\alpha={}, \epsilon={}$".format(alpha, ep_s[j]))
                plt.show(block=True)

            # Joing angle trajectories
            if False:
                plot_A([q_traj_des, joints_q_vec[-1], joints_q_vec[-2], joints_q_vec[0]], learnable_joints, ["des", "it="+str(N_ILC-1), "it="+str(N_ILC-2), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
                plt.suptitle("Angle Positions")
                plt.show(block=False)
            if False:
                # Learned disturbances
                plot_A([disturbanc_vec[-1, 1:], disturbanc_vec[-2, 1:], disturbanc_vec[-4, 1:], disturbanc_vec[1, 1:]], learnable_joints, ["d", "it-2", "it-4", "it=1"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
                plt.suptitle("Disturbance")
                plt.show(block=True)


            # Cartesian trajectories
            if False:
                ls = ['x', 'y', 'z']
                fig, axs = plt.subplots(3,1, figsize=(16,11))
                for ii in range(3):
                    # axs[ii].plot(xyz_vec[-1][:, ii], c=colors[ii], label=ls[ii])
                    axs[ii].plot(xyz_traj[:, ii] + T_home[ii, -1] - xyz_vec[-1][:, ii], c=colors[ii], linestyle='--', label=ls[ii]+'')
                    # lines = plt.plot(xyz_vec[j, ] - xyz_traj - T_home[:3, -1])
                    # plt.legend(iter(lines), (i for i in ['x', 'y', 'z']))
                    axs[ii].legend(loc=1)
                    axs[ii].grid(True)
                    axs[ii].set_ylabel(ls[ii]+" error [m]")
                axs[ii].set_xlabel("t [s]")
                
                plt.suptitle(r"Cartesian Space errors for: $\alpha={}, \epsilon={}$".format(alpha, ep_s[j]))
                plt.show()
            
        # plt.title(r"$\theta_{}$ - Error Norm Trajectories".format(j+1))
        # plt.legend()
        # plt.show()


plot_A(error_list, learnable_joints, label_list, dt=1.0, xlabel=r"$ITERATION$ [j]", ylabel=r"Angle Trajectory Error [$rad$]")
plt.suptitle("Error Norms")
plt.show(block=True)