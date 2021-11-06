# %%
import numpy as np

import __add_path__
import matplotlib.pyplot as plt
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
      axs.flatten()[iii].plot(timesteps, lines_list[i][:, ix].squeeze(), color=colors[ix], linestyle=line_types[i], label=r"$\theta_{}$ {}".format(ix, labels[i]))
      r"$\theta_{}$ {}".format(ix, labels[i])
      axs.flatten()[iii].legend(loc=1)
      axs.flatten()[iii].grid(True)
  fig.text(0.5, 0.04, xlabel, ha='center')
  fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

# Data loading
filename = "../data/"
with open(filename, 'rb') as f:
  # The desired trajectory to be learned
  q_traj_des = np.load(f)
  # a. System Trajectories  
  joints_q_vec    = np.load(f) 
  joints_vq_vec   = np.load(f)
  joints_aq_vec   = np.load(f)
  xyz_vec         = np.load(f)
  # b. ILC Trajectories
  disturbanc_vec  = np.save(f)
  joints_d_vec    = np.save(f)  
  u_ff_vec        = np.save(f)      
  # c. Measurments
  torque_vec      = np.save(f)
  # d. Trajectory error norms
  error_norms     = np.save(f)
  
  
  
  
learnable_joints = [0, 1, 2, 3, 4, 5, 6]
  
# Joint Velocity trajectories
plot_A([u_arr, q_v_traj[1:]], learnable_joints, ["des", "it="+str(j)], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
plt.suptitle("Angle Velocities")
plt.show(block=False)

# Joing angle trajectories
plot_A([q_traj_des, q_traj, joints_q_vec[-1], joints_q_vec[0]], learnable_joints, ["des", "it="+str(j), "it="+str(j-4), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
plt.suptitle("Angle Positions")
plt.show()

# Learned disturbances
plot_A([disturbanc_vec[j, 1:], disturbanc_vec[j-2, 1:], disturbanc_vec[j-4, 1:], disturbanc_vec[1, 1:]], learnable_joints, ["d", "it-2", "it-4", "it=1"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
plt.suptitle("Disturbance")
plt.show()


# Cartesian trajectories
ls = ['x', 'y', 'z']
fig, axs = plt.subplots(3,1, figsize=(12,8))
for ii in range(3):
    axs[ii].plot(xyz_vec[j, ][:, ii], c=colors[ii], label=ls[ii])
    axs[ii].plot(xyz_traj[:, ii] + T_home[ii, -1], c=colors[ii], linestyle='--', label=ls[ii]+'_des')  
    # lines = plt.plot(xyz_vec[j, ] - xyz_traj - T_home[:3, -1])
    # plt.legend(iter(lines), (i for i in ['x', 'y', 'z']))
    axs[ii].legend(loc=1)
plt.show()