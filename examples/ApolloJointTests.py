# %%
import numpy as np

import __add_path__
import matplotlib.pyplot as plt
from juggling_apollo.settings import dt
from apollo_interface.Apollo_It import ApolloInterface
from kinematics.ApolloKinematics import ApolloArmKinematics

np.set_printoptions(precision=4, suppress=True)


colors = ["r", 'g', 'b', 'k', 'c', 'm', 'y']
line_types = ["-", "--", ":", '-.']

def plot_joints(joints_traj, dt=1.0, title="", block=True, limits=None):
  timesteps = dt*np.arange(joints_traj.reshape(-1, 7).shape[0])
  if block:
    fig, axs = plt.subplots(7, 1)
    for i in range(7):
      lines = axs[i].plot(timesteps, joints_traj[:, i], c=colors[i], label=r'$\theta_{}$'.format(i))
      if limits is not None:
        axs[i].axhspan(limits[i].a, limits[i].b, color=colors[i], alpha=0.3, label='feasible set')
        axs[i].set_ylim([min(-np.pi, limits[i].a), max(np.pi, limits[i].b)])
      axs[i].legend(loc=1)
      
    plt.suptitle(title)
  else:
    for i in range(7):
      lines = plt.plot(timesteps, joints_traj[:, i], c=colors[i], label=r'$\theta_{}$'.format(i))
      plt.legend()
    plt.title(title)
  plt.show(block=block)


def plot_A(lines_list, indexes_list, labels, dt=1, xlabel="", ylabel="", limits=None):
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
      axs.flatten()[iii].plot(timesteps, lines_list[i][:, ix].squeeze(), color=colors[ix], linestyle=line_types[i], label=r"$\theta_{}$ {}".format(ix+1, labels[i]))
      if limits is not None:
        axs.flatten()[iii].axhspan(limits[ix].a, limits[ix].b, color=colors[ix], alpha=0.3, label='feasible set')
        axs.flatten()[iii].set_ylim([min(-np.pi, limits[ix].a), max(np.pi, limits[ix].b)])
      axs.flatten()[iii].legend(loc=1)
      axs.flatten()[iii].grid(True)
  fig.text(0.5, 0.04, xlabel, ha='center')
  fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')


print("testing_apollo")

# 0. Create Apollo objects
# Init state  ([{ 1-Dim }])
home_pose = np.array([ 0.6484, -0.6194, -1.8816, 1.0706, -2.4248, 1.1782, -2.4401])
N_joints = len(home_pose)

# A) INTERFACE: create rArmInterface and go to home position
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True)
T_home = rArmKinematics.FK(home_pose)
T_home[:3, :3] = np.array([[0.0, -1.0, 0.0],  # uppword orientation(cup is up)
                           [0.0,  0.0, 1.0],
                           [-1.0, 0.0, 0.0]], dtype='float')
q_home = rArmKinematics.IK_best(T_home)
print(T_home)
print(q_home.T)
# %%
# Learn Throw
N_joints     = 7  # number of joints
N_1          = 500
N_step_start = 200
N_step_end   = 380

T_FULL = N_1*dt
T_signal = (N_step_end-N_step_start)*dt


# Input signal (desired velocity)
timesteps = np.arange(N_step_end-N_step_start).reshape(-1,1)*dt

# Sinus Like Signal
amplitudes = [0.5, 1, 2, 4]
u_des = np.sin(np.pi/T_signal * timesteps)

# Step Like Signal
amplitudes = [0.5, 1, 2, 4]
# u_des = 1.0

v_traj_des = np.zeros((N_1, N_joints, 1))

if False:
  plot_joints(v_traj_des)

# Data collection
# a. System Trajectories
N_trials = len(amplitudes)
joints_q_vec  = np.zeros([N_trials, N_1+1, N_joints, 1], dtype='float')
joints_vq_vec = np.zeros([N_trials, N_1+1, N_joints, 1], dtype='float')
joints_aq_vec = np.zeros([N_trials, N_1+1, N_joints, 1], dtype='float')




def simulate_vel(v_des, alpha=200.0):
    N = v_des.size
    vels = np.zeros((N,7,1))
    for i in range(N):
        # vels[i] =  alpha*dt*vels[i-1] + (1.0-alpha*dt)*v_des[i] 
        vels[i] = vels[i-1] + alpha*dt*(v_des[i]-vels[i-1])
    return vels

if False:
    # All joints together
    for j, ampl in enumerate(amplitudes):
        v_traj_des[N_step_start:N_step_end] = ampl*u_des
        q_traj, q_v_traj, q_a_traj, dP_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=v_traj_des, joint_home_config=q_home, repetitions=1, it=j)
        if True:
            v_traj_des[N_step_start:N_step_end,:] = v_traj_des[N_step_start:N_step_end]
            plot_A([v_traj_des, q_v_traj[1:]], learnable_joints, ["des", "it="+str(j)], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
            plt.suptitle("Angle Velocities")
            plt.show(block=False)
        if True:
            v_traj_des[N_step_start:N_step_end,:] = ampl*u_des
            plot_A([q_traj], learnable_joints, ["measured"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle position [$rad$]", limits=rArmKinematics.limits)
            plt.suptitle("Angle Positions")
            plt.show(block=True)


## CHOOOSE JOINTS TO Check
learnable_joints = [1]

# All joints one-by-one
every_N = 1
for j, ampl in enumerate(amplitudes):
    print("IT: " + str(j))
    for i in learnable_joints:
        print("\tJOINT: " + str(i+1))
        v_traj_des[:] = 0.0
        v_traj_des[N_step_start:N_step_end,i] = ampl*u_des

        # plot_A([simulate_vel(v_traj_des[:,i])], learnable_joints, ["sim"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
        # plt.show()
        # Main Simulation
        q_traj, q_v_traj, q_a_traj, dP_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=v_traj_des, joint_home_config=q_home, repetitions=1, it=i)

        # Collect Data
        joints_q_vec[j,:,i]     = q_traj[:,i]
        joints_vq_vec[j,:,i]    = q_v_traj[:,i]
        joints_aq_vec[j,:,i]    = q_a_traj[:,i]

    if True and i%every_N==0:
        plot_A([v_traj_des, joints_vq_vec[j,1:], simulate_vel(v_traj_des[:,i])], learnable_joints, ["des", "it="+str(j), "sim"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
        plt.suptitle("Angle Velocities")
        plt.show(block=True)
