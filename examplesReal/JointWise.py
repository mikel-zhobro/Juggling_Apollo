import numpy as np
import time

import __add_path__
from juggling_apollo.utils import steps_from_time, plt
from juggling_apollo.JugglingPlanner import calc
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys2
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from kinematics.utilities import clip_c
from utils import plot_A, save, colors, line_types


np.set_printoptions(precision=4, suppress=True)

end_repeat = 14   # repeat the last position value this many time
SAVING = False

print("juggling_apollo")

# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True, noise=0.0)  ## noise noisifies the forward dynamics only
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)  ## noise noisifies the forward dynamics only
T_home = np.eye(4, dtype='float')
T_home[:3, -1] = [0.32, 0.81, -0.49]
T_home[:3, :3] = np.array([[0.0, -1.0, 0.0],  # uppword orientation(cup is up)
                           [0.0,  0.0, 1.0],
                           [-1.0, 0.0, 0.0]], dtype='float')

####################################################################################################################################
####################################################################################################################################
# A. COMPUTE TRAJECTORIES IN CARTESIAN AND JOINT SPACE
# 1. --------------- Compute juggling params for given E, tau and ---------------
# A) Orientation
#    World: x-left, y-forward, z-up
#    TCP:   x-down, y-right,   z-forward
# B) Position
#    only z changes, x and y stay contant for 1D case
Ex = 0.15                                                                 # Ex = z_catch - z_throw
tau = 0.5                                                                 # tau = T_hand + T_empty (length of the repeatable part)
dwell_ration = 0.6                                                        # what part of tau is used for T_hand
T_hand, T_empty, ub_throw, H, z_catch = calc(tau, dwell_ration, Ex, slower=3.21)  # 2.0 gives error
T_throw_first = T_hand*0.5                                                # Time to use for the first throw from home position

T_fly = T_hand + 2*T_empty
T_FULL = T_empty + T_hand
N_1 = steps_from_time(T_FULL, dt)-1                                       # size of our vectors(i.e. length of the learning interval)
N_throw = steps_from_time(T_throw_first, dt)-1                            # timestep where throw must happen
N_throw_empty = steps_from_time(T_throw_first+T_empty, dt)-1              # timestep where catch must happen
N_repeat_point = steps_from_time(T_throw_first, dt)-1                     # timestep from where the motion should be repeated
print('H: ' + str(H))
print('T_fly: ' + str(T_fly))
# ---------------           ---------------
# --------------- Min Jerk ---------------
smooth_acc = False
ub_catch = -ub_throw*0.9
i_a_end = 0
tt=[0.0,      T_throw_first,     T_throw_first+T_empty,   T_FULL  ]
xx=[0.0,      0.0,               z_catch,                 0.0     ]
uu=[0.0,      ub_throw/12.0,      ub_catch/12.0,            0.0     ]

print(tt)
z_des, velo, accel, jerk = get_minjerk_trajectory(dt, smooth_acc=smooth_acc, i_a_end=i_a_end, tt=tt, xx=xx, uu=uu, extra_at_end=end_repeat+1)  # Min jerk trajectories (out of the loop since trajectory doesn't change)

tt=[0.0,      T_throw_first,     T_throw_first+T_empty,   T_FULL  ]
xx=[0.0,      0.0,               z_catch/4.0,               0.0     ]
uu=[0.0,      ub_throw/12.0,      ub_catch/12.0,            0.0     ]

y_des, velo2, accel2, jerk2 = get_minjerk_trajectory(dt, smooth_acc=smooth_acc, i_a_end=i_a_end, tt=tt, xx=xx, uu=uu, extra_at_end=end_repeat+1)  # Min jerk trajectories (out of the loop since trajectory doesn't change)


if False:
  print(z_catch)
  plotMJ(dt, tt, xx, uu, smooth_acc, (z_des, velo, accel, jerk))
  plotMJ(dt, tt, xx, uu, smooth_acc, (y_des, velo2, accel2, jerk2))
  plt.show()
#---------------        ---------------
####################################################################################################################################
####################################################################################################################################

# Cartesian -> JointSpace                   <------------------------------------------------------------------------------------------ Min Jerk Trajectory (CARTESIAN AND JOINT SPACE)
thetas                   = np.zeros_like(z_des)
xyz_traj_des             = np.zeros((thetas.size, 3))
xyz_traj_des[:,2]        = z_des
xyz_traj_des[:,1]        = y_des
q_traj_des, q_start, psi_params   = rArmKinematics.seqIK(xyz_traj_des, thetas, T_home)  # [N_1, 7]
q_traj_des_nn, q_start_nn, _   = rArmKinematics_nn.seqIK(xyz_traj_des, thetas, T_home)  # [N_1, 7]
assert np.allclose(q_start , q_traj_des[0])
assert np.allclose(q_start_nn , q_traj_des_nn[0])

if False:
  rArmKinematics.plot(q_traj_des, *psi_params)
  
if False:
  from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot3D(xyz_traj_des[:,0], xyz_traj_des[:,1], xyz_traj_des[:,2], 'gray')
  plt.show()

# B. Initialize ILC
def kf_params(n_m=0.02, epsilon=1e-5, n_d=0.06):
  kf_dpn_params = {
    'M': n_m*np.eye(N_1+end_repeat, dtype='float'),       # covariance of noise on the measurment
    'P0': n_d*np.eye(N_1+end_repeat, dtype='float'),      # initial disturbance covariance
    'd0': np.zeros((N_1+end_repeat, 1), dtype='float'),   # initial disturbance value
    'epsilon0': epsilon,                                  # initial variance of noise on the disturbance
    'epsilon_decrease_rate': 1.0                          # the decreasing factor of noise on the disturbance
  }
  return kf_dpn_params

n_ms = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
n_ds = [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
ep_s = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
# ep_s = [5e-3] * 7     # works well for velocity disturbance
# ep_s = [1e-2] * 7     # works well for velocity disturbance
alpha = 16.0
my_ilcs = [
  # ILC(dt=dt, sys=ApolloDynSys(dt, alpha_=alpha), kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i]), x_0=[q_start[i, 0], 0.0])    # include the initial state in the dynamics of the system
  ILC(dt=dt, sys=ApolloDynSys(dt, alpha_=alpha), kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i]), x_0=[0.0, 0.0])                # make sure to make up for the initial state during learning
  for i in range(N_joints)]

for ilc in my_ilcs:
  ilc.initILC(N_1=N_1+end_repeat, impact_timesteps=[False]*(N_1+end_repeat))  # ignore the ball

# C. LEARN BY ITERATING
# Learn Throw
ILC_it = 16  # number of ILC iteration

# Data collection
q_traj_des_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
# a. System Trajectories
joints_q_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
joints_vq_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
joints_aq_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
xyz_vec        = np.zeros([ILC_it, N_1+1+end_repeat, 3], dtype='float')
# b. ILC Trajectories
disturbanc_vec = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
joints_d_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
u_ff_vec       = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
u_eff_vec      = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
# c. Measurments
joint_torque_vec     = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
# d. Trajectory error norms
error_norms    = np.zeros([ILC_it, N_joints, 1], dtype='float')

# ILC Vectors Init
u_ff = [None] * N_joints
y_meas = np.zeros((N_1+end_repeat, N_joints), dtype='float')



####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

# Extra Loop (In case we want to try out smth on different combination of joints)
every_N = 113
for jjoint in range(1):
  ## CHOOOSE JOINTS THAT LEARN
  jjoint = "all"
  # learnable_joints = [0,1,2,3,5]
  learnable_joints = [0,1,2,3,5]
  non_learnable_joints = set(range(7)) - set(learnable_joints)
  for i in non_learnable_joints:
    q_traj_des[:,i] = 0.0
  for ilc in my_ilcs:
    ilc.resetILC()
  y_meas = np.zeros((N_1+end_repeat, N_joints), dtype='float')
  u_ff = [None] * 7


  # Main Loop
  ## Cartesian error params
  # rArmKinematics_nn:  kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
  # rArmKinematics:     kinematics with noise (used for its (wrong)IK calculations)
  damp            = 1e-12
  mu              = 1e-2
  CARTESIAN_ERROR = False
  q_traj_des_i    = q_traj_des.copy()   # Changing (possibly wrong/noisy) desired trajectory to make up for kinematics errors

  for j in range(ILC_it):  
    # Learn feed-forward signal
    # u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=q_traj_des_i[:, i], y_meas=y_meas[:, i],             # initial state considered in the dynamics
    u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=q_traj_des_i[:, i] - q_start[i], y_meas=y_meas[:, i] - q_start[i],             # substract the initial state from the desired joint traj
                           verbose=False,  # bool(i in learnable_joints and j%every_N==0 and False),
                           lb=-1.0,ub=1.0) for i, ilc in enumerate(my_ilcs)]
    u_arr = np.array(u_ff, dtype='float').squeeze().T
    for i in non_learnable_joints:
      u_arr[:,i] = 0.0

  
  
    # CLIP
    # u_arr = np.clip(u_arr,-1.0,1.0)
    # Main Simulation
    q_traj, q_v_traj, q_a_traj, dP_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL+end_repeat*dt, u=u_arr, joint_home_config=q_start, repetitions=1, it=j)

    # For the next iteration
    if CARTESIAN_ERROR:
      xyz_traj_meas = rArmKinematics_nn.seqFK(q_traj)[:, :3, -1]                    # actual cartesian errors
      d_xyz = xyz_traj_des + T_home[:3, -1] - xyz_traj_meas                         # [N_1, 3] d_xyz = xyz_des - xyz_i
      for i in range(N_1):
        J_invj = np.linalg.pinv(rArmKinematics.J(q_traj[i+1])[:3,:])                # q_traj[1:]  # N_1x7
        q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(d_xyz[i].reshape(3, 1))
      q_start = q_traj_des_i[0]


    # System Output
    y_meas = q_traj[1:]
    d_xyz =      xyz_traj_des + T_home[:3, -1] - rArmKinematics_nn.seqFK(q_traj)[:, :3, -1]         # measured cartesian error: calculated using the noise-less FK
    d_xyz_best = xyz_traj_des + T_home[:3, -1] - rArmKinematics_nn.seqFK(q_traj_des)[:, :3, -1]
    d_xyz_achievable = xyz_traj_des + T_home[:3, -1] - rArmKinematics.seqFK(q_traj_des)[:, :3, -1]

    # Collect Data
    q_traj_des_vec[j]   = q_traj_des_i
    joints_q_vec[j]     = q_traj
    joints_vq_vec[j]    = q_v_traj
    joints_aq_vec[j]    = q_a_traj
    u_ff_vec[j, :-1]      = u_arr
    u_eff_vec[j, :-1]     = u_vec
    joint_torque_vec[j] = dP_N_vec
    disturbanc_vec[j, 1:] = np.squeeze([ilc.d for ilc in my_ilcs]).T  # learned joint space disturbances
    joints_d_vec[j, 1:]   = np.squeeze(q_traj_des[1:]-y_meas)         # actual joint space error
    xyz_vec[j]          = rArmKinematics.seqFK(q_traj)[:, :3, -1]   # actual cartesian errors
    error_norms[j]     = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T


    if True and j%every_N==0:  # Velocities
      plot_A([u_arr, q_v_traj[1:]], learnable_joints, 
             fill_between=[np.max(u_eff_vec, axis=0)[1:], np.min(u_eff_vec, axis=0)[1:]],
             labels=["desired", "real"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
      plt.suptitle("Angle Velocities")
      plt.show(block=False)

    if True and j%every_N==0:  # Positions
      plot_A([q_traj_des, q_traj, joints_q_vec[j-1], joints_q_vec[0]], learnable_joints, ["des", "it="+str(j), "it="+str(j-4), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
      plt.suptitle("Angle Positions")
      plt.show()

    if False and j%every_N==0:  # Disturbances
      plot_A([disturbanc_vec[j, 1:], disturbanc_vec[j-2, 1:], disturbanc_vec[j-4, 1:], disturbanc_vec[1, 1:]], learnable_joints, ["d", "it-2", "it-4", "it=1"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
      plt.suptitle("Disturbance")
      plt.show()

    if False and j%every_N==0:  # Errors
      ls = ['x', 'y', 'z']
      fig, axs = plt.subplots(3,1, figsize=(12,8))
      for ii in range(3):
        # axs[ii].plot(xyz_vec[j, ][:, ii], c=colors[ii], label=ls[ii])
        axs[ii].plot(np.abs(d_xyz[:, ii]), c=colors[ii], label="d_"+ls[ii], linestyle=line_types[0])
        axs[ii].plot(np.abs(d_xyz_best[:, ii]), c=colors[ii], label="d_"+ls[ii]+"_best", linestyle=line_types[1])
        axs[ii].plot(np.abs(d_xyz_achievable[:, ii]), c=colors[ii], label="d_"+ls[ii]+"_achievable", linestyle=line_types[2])
        axs[ii].legend(loc=1)
      plt.show()

    if False and j%every_N==0:  # How desired  trajectory changes
      plot_A([q_traj_des, q_traj_des_vec[j], q_traj_des_vec[j-1], q_traj_des_vec[0]], learnable_joints, ["des", "it="+str(j), "it="+str(j-1), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
      plt.suptitle("Desired Joint Trajectories")
      plt.show(block=False)

    print("ITERATION: " + str(j+1))
    print(          "j. -----------degree----------      L2-norm        L1-norm        e_end      <- unnormalized")
    for i in learnable_joints:
      print(str(i) + ". Trajectory_track_error_norm: {:13.8f}  {:13.8f} {:13.8f}".format(np.linalg.norm(180.0/np.pi*joints_d_vec[j, :, i]), 
                                                                                         np.linalg.norm(180.0/np.pi*joints_d_vec[j, :, i], ord=1),
                                                                                         np.abs(180.0/np.pi*joints_d_vec[j, -1, i])
                                                                                         ))
    ls = ['x', 'y', 'z']
    print(          "j. -----------meters----------      L2-norm        L1-norm        e_end      <- unnormalized")
    for i in range(3):
      print(ls[i] + ". Trajectory_track_error_norm: {:13.8f}  {:13.8f} {:13.8f}".format(np.linalg.norm(d_xyz[:, i]), 
                                                                                        np.linalg.norm(d_xyz[:, i], ord=1),
                                                                                        np.abs(d_xyz[-1, i])
                                                                                        ))
      
  if False:
    ls = ['x', 'y', 'z']
    fig, axs = plt.subplots(3,1, figsize=(12,8))
    for ii in range(3):
      # axs[ii].plot(xyz_vec[j, ][:, ii], c=colors[ii], label=ls[ii])
      axs[ii].plot(np.abs(d_xyz[:, ii]), c=colors[ii], label="d_"+ls[ii], linestyle=line_types[0])
      axs[ii].plot(np.abs(d_xyz_best[:, ii]), c=colors[ii], label="d_"+ls[ii]+"_best", linestyle=line_types[1])
      axs[ii].plot(np.abs(d_xyz_achievable[:, ii]), c=colors[ii], label="d_"+ls[ii]+"_achievable", linestyle=line_types[2])
      # axs[ii].plot(xyz_traj_des[:, ii] + T_home[ii, -1], c=colors[ii], linestyle='--', label=ls[ii]+'_des')
      # lines = plt.plot(xyz_vec[j, ] - xyz_traj_des - T_home[:3, -1])
      # plt.legend(iter(lines), (i for i in ['x', 'y', 'z']))
      axs[ii].legend(loc=1)
    plt.show()
    
    
  plot_A([q_traj_des, q_traj, joints_q_vec[j-1], joints_q_vec[0]], learnable_joints, ["des", "it="+str(j), "it="+str(j-4), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
  plt.suptitle("Angle Positions")
  plt.show(block=False)
  
  plot_A([u_arr, q_v_traj[1:]], learnable_joints, fill_between=[np.max(u_eff_vec, axis=0)[1:], np.min(u_eff_vec, axis=0)[1:]],
          labels=["desired", "real"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
  plt.suptitle("Angle Velocities")
  plt.show(block=True)

  if SAVING:
    # Saving Results
    filename = "data/AllJoints3/joint_{}_alpha_{}_eps_{}_".format(jjoint, alpha, ep_s[0]) + time.strftime("%Y_%m_%d-%H_%M_%S")
    save(filename,
         q_start=q_start, T_home=T_home,                                                   # Home
         xyz_traj_des=xyz_traj_des, q_traj_des=q_traj_des,                                         # Desired Trajectories
         joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
         joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
         disturbanc_vec=disturbanc_vec, u_ff_vec=u_ff_vec,                                 # Learned Trajectories (uff and disturbance)
         xyz_vec=xyz_vec, joints_d_vec=joints_d_vec, error_norms=error_norms,              # Progress Measurments
         tt=tt, xx=xx, uu=uu,                                                              # Minjerk Params
         learnable_joints=learnable_joints, alpha=alpha, n_ms=n_ms, n_ds=n_ds, ep_s=ep_s)  # ILC parameters

    with open('data/AllJoints3/list_files_all.txt', 'a') as f:
        f.write(filename + "\n")



# Run Simulation with several repetition
# rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_arr[:-end_repeat], joint_home_config=q_start, repetitions=25, it=j)
rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL+end_repeat*dt, u=u_arr, joint_home_config=q_start, repetitions=25, it=j)
