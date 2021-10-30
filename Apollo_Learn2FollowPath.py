# %%
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.JugglingPlanner import calc, traj_nb_2_na_1

from juggling_apollo.settings import dt, ABS, g, ABS
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import ApolloDynSys
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics


def plot_joints(joints_traj):
  plt.figure()
  lines = plt.plot(joints_traj.reshape(-1, 7))
  plt.legend(iter(lines), ("th: {}".format(i) for i in range(len(lines))))
  plt.show()


print("juggling_apollo")

# 0. Create Apollo objects
# Init state  ([{ 1-Dim }])
y_home = 0.0 # starting position for the hand
home_pose = np.array([np.pi/4, 0.0, 0.0, np.pi/4, -2*np.pi/3, np.pi/2, -np.pi/2])
# home_pose = np.array([0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

# A) INTERFACE: create rArmInterface and go to home position
rArmInterface = ApolloInterface(r_arm=True)
rArmInterface.go_to_home_position(home_pose)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True)
T_home = rArmKinematics.FK(home_pose)
T_home[:3, :3] = np.array([[0.0, -1.0, 0.0],  # uppword orientation(cup is up)
                           [0.0,  0.0, 1.0],
                           [-1.0, 0.0, 0.0]], dtype='float')
home_R = T_home[:3, :3]
home_position = T_home[:3, 3:]
N_joints = len(home_pose)


# 1. Compute juggling params for given E, tau and
# A) Orientation
# x shows up in our base coordinate frame(constant for 1D case(= y_tcp))
# z shows on the side (should stay constant(= -x_tcp) for the 1D case)
# y shows horizontally back to the robot(should stay constant for 1D case ( = -z_tcp))
# B) Position
# only x changes, y and z stay contant for 1D case

Ex = 0.15                                                                 # Ex = x_catch - x_throw
tau = 0.5                                                                 # tau = T_hand + T_empty (length of the repeatable part)
dwell_ration = 0.6                                                        # what part of tau is used for T_hand
T_hand, ub_throw, T_empty, H,  z_catch = calc(tau, dwell_ration, Ex)
T_throw_first = T_hand*0.5                                                # Time to use for the first throw from home position

T_fly = T_hand + 2*T_empty
T_FULL = T_throw_first + T_empty + T_hand - T_throw_first
N_1 = steps_from_time(T_FULL, dt)-1                                       # size of our vectors(i.e. length of the learning interval)
N_throw = steps_from_time(T_throw_first, dt)-1                            # timestep where throw must happen
N_throw_empty = steps_from_time(T_throw_first+T_empty, dt)-1              # timestep where catch must happen
N_repeat_point = steps_from_time(T_throw_first, dt)-1                     # timestep from where the motion should be repeated

print('H: ' + str(H))
print('T_fly: ' + str(T_fly))


# %%
# Learn Throw
ILC_it = 55  # number of ILC iteration

# Data collection
# System Trajectories
joints_q_vec  = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')
joints_vq_vec = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')
joints_aq_vec = np.zeros([ILC_it, N_1+1, N_joints, 1], dtype='float')
xyz_vec       = np.zeros([ILC_it, N_1+1, 3], dtype='float')
# ILC Trajectories
joints_d_vec  = np.zeros([ILC_it, N_1+1, N_joints], dtype='float')
u_ff_vec      = np.zeros([ILC_it, N_1+1, N_joints], dtype='float')
# Measurments
torque_vec    = np.zeros([ILC_it, N_1+1, N_joints], dtype='float')

# ILC Loop
u_ff = [None] * N_joints
y_meas = [None] * N_joints
y_meas = np.zeros((N_1, N_joints), dtype='float')

# Min Jerk Params
# new MinJerk
smooth_acc = False
ub_catch = -ub_throw*0.9
i_a_end = None
# a0 = None;        a1 = None;       a2 = None;          a3 = None
tt=[0,        T_throw_first,     T_throw_first+T_empty,   T_FULL  ]
xx=[y_home,   0.0,               z_catch,                 y_home   ]
uu=[0.0,      ub_throw,          ub_catch,                0.0     ]
if False:
  print(uu[-1])
  xvaj = get_minjerk_trajectory(dt, tt=tt, xx=xx, uu=uu, smooth_acc=smooth_acc)
  thetas = np.zeros_like(xvaj[0])
  position_traj = np.zeros((thetas.size, 3))
  position_traj[:, 0] = xvaj[0]
  joints_trajs, q_start, psi_params = rArmKinematics.seqIK(position_traj, thetas, T_home)  # [N, 7]
  rArmKinematics.plot(joints_trajs, *psi_params)
  
  plotMJ(dt, tt, xx, uu, smooth_acc, xvaj)


extra_rep = 2


# Min jerk trajectories (out of the loop since trajectory doesn't change)
y_des, velo, accel, jerk = get_minjerk_trajectory(dt, smooth_acc=smooth_acc, i_a_end=i_a_end, tt=tt, xx=xx, uu=uu)
# Transform minjerk to jointspace (out of the loop since trajectory doesn't change)
thetas = np.zeros_like(y_des)
position_traj = np.zeros((thetas.size, 3))
position_traj[:,2] = y_des
joints_traj_des, q_start, _ = rArmKinematics.seqIK(position_traj, thetas, T_home)  # [N, 7]

# q_s, q_v_s, q_a_s, dP_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=joints_traj_des.squeeze(), joint_home_config=q_start, repetitions=10, it=0, go2position=True)

# Init ilc
# Here we want to set some convention to avoid missunderstandins later on.
# 1. the state is [xb, xp, ub, up]^T
# 2. the system can have as input either velocity u_des or the force F_p
# I. SYSTEM DYNAMICS
input_is_velocity = True
kf_dpn_params = {
  'M': 0.031*np.eye(N_1, dtype='float'),    # covariance of noise on the measurment
  'P0': 0.1*np.eye(N_1, dtype='float'),     # initial disturbance covariance
  'd0': np.zeros((N_1, 1), dtype='float'),  # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1                # the decreasing factor of noise on the disturbance
}

# my_ilcs = [ILC(dt=dt, sys=ApolloDynSys(dt, input_is_velocity), kf_dpn_params=kf_dpn_params, x_0=[q_start[i, 0], 0]) for i in range(N_joints)]
my_ilcs = [ILC(dt=dt, sys=ApolloDynSys(dt, input_is_velocity), kf_dpn_params=kf_dpn_params, x_0=[0.0, 0.0]) for i in range(N_joints)]
for ilc in my_ilcs:
  ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball




for j in range(ILC_it):
  # Learn feed-forward signal
  # u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=joints_traj_des[:, i], y_meas=y_meas[:, i]) for i, ilc in enumerate(my_ilcs)]
  u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=joints_traj_des[:, i]-q_start[i], y_meas=y_meas[:, i]) for i, ilc in enumerate(my_ilcs)]
  u_arr = np.array(u_ff, dtype='float').squeeze().reshape(N_joints, -1).T
  plt.figure()
  plt.plot(u_arr)
  plt.show()
  # Main Simulation
  q_s, q_v_s, q_a_s, dP_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_arr, joint_home_config=q_start, repetitions=1, it=j)

  # a. System output
  y_meas = q_s[1:]

  # 5. Collect data for plotting
  joints_q_vec[j, ] =  q_s
  joints_vq_vec[j, ] = q_v_s
  joints_aq_vec[j, ] = q_a_s
  xyz_vec[j, ] = rArmKinematics.seqFK(q_s)[:, :3, -1]

  joints_d_vec[j, 1:] = np.squeeze(joints_traj_des[1:]-y_meas)
  u_ff_vec[j, :-1] = u_vec
  torque_vec[j, ] = dP_N_vec

  print("ITERATION: " + str(j+1)
        + ", \n\Trajectory_track_error_norm: " + str(np.linalg.norm(joints_d_vec[j]))
        )


# %%
# Extra to catch the ball
q_s_ex, q_v_s_ex, q_a_s_ex, dP_N_vec_ex, u_vec_arr = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_arr, joint_home_config=home_pose, repetitions=5, it=j)

# Evauluate last iteration
# if j%(ILC_it-1)==0:
if True:
  q_s_full =   np.append(q_s[1:], q_s_ex, 0)
  q_v_s_full =   np.append(q_v_s[1:], q_v_s_ex, 0)
  q_a_s_full =   np.append(q_a_s[1:], q_a_s_ex, 0)
  dP_N_vec_full =  np.append(dP_N_vec[1:], dP_N_vec_ex, 0)
  dP_N_vec_full =  np.append(dP_N_vec[1:], dP_N_vec_ex, 0)
  u_vec_full =    np.append(u_vec, u_vec_arr, 0)
  y_dess = np.append(y_des[1:], np.tile(y_des[N_throw_empty+1:], extra_rep), 0)

  plot_simulation(dt,
                  F_vec_full, x_b_vec_full, u_b_vec_full, x_p_vec_full,
                  u_p_vec_full, dP_N_vec_full, gN_vec_full, y_dess,
                  title="Iteration: " + str(j),
                  vertical_lines={T_throw_first:        "T_throw1",              T_throw_first+T_empty: "T_catch_released_ball",
                                  T_FULL-T_empty: "T_throw_released_ball", T_FULL:          "T_catch1",
                                  T_FULL-T_empty+T_fly:          "T_catch_released_ball"})
# Plot the stuff
# plotIterations(u_ff_vec.T, "uff", dt, every_n=2)
# plotIterations(d_vec.T, "Error on plate trajectory", dt, every_n=2)
# plotIterations(x_p_vec.T, "Plate trajectory", dt, every_n=2)
# plotIterations(u_throw_vec, "ub0", every_n=1)
plt.show()