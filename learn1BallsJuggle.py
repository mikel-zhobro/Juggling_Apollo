# %%
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt, ABS, g, ABS
from juggling_apollo.ILC import ILC
from juggling_apollo.JugglingPlanner import calc, traj_nb_2_na_1
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import BallAndPlateDynSys as DynamicSystem
from kinematics.fk import FK, IK_trajectory

# %%
print("juggling_apollo")


E = 0.15
# E = 0.0
tau = 0.5
dwell_ration = 0.6
catch_throw_ratio = 0.5
T_hand, ub_throw, T_empty, H,  z_catch = calc(tau, dwell_ration, E)
T_throw = T_hand*(1-catch_throw_ratio)

T_fly = T_hand + 2*T_empty
T_FULL = T_throw + T_empty + T_hand - T_throw
N_1 = steps_from_time(T_FULL, dt)-1    # size of our vectors(i.e. length of the interval)
N_throw = steps_from_time(T_throw, dt)-1   # needed to measure z_throw
N_throw_empty = steps_from_time(T_throw+T_empty, dt)-1   # needed to measure z_throw
N_repeat_point = steps_from_time(T_throw, dt)-1                  # needed to measure z_catch and T_catch
N_repeat_point = 1               # needed to measure z_catch and T_catch
# N_catch_1=steps_from_time(T_hand-T_throw, dt)-1

print('H: ' + str(H))
print('T_fly: ' + str(T_fly))

# Init state
x_ruhe = -0.1                # starting position for the plate
x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe
x000 =  [[x0[0]], x0[1], [x0[2]], x0[3]]

# %%
# Learn Throw
ILC_it = 50  # number of ILC iteration


# Init ilc
# Here we want to set some convention to avoid missunderstandins later on.
# 1. the state is [xb, xp, ub, up]^T
# 2. the system can have as input either velocity u_des or the force F_p
# I. SYSTEM DYNAMICS
input_is_velocity = True
sys = DynamicSystem(dt, input_is_velocity=input_is_velocity)
kf_dpn_params = {
  'M': 0.031*np.eye(N_1, dtype='float'),      # covariance of noise on the measurment
  'P0': 0.1*np.eye(N_1, dtype='float'),     # initial disturbance covariance
  'd0': np.zeros((N_1, 1), dtype='float'),  # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1              # the decreasing factor of noise on the disturbance
}
my_ilc = ILC(dt, sys, kf_dpn_params=kf_dpn_params, x_0=x0)
my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball

sim = Simulation(input_is_force=False, x0=x0, air_drag=True, plate_friction=True)
sim.reset()

# Data collection
# System Trajectories
x_p_vec = np.zeros([ILC_it, my_ilc.N_1+1], dtype='float')
u_p_vec = np.zeros([ILC_it, my_ilc.N_1+1], dtype='float')
x_b_vec = np.zeros([ILC_it, my_ilc.N_1+1, len(x000[0])], dtype='float')
u_b_vec = np.zeros([ILC_it, my_ilc.N_1+1, len(x000[0])], dtype='float')
# ILC Trajectories
d_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
u_ff_vec = np.zeros([ILC_it, my_ilc.N_1], dtype='float')
# Measurments
u_throw_vec = np.zeros([ILC_it, 1], dtype='float')
u_d_T_catch_1_vec = np.zeros([ILC_it, 1], dtype='float')

# ILC Loop
u_ff =None
y_meas = None
d_T_catch_1 = 0

# disturbance to be learned
period = 0.02/dt
disturbance = 150*np.sin(2*np.pi/period*np.arange(my_ilc.N_1), dtype='float')  # disturbance on the plate position(0:my_ilc.N_1-1)

# Min Jerk Params
# new MinJerk
smooth_acc = False
ub_catch = -ub_throw
i_a_end = None
# a0 = None;        a1 = None;       a2 = None;          a3 = None
tt=[0,        T_throw,     T_throw+T_empty,   T_FULL  ]
xx=[x0[0],    0.0,         z_catch,           x0[0]   ]
uu=[0.0,      ub_throw,    ub_catch,          0.0     ]
if True:
  print(uu[-1])
  xvaj = get_minjerk_trajectory(dt, tt=tt, xx=xx, uu=uu, smooth_acc=smooth_acc)
  plotMJ(dt, tt, xx, uu, smooth_acc, xvaj)
  joints_traj = IK_trajectory(xvaj[0], np.zeros_like(xvaj[0]), np.zeros((7, ), dtype='float'))
  plt.show()

extra_rep = 2
for j in range(ILC_it):
  # Learn feed-forward signal
  uu[1] = ub_throw  # update
  # uu[4] = ub_throw  # update
  y_des, velo, accel, jerk = get_minjerk_trajectory(dt, smooth_acc=smooth_acc, i_a_end=i_a_end, tt=tt, xx=xx, uu=uu)
  u_ff = my_ilc.learnWhole(u_ff_old=u_ff, y_des=y_des, y_meas=y_meas)

  # Main Simulation
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
    sim.simulate_one_iteration(dt=dt, T=T_FULL, x0=x000, u=u_ff, d=disturbance, it=j)

  # Extra to catch the ball
  [x_b_ex, u_b_ex, x_p_ex, u_p_ex, dP_N_vec_ex, gN_vec_ex, F_vec_ex] = \
    sim.simulate_one_iteration(dt=dt, T=T_hand+T_empty-dt, u=u_ff[N_repeat_point:], d=disturbance[N_repeat_point:], it=j, repetitions=extra_rep)

  # Measurments
  gN_vec_full = np.append(gN_vec, gN_vec_ex, 0)
  gN_vec_full_1 = gN_vec_full[:,0]
  # a. System output
  y_meas = x_p[1:]
  # b. Throw point
  N_throw_time = np.argmax(gN_vec_full_1[:N_throw_empty]>ABS)
  T_throw_time = N_throw_time*dt
  # c. Catch point
  N_catch_time = N_throw_empty+np.argmax(gN_vec_full_1[N_throw_empty:]<=ABS) # plate_ball
  d_T_catch_1 = N_catch_time*dt - T_throw - T_fly
  # N_catch_time2 = N_throw_empty+np.argmax(gN_vec_full_1[N_throw_empty:]<=ABS) # plate_ball
  # d_T_catch_1 = N_catch_time2*dt - T_FULL
  # Newton updates
  ub_throw = ub_throw - 0.06*g*d_T_catch_1

  # 5. Collect data for plotting
  d_vec[j, :] = np.squeeze(y_des[1:]-np.squeeze(y_meas[:]))
  x_p_vec[j, :] = np.squeeze(x_p)
  u_p_vec[j, :] = np.squeeze(u_p)
  u_ff_vec[j, :] = np.squeeze(u_ff)
  u_d_T_catch_1_vec[j] = d_T_catch_1
  u_throw_vec[j] = ub_throw
  print("ITERATION: " + str(j+1)
        + ", \n\tU_throw: " + str(ub_throw) + ", " + str(u_p[N_throw_time-1])
        + ", \n\tError on throw time: " + str(T_throw_time - T_throw)
        + ", \n\tError on throw hight: " + str(x_p[N_throw_time])
        + ", \n\tError on catch time: " + str(d_T_catch_1)
        + ", \n\tError on catch hight: " + str(np.append(x_p, x_p_ex, 0)[N_catch_time]-z_catch)
        )


# %%

# Evauluate last iteration
# if j%(ILC_it-1)==0:
if False:
  gN_vec_full =   np.append(gN_vec[1:], gN_vec_ex, 0)
  x_b_vec_full =  np.append(x_b[1:], x_b_ex, 0)
  x_p_vec_full =  np.append(x_p[1:], x_p_ex, 0)
  u_b_vec_full =  np.append(u_b[1:], u_b_ex, 0)
  u_p_vec_full =  np.append(u_p[1:], u_p_ex, 0)
  dP_N_vec_full = np.append(dP_N_vec[1:], dP_N_vec_ex, 0)
  F_vec_full =    np.append(F_vec, F_vec_ex, 0)
  # y_des = np.append(y_des, np.append(y_des[N_throw_empty+1:], y_des[N_throw_empty+1:]))
  y_dess = np.append(y_des[1:], np.tile(y_des[N_repeat_point+1:], extra_rep))
  plot_simulation(dt,
                  F_vec_full, x_b_vec_full, u_b_vec_full, x_p_vec_full,
                  u_p_vec_full, dP_N_vec_full, gN_vec_full, y_dess,
                  title="Iteration: " + str(j),
                  vertical_lines={T_throw:        "T_throw1",
                                  T_throw+T_fly:          "T_catch1"})
# Plot the stuff
# plotIterations(u_ff_vec.T, "uff", dt, every_n=2)
# plotIterations(d_vec.T, "Error on plate trajectory", dt, every_n=2)
# plotIterations(x_p_vec.T, "Plate trajectory", dt, every_n=2)
# plotIterations(u_throw_vec, "ub0", every_n=1)
  plotIterations(u_d_T_catch_1_vec, "Error on catch-released1", every_n=1)
  plt.show()

# %% Simulate
if True:
  visual = False
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
    sim.simulate_one_iteration(dt=dt, T=T_FULL, x0=x000, u=u_ff, d=disturbance, it=j, repetitions=1, visual=visual)

  # Extra to catch the ball
  extra_rep = 155
  [x_b_ex, u_b_ex, x_p_ex, u_p_ex, dP_N_vec_ex, gN_vec_ex, F_vec_ex] = \
    sim.simulate_one_iteration(dt=dt, T=T_hand+T_empty-dt, u=u_ff[N_repeat_point:], d=disturbance[N_repeat_point:], it=j, repetitions=extra_rep, visual=visual)


  gN_vec_full =   np.append(gN_vec[1:], gN_vec_ex, 0)
  x_b_vec_full =  np.append(x_b[1:], x_b_ex, 0)
  x_p_vec_full =  np.append(x_p[1:], x_p_ex, 0)
  u_b_vec_full =  np.append(u_b[1:], u_b_ex, 0)
  u_p_vec_full =  np.append(u_p[1:], u_p_ex, 0)
  dP_N_vec_full = np.append(dP_N_vec[1:], dP_N_vec_ex, 0)
  F_vec_full =    np.append(F_vec, F_vec_ex, 0)
  y_dess = np.append(y_des[1:], np.tile(y_des[N_repeat_point:-1], extra_rep))

  plot_simulation(dt, F_vec_full, x_b_vec_full, u_b_vec_full, x_p_vec_full, u_p_vec_full, dP_N_vec_full, gN_vec_full, y_dess, title="Iteration: " + str(ILC_it))
  plt.show()