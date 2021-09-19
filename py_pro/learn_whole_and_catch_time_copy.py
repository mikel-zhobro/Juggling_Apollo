# %%
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.JugglingPlanner import calc, traj_nb_2_na_1
# %%
print("juggling_apollo")


# Init state
x_ruhe = -0.2                # starting position for the plate
x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe
E = 0.25
tau = 0.53
dwell_ration = 0.68
catch_throw_ratio = 0.5
T_throw, T_hand, ub_catch, ub_throw, T_empty, H,  z_catch = calc(tau, dwell_ration, catch_throw_ratio, E)
y_des = traj_nb_2_na_1(T_throw, T_hand, ub_catch, ub_throw, T_empty, z_catch, [x_ruhe,x_ruhe], dt, smooth_acc=False, plot=True)

T_fly = T_hand + 2*T_empty
T_FULL = T_throw + T_fly
N_1 = steps_from_time(T_FULL, dt)-1    # size of our vectors(i.e. length of the interval)
N_throw = steps_from_time(T_throw, dt)-1   # needed to measure z_throw
N_throw_empty = steps_from_time(T_throw+T_empty, dt)-1   # needed to measure z_throw
N_half_1 = int(N_1/3)                  # needed to measure z_catch and T_catch

print('H: ' + str(H))
print('T_fly: ' + str(T_fly))

# %%
kf_dpn_params = {
  'M': 0.031*np.eye(N_1, dtype='float'),      # covariance of noise on the measurment
  'P0': 0.1*np.eye(N_1, dtype='float'),     # initial disturbance covariance
  'd0': np.zeros((N_1, 1), dtype='float'),  # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1              # the decreasing factor of noise on the disturbance
}

my_ilc = ILC(dt, kf_dpn_params=kf_dpn_params, x_0=x0, t_f=0, t_h=0)

sim = Simulation(x0=x0, input_is_force=False, air_drag=False, plate_friction=True)
sim.reset()

# Learn Throw
ILC_it = 20  # number of ILC iteration

# reset ilc
my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball
y_des, u_ff, ub_throw, t_catch = my_ilc.learnWhole2(ub_throw=ub_throw, ub_catch=ub_catch,
                                                    T_throw=T_throw, T_hand=T_hand, T_empty=T_empty,
                                                    z_catch=z_catch)

# collect: dup, x_p, x_b, u_p
dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
x_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_des_vec = np.zeros([ILC_it, my_ilc.N_1], dtype='float')
u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
t_catch_vec = np.zeros([ILC_it, 1], dtype='float')
u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
u_Tb_vec = np.zeros([ILC_it, 1], dtype='float')

# ILC Loop
d1_meas = 0
d2_meas = 0
y_meas = None

# disturbance to be learned
period = 0.02/dt
# TODO: (disturbance can be a generator function)
x000 =  [[x0[0],2*H], x0[1], [x0[2],0], x0[3]]
disturbance = 150*np.sin(2*np.pi/period*np.arange(my_ilc.N_1), dtype='float')  # disturbance on the plate position(0:my_ilc.N_1-1)
sim.simulate_one_iteration(dt=dt, T=T_throw+T_empty, x0=x000, u=u_ff[:N_throw_empty], d=disturbance[:N_throw_empty], repetitions=1, visual=True)
sim.simulate_one_iteration(dt=dt, T=T_FULL-T_throw-T_empty, u=u_ff[N_throw_empty:], d=disturbance[N_throw_empty:], repetitions=5, visual=True)
for j in range(ILC_it):
  # Main Simulation
  x000 =  [[x0[0],2*H], x0[1], [x0[2],0], x0[3]]
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt=dt, T=T_FULL, x0=x000, u=u_ff, d=disturbance, it=j)

  # Measurments
  N_fly_time = min(N_1, N_half_1+np.argmax(gN_vec[N_half_1:,1]<=1e-5))
  fly_time_meas = N_fly_time*dt - T_throw
  y_meas = x_p[1:]
  d2_meas = fly_time_meas - T_fly  # disturbance on ball flight time
  d3_meas = 0
  # d3_meas = d2_meas
  N_catch_1 = steps_from_time(T_empty/2, dt)-1
  # d3_meas = - max(gN_vec[-N_catch_1:])
  # d3_meas =max(x_b[-N_h2_1:] - x0[0])

  # LEARN THROW
  y_des, u_ff, ub_throw, t_catch = my_ilc.learnWhole2(ub_throw=ub_throw, ub_catch=ub_catch,
                                                      u_ff_old=u_ff, y_meas=y_meas, d2_meas=d2_meas, d3_meas=0,
                                                      T_throw=T_throw, T_hand=T_hand, T_empty=T_empty,
                                                      z_catch=z_catch)

  # 5. Collect data for plotting
  # dup_vec[j, :] = np.squeeze(my_ilc.kf_dpn.d)
  dup_vec[j, :] = np.squeeze(y_des[1:]-np.squeeze(y_meas[:]))
  x_p_vec[j, :] = np.squeeze(x_p)
  u_p_vec[j, :] = np.squeeze(u_p)
  u_des_vec[j, :] = np.squeeze(u_ff)
  u_d2_vec[j] = d2_meas
  u_b0_vec[j] = ub_throw
  t_catch_vec[j] = t_catch
  u_Tb_vec[j] = fly_time_meas
  print("ITERATION: " + str(j+1)  # noqa
        + ", \n\tUb0: " + str(ub_throw)
        + ", \n\tFly time: " + str(fly_time_meas)  # flake8: W503
        + ", \n\tError on fly time: " + str(d2_meas)
        + ", \n\tError on catch hight: " + str(x_p[N_fly_time])
        + ", \n\tThrow/Catch time: " + str(T_throw) + " / " + str(T_hand-T_throw)  # flake8: W503
        + ", \n\tBiggest jump after catch: " + str(-d3_meas)
        )  # noqa: W503
  # if j%6==0:
  #   sim.simulate_one_iteration(dt=dt, T=Tb+Th, x_b0=x0[0], x_p0=x0[1], u_b0=x0[2], u_p0=x0[3], u=u_ff, d=disturbance, visual=True, repetitions=3, pause_on_hight=0)
plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, y_des, title="Iteration: " + str(ILC_it),
                vertical_lines={T_hand-T_throw: "T_throw", T_hand: "T_catch"})
x_H=np.max(x_b)
u_H=0
x000= [[x0[0], x_H], x0[1], [x0[2], u_H], x0[3]]
sim.simulate_one_iteration(dt=dt, T=T_throw+T_empty, x0=x000, u=u_ff[:N_throw_empty], d=disturbance[:N_throw_empty], repetitions=1, visual=True, slow=10)
sim.simulate_one_iteration(dt=dt, T=T_FULL-T_throw-T_empty, u=u_ff[N_throw_empty:], d=disturbance[N_throw_empty:], repetitions=5, visual=True)


print("T_fly: ", T_fly, " T_throw", T_throw)
plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, y_des, title="Iteration: " + str(ILC_it),
                vertical_lines={T_hand-T_throw: "T_throw", T_hand: "T_catch"})
# plt.plot(u_b_catch_vec)
# %%
# plt.plot(np.squeeze(x_b), label="x_b")
# plt.plot(y_des, label="y_des")
# plt.legend()
# plt.plot()
# (y_des[1:].shape-np.squeeze(y_meas[:])).shape

# plt.plot(y_des[1:]-np.squeeze(y_meas[:]))
# plt.show()
# plotIterations(t_catch_vec, "t_catch", dt, every_n=3)
# plotIterations(dup_vec.T[:,5:], "dup", dt, every_n=3)

# plt.plot(x_p)
# plt.plot(y_des)

# %%
# Plot the stuff
# plotIterations(u_des_vec.T, "uff", dt, every_n=1)
# plotIterations(dup_vec.T, "dup", dt, every_n=1)
# plotIterations(x_p_vec.T, "x_p", dt, every_n=1)
# plotIterations(u_Tb_vec-Th/2, "Tb", every_n=1)
# plotIterations(u_b0_vec, "ub0", every_n=1)
# plotIterations(u_d2_vec, "d2", every_n=1)
# %%

plt.show()
