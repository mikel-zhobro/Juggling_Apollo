# %%
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC

# %%
print("juggling_apollo")

# Throw and catch point
Hb = 1
Tb, ub_00 = plan_ball_trajectory(hb=Hb)  # important since input cannot influence the first state
Th = Tb/2
N_1 = steps_from_time(Tb+Th, dt)-1  # size of our vectors(i.e. length of the interval)
N_h2_1 = steps_from_time(Th/2, dt)-1  # size of our vectors(i.e. length of the interval)
N_half_1 = int(N_1/3)
# Init state
x_ruhe = -0.4
x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe

# %%
kf_d1d2_params = {
  'M': 0.1*np.eye(2, dtype='float'),        # covariance of noise on the measurment
  'P0': 0.2*np.eye(2, dtype='float'),       # initial disturbance covariance
  'd0': np.zeros((2, 1), dtype='float'),    # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1              # the decreasing factor of noise on the disturbance
}
kf_dpn_params = {
  'M': 0.1*np.eye(N_1, dtype='float'),      # covariance of noise on the measurment
  'P0': 0.1*np.eye(N_1, dtype='float'),     # initial disturbance covariance
  'd0': np.zeros((N_1, 1), dtype='float'),  # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1              # the decreasing factor of noise on the disturbance
}

my_ilc = ILC(dt, kf_d1d2_params=kf_d1d2_params, kf_dpn_params=kf_dpn_params, x_0=x0, t_f=Tb, t_h=Th)

sim = Simulation(input_is_force=False, air_drag=True, plate_friction=True)

# Learn Throw
ILC_it = 33  # number of ILC iteration
ub_0 = ub_00
t_catch = Tb+Th
# reset ilc
# my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball
# impact_timesteps =
my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball
y_des, u_ff, ub_0, ub_catch = my_ilc.learnWhole(ub_00, t_catch=t_catch)

# collect: dup, x_p, x_b, u_p
dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
x_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_des_vec = np.zeros([ILC_it, my_ilc.N_1], dtype='float')
u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
u_b_catch_vec = np.zeros([ILC_it, 1], dtype='float')
u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
u_Tb_vec = np.zeros([ILC_it, 1], dtype='float')

# ILC Loop
d1_meas = 0
d2_meas = 0
y_meas = None

# disturbance to be learned
period = 0.02/dt
disturbance = 250*np.sin(2*np.pi/period*np.arange(my_ilc.N_1), dtype='float')  # disturbance on the plate position(0:my_ilc.N_1-1)
for j in range(ILC_it):

  # Main Simulation
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
    sim.simulate_one_iteration(dt=dt, T=Tb+Th, x_b0=x0[0], x_p0=x0[1], u_b0=x0[2], u_p0=x0[3], u=u_ff, d=disturbance)

  # Measurments
  N_fly_time = N_half_1+np.argmax(gN_vec[N_half_1:]<=1e-5)
  fly_time_meas = N_fly_time*dt
  y_meas = x_p[1:]
  d1_meas = max(x_b) - Hb  # disturbance on height
  d2_meas = fly_time_meas - Th/2 - Tb  # disturbance on ball flight time
  # d3_meas = 0
  # d3_meas = d2_meas
  d3_meas = - max(gN_vec[-N_h2_1:])
  # d3_meas =max(x_b[-N_h2_1:] - x0[0])

  # LEARN THROW
  y_des, u_ff, ub_0, t_catch = my_ilc.learnWhole(ub_0=ub_0, t_catch=t_catch, u_ff_old=u_ff, y_meas=y_meas, d1_meas=d1_meas, d2_meas=d2_meas, d3_meas=d3_meas)

  # 5. Collect data for plotting
  # dup_vec[j, :] = np.squeeze(my_ilc.kf_dpn.d)
  dup_vec[j, :] = np.squeeze(y_des[1:]-np.squeeze(y_meas[:]))
  x_p_vec[j, :] = np.squeeze(x_p)
  u_p_vec[j, :] = np.squeeze(u_p)
  u_des_vec[j, :] = np.squeeze(u_ff)
  u_d2_vec[j] = my_ilc.kf_d1d2.d[1]  # ub_0
  u_b0_vec[j] = ub_0
  u_b_catch_vec[j] = ub_catch
  u_Tb_vec[j] = fly_time_meas
  print("ITERATION: " + str(j+1)  # noqa
        + ", \n\tUb0: " + str(ub_0) + ", " + str(u_p[N_h2_1-1])  # flake8: W503
        + ", \n\tFly time: " + str(fly_time_meas-Th/2)  # flake8: W503
        + ", \n\tError on fly time: " + str(d2_meas)
        + ", \n\tError on catch hight: " + str(x_p[N_fly_time])
        + ", \n\tBiggest jump after catch: " + str(d3_meas)
        )  # noqa: W503

print("Tb: ", Tb, " Th/2", Th/2)
# plt.plot(u_b_catch_vec)
plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
# %%
# plt.plot(np.squeeze(x_b), label="x_b")
# plt.plot(y_des, label="y_des")
# plt.legend()
# plt.plot()
# (y_des[1:].shape-np.squeeze(y_meas[:])).shape

# plt.plot(y_des[1:]-np.squeeze(y_meas[:]))
# plt.show()
plotIterations(dup_vec.T, "dup", dt, every_n=3)

# plt.plot(x_p)
# plt.plot(y_des)

# %% Run the simulation 5 repetitions
# [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
#   sim.simulate_one_iteration(dt=dt, T=Tb+Th, x_b0=x0[0], x_p0=x0[1], u_b0=x0[2], u_p0=x0[3], u=u_ff, repetitions=5)
# plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
# %%
# Plot the stuff
# plotIterations(u_des_vec.T, "uff", dt, every_n=1)
# plotIterations(dup_vec.T, "dup", dt, every_n=1)
# plotIterations(x_p_vec.T, "x_p", dt, every_n=1)
# plotIterations(u_Tb_vec-Th/2, "Tb", every_n=1)
# plotIterations(u_b0_vec, "ub0", every_n=1)
# plotIterations(u_d2_vec, "d2", every_n=1)
# %%
