# %%
import numpy as np
from juggling_apollo.settings import dt, g, ABS
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.ILC import ILC
from learnThrow import learnThrow

# %%
print("juggling_apollo")

# Throw and catch point
Hb = 1
Tb, ub_00 = plan_ball_trajectory(hb=Hb)  # important since input cannot influence the first state
Th = 2*Tb/3
N_1 = steps_from_time(Tb+Th, dt)-1  # size of our vectors(i.e. length of the interval)
N_h2_1 = steps_from_time(Th/2, dt)-1  # size of our vectors(i.e. length of the interval)
N_half_1 = int(N_1/3)
# Init state
x_ruhe = -0.4
x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe

# %%
# Learn Throw
ILC_it = 13  # number of ILC iteration
ub_0 = ub_00
t_catch = 2*Th/3
N_catch_1=steps_from_time(t_catch, dt)-1
t_throw = Th - t_catch

# Init ilc
kf_dpn_params = {
  'M': 0.031*np.eye(N_1, dtype='float'),      # covariance of noise on the measurment
  'P0': 0.1*np.eye(N_1, dtype='float'),     # initial disturbance covariance
  'd0': np.zeros((N_1, 1), dtype='float'),  # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1              # the decreasing factor of noise on the disturbance
}
my_ilc = ILC(dt, kf_dpn_params=kf_dpn_params, x_0=x0)
my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball


sim = Simulation(input_is_force=False, x0=x0, air_drag=True, plate_friction=True)
sim.reset()


# collect: dup, x_p, x_b, u_p
dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
x_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_des_vec = np.zeros([ILC_it, my_ilc.N_1], dtype='float')
u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
u_Tb_vec = np.zeros([ILC_it, 1], dtype='float')

# ILC Loop
u_ff =None
d1_meas = 0
d2_meas = 0
y_meas = None

# disturbance to be learned
period = 0.02/dt
disturbance = 150*np.sin(2*np.pi/period*np.arange(my_ilc.N_1), dtype='float')  # disturbance on the plate position(0:my_ilc.N_1-1)

# Min Jerk Params
# new MinJerk
t_end = Tb + Th
# a0 = None;        a1 = None;       a2 = None;          a3 = None
tt=[0,        t_throw,    t_throw+Tb/4,    t_end-t_catch,    t_end]
xx=[x0[0],    0.0,       -0.2,             0.0,              x0[0]]
uu=[x0[2],    ub_0,       0.0,            -ub_0/3,           x0[2]]

for j in range(ILC_it):
  # Learn feed-forward signal
  uu[1] = ub_0
  y_des, u_ff = my_ilc.learnWhole(tt=tt, xx=xx, uu=uu, u_ff_old=u_ff, y_meas=y_meas)

  # Main Simulation
  x000 = [x0[0], x0[1], x0[2], x0[3]]
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
    sim.simulate_one_iteration(dt=dt, T=Tb+Th, x0=x000, u=u_ff, d=disturbance, it=j)
  # if j % 3 == 0:
  #    sim.simulate_one_iteration(dt=dt, T=Tb+Th, x0=x000, u=u_ff, d=disturbance, it=j, visual=True, repetitions=3)
  #    plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, y_des, title="Iteration: " + str(j),
  #                    vertical_lines={Th-t_catch: "T_throw", Tb+Th-t_catch: "T_catch"})

  # Measurments
  # a. System output
  y_meas = x_p[1:]
  # b. Throw point
  N_throw_time = np.argmax(gN_vec[:N_half_1]>ABS)
  T_throw_time = N_throw_time*dt
  # c. Catch point
  N_catch_time = N_half_1+np.argmax(gN_vec[N_half_1:]<=ABS)
  fly_time_meas = N_catch_time*dt - t_throw
  d2_meas = fly_time_meas - Tb  # disturbance on ball flight time
  # d. Ball Height
  d1_meas = max(x_b) - Hb  # disturbance on height

  # Newton updates
  ub_0 = ub_0 - 0.3*0.5*g*d2_meas  # move in oposite direction of error # TODO: vectorized

  # 5. Collect data for plotting
  # dup_vec[j, :] = np.squeeze(my_ilc.kf_dpn.d)
  dup_vec[j, :] = np.squeeze(y_des[1:]-np.squeeze(y_meas[:]))
  x_p_vec[j, :] = np.squeeze(x_p)
  u_p_vec[j, :] = np.squeeze(u_p)
  u_des_vec[j, :] = np.squeeze(u_ff)
  u_d2_vec[j] = d2_meas 
  u_b0_vec[j] = ub_0
  u_Tb_vec[j] = fly_time_meas
  print("ITERATION: " + str(j+1) 
        + ", \n\tUb0: " + str(ub_0) + ", " + str(u_p[N_h2_1-1]) 
        + ", \n\tFly time: " + str(fly_time_meas) 
        + ", \n\tError on throw time: " + str(T_throw_time - t_throw)
        + ", \n\tError on throw hight: " + str(x_p[N_throw_time])
        + ", \n\tError on catch time: " + str(d2_meas)
        + ", \n\tError on catch hight: " + str(x_p[N_catch_time])
        + ", \n\tThrow/Catch time: " + str(t_throw) + " / " + str(t_catch)
        ) 

# Evauluate last iteration
plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, y_des, title="Iteration: " + str(ILC_it),
                vertical_lines={Th-t_catch: "T_throw", Tb+Th-t_catch: "T_catch"})

# %%
# Plot the stuff
# plotIterations(u_des_vec.T, "uff", dt, every_n=2)
# plotIterations(dup_vec.T, "Error on plate trajectory", dt, every_n=2)
# plotIterations(x_p_vec.T, "Plate trajectory", dt, every_n=2)
# plotIterations(u_Tb_vec-Th/2, "T_fly", every_n=1)
# plotIterations(u_b0_vec, "ub0", every_n=1)
# plotIterations(u_d2_vec, "Error on catch-time", every_n=1)
# plt.show()

# %% Simulate
x_H=max(x_b); u_H=0
x000 = [[x0[0]], x0[1], [x0[2]], x0[3]]
sim.simulate_one_iteration(dt=dt, T=Tb+Th, x0=x000, u=u_ff, d=disturbance, 
                           visual=True, repetitions=12, pause_on_hight=0, it=ILC_it)