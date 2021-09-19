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
kf_dpn_params = {
  'M': 0.1*np.eye(N_1, dtype='float'),      # covariance of noise on the measurment
  'P0': 0.1*np.eye(N_1, dtype='float'),     # initial disturbance covariance
  'd0': np.zeros((N_1, 1), dtype='float'),  # initial disturbance value
  'epsilon0': 0.3,                          # initial variance of noise on the disturbance
  'epsilon_decrease_rate': 1              # the decreasing factor of noise on the disturbance
}

my_ilc = ILC(dt, kf_dpn_params=kf_dpn_params, x_0=x0, t_f=Tb, t_h=Th)
my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball
y_des, u_ff, ub_0 = my_ilc.learnWhole(ub_00)

sim = Simulation(input_is_force=False, x0=x0, air_drag=True, plate_friction=True)
sim.reset()
x_b0 = [-0.4, 0.6, 0.3]
u_b0 = [0.0, 0.0, 0.0]
x_b0 = [-0.4]
u_b0 = [0.0]

ILC_it = 23
# collect: dup, x_p, x_b, u_p
dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
x_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
x_b_vec = np.zeros([ILC_it, my_ilc.N_1 + 1, len(x_b0)], dtype='float')
u_des_vec = np.zeros([ILC_it, my_ilc.N_1], dtype='float')

u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
t_catch_vec = np.zeros([ILC_it, 1], dtype='float')
u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
u_Tb_vec = np.zeros([ILC_it, 1], dtype='float')


# Main Simulation
for j in range(ILC_it):
  x000=[x_b0, x0[1], u_b0, x0[3]]
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
    sim.simulate_one_iteration(dt=dt, T=Tb+Th, x0=x000, u=u_ff, repetitions=1)

  # 5. Collect data for plotting
  u_des_vec[j, :] = np.squeeze(u_ff)
  x_p_vec[j, :] = np.squeeze(x_p)
  u_p_vec[j, :] = np.squeeze(u_p)
  x_b_vec[j] = x_b
  # dup_vec[j, :] = np.squeeze(y_des[1:]-np.squeeze(y_meas[:]))
  # u_d2_vec[j] = d2_meas
  # u_b0_vec[j] = ub_0
  # t_catch_vec[j] = t_catch
  # u_Tb_vec[j] = fly_time_meas
  # print("ITERATION: " + str(j+1)
  #       + ", \n\tUb0: " + str(ub_0) + ", " + str(u_p[N_h2_1-1])
  #       + ", \n\tFly time: " + str(fly_time_meas)
  #       + ", \n\tError on fly time: " + str(d2_meas)
  #       + ", \n\tError on catch hight: " + str(x_p[N_fly_time])
  #       + ", \n\tThrow/Catch time: " + str(t_throw) + " / " + str(t_catch)
  #       + ", \n\tBiggest jump after catch: " + str(-d3_meas)
  #       )
sim.simulate_one_iteration(dt=dt, T=Tb+Th, x0=x000, u=u_ff, visual=True, repetitions=2)
plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)


plt.show()