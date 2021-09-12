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
my_ilc.initILC(N_1=N_1, impact_timesteps=[False]*N_1)  # ignore the ball
y_des, u_ff, ub_0 = my_ilc.learnWhole(ub_00)

sim = Simulation(input_is_force=False, air_drag=True, plate_friction=True)

x_b0 = [-0.4, 0.6]
u_b0 = [0.0, 0.0]
# Main Simulation
[x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
  sim.simulate_one_iteration(dt=dt, T=Tb+Th, x_b0=x_b0, x_p0=x0[1], u_b0=u_b0, u_p0=x0[3], u=u_ff, visual=True, repetitions=6)

plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
