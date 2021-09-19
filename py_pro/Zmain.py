# %%
# #!/usr/bin/env python2
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC

# %%
print("juggling_apollo")

# Throw and catch point
Hb = 1
Tb, ub_00 = plan_ball_trajectory(hb=Hb)  # important since input cannot influence the first state
Th = Tb/2
N_1 = steps_from_time(Th/2, dt)-1  # size of our vectors(i.e. length of the interval)
# Init state
x_ruhe = -0.4
x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe

# %%
kf_dpn_params = {
  'M': 0.1*np.eye(N_1, dtype='float'),
  'd0': np.zeros((N_1, 1), dtype='float'),
  'P0': 0.1*np.eye(N_1, dtype='float'),
  'epsilon0': 0.1,
  'epsilon_decrease_rate': 0.9
}

my_ilc = ILC(dt, kf_dpn_params=kf_dpn_params, x_0=x0, t_f=Tb, t_h=Th)

sim = Simulation(x0=x0, input_is_force=False, air_drag=True, plate_friction=True)
sim.reset()
# Learn Throw
ILC_it = 25  # number of ILC iteration
ub_0 = ub_00
# reset ilc
my_ilc.initILC(N_1=N_1, impact_timesteps=[True]*N_1)
_, u_ff, ub_0 = my_ilc.learnThrowStep(ub_00)

# Extra simulation to measure time of flight
T_sim_extra = 2*Tb
N_sim_extra = steps_from_time(T_sim_extra, dt)

# collect: dup, x_p, x_b, u_p
dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
x_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
x_b_vec = np.zeros([ILC_it, N_sim_extra], dtype='float')
u_p_vec = np.zeros([ILC_it, my_ilc.N_1 + 1], dtype='float')
u_des_vec = np.zeros([ILC_it, my_ilc.N_1], dtype='float')
u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
u_Tb_vec = np.zeros([ILC_it, 1], dtype='float')

# ILC Loop
d1_meas = 0
d2_meas = 0
y_meas = None

# disturbance to be learned
period = 0.02/dt
disturbance = 200*np.sin(2*np.pi/period*np.arange(my_ilc.N_1), dtype='float')  # disturbance on the plate position(0:my_ilc.N_1-1)
for j in range(ILC_it):

  # Main Simulation
  x0 = [x0[0], x0[1], x0[2], x0[3]]
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
    sim.simulate_one_iteration(dt=dt, T=my_ilc.t_h/2, x0=x0, u=u_ff, d=disturbance)

  # Extra simulation to measure time of flight
  x0 = [x_b[-1], 0, u_b[-1], 0]
  [x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra, F_vec_extra] = \
    sim.simulate_one_iteration(dt=dt, T=2*Tb, x0=x0, u=np.zeros([N_sim_extra, 1]))

  # Measurments
  fly_time_meas = (1+np.argmax(x_b_extra[1:]<=1e-5))*dt
  y_meas = x_p[1:]
  d1_meas = max(x_b) - Hb  # disturbance on height
  d2_meas = fly_time_meas - Tb  # disturbance on ball flight time

  # LEARN THROW
  y_des, u_ff, ub_0 = my_ilc.learnThrowStep(ub_0=ub_0, u_ff_old=u_ff, y_meas=y_meas, d1_meas=d1_meas, d2_meas=d2_meas)

  # 5. Collect data for plotting
  dup_vec[j, :] = np.squeeze(my_ilc.kf_dpn.d)
  x_p_vec[j, :] = np.squeeze(x_p)
  x_b_vec[j, :] = np.squeeze(x_b_extra)
  u_p_vec[j, :] = np.squeeze(u_p)
  u_des_vec[j, :] = np.squeeze(u_ff)
  u_d2_vec[j] = d2_meas  # ub_0
  u_b0_vec[j] = ub_0
  u_Tb_vec[j] = fly_time_meas
  print("ITERATION: " + str(j+1)  # noqa
        + ", Fly time: " + str(fly_time_meas)  # flake8: W503
        + ", Error on fly time: " + str(d2_meas))  # noqa: W503

plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
plot_simulation(dt, F_vec_extra, x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra)


# %%
# Plot the stuff
plotIterations(dup_vec.T, "dup", dt, every_n=3)
plotIterations(x_p_vec.T, "x_p", dt, every_n=3)
plotIterations(x_b_vec.T, "x_b", dt, every_n=3)
plotIterations(u_Tb_vec, "Tb", every_n=3)
plotIterations(u_b0_vec, "ub0", every_n=3)
plotIterations(u_d2_vec, "d2", every_n=3)
