# %%
# #!/usr/bin/env python2
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.JugglingPlanner import calc

def learnThrow(T_fly, T_throw, Hb, ub_throw, x_catch, x_ruhe=-0.4, ILC_it=15, plot=False):
  # %%
  print("juggling_apollo")

  # Throw and catch point
  # T_fly, ub_throw = plan_ball_trajectory(hb=Hb)  # important since input cannot influence the first state
  # T_throw = T_fly/fly_throw_ratio
  N_throw = steps_from_time(T_throw, dt)-1  # size of our vectors(i.e. length of the interval)
  N_fly = steps_from_time(T_fly, dt)-1  # size of our vectors(i.e. length of the interval)

  # %%
  kf_dpn_params = {
    'M': 0.1*np.eye(N_throw, dtype='float'),
    'd0': np.zeros((N_throw, 1), dtype='float'),
    'P0': 0.1*np.eye(N_throw, dtype='float'),
    'epsilon0': 0.3,
    'epsilon_decrease_rate': 1
  }
  # Init state
  x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe
  my_ilc = ILC(dt, kf_dpn_params=kf_dpn_params, x_0=x0, t_f=T_fly, t_h=T_throw)
  sim = Simulation(x0=x0, input_is_force=False, air_drag=True, plate_friction=True)
  sim.reset()
  
  # Learn Throw
  # reset ilc
  my_ilc.initILC(N_1=N_throw, impact_timesteps=[False]*N_throw)  # ignore ball
  y_des, u_ff, ub_throw = my_ilc.learnThrowStep(ub_throw, x_catch=x_catch, T_throw=T_throw)

  # Extra simulation to measure time of flight
  T_sim_extra = 2*T_fly
  N_sim_extra = steps_from_time(T_sim_extra, dt)
  N_fly_2 = steps_from_time(T_fly/2, dt)
  T_catch_wait = T_sim_extra - T_fly/2
  N_catch_wait = N_sim_extra - N_fly_2
  
  # collect: dup, x_p, x_b, u_p
  dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
  x_p_vec = np.zeros([ILC_it+1, N_throw + 1], dtype='float')
  x_b_vec = np.zeros([ILC_it, N_sim_extra], dtype='float')
  u_p_vec = np.zeros([ILC_it, N_throw + 1], dtype='float')
  u_des_vec = np.zeros([ILC_it, N_throw], dtype='float')
  u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
  u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
  u_T_fly_vec = np.zeros([ILC_it, 1], dtype='float')
  u_real_dist_vec = np.zeros([ILC_it+1, my_ilc.kf_dpn.d.size], dtype='float')

  # ILC Loop
  d1_meas = 0
  d2_meas = 0
  y_meas = None

  # disturbance to be learned
  period = 0.02/dt
  disturbance = 100*np.sin(2*np.pi/period*np.arange(N_throw), dtype='float')  # disturbance on the plate position(0:my_ilc.N_throw-1)
  for j in range(ILC_it):

    # Main Simulation
    x000 = [x0[0], x0[1], x0[2], x0[3]]
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
      sim.simulate_one_iteration(dt=dt, T=T_throw, x0=x000, u=u_ff, d=disturbance)

    # Extra simulation to measure time of flight
    x000 = [x_b[-1], 0, u_b[-1], 0]
    [xb_1, u_bb_1, x_pp_1, u_pp_1, dP_NN_1, gN_vecc_1, F_vecc_1] = \
      sim.simulate_one_iteration(dt=dt, T=T_fly/2, x0=x000, u=np.zeros([N_fly_2, 1]))
    x000 = [xb_1[-1], x_catch, u_bb_1[-1], 0]
    [xb_2, u_bb_2, x_pp_2, u_pp_2, dP_NN_2, gN_vecc_2, F_vecc_2] = \
      sim.simulate_one_iteration(dt=dt, T=T_catch_wait, x0=x000, u=np.zeros([N_catch_wait, 1]))

    x_b_extra = np.vstack([xb_1[1:], xb_2[1:]])
    gN_vecc_extra = np.vstack([gN_vecc_1[1:], gN_vecc_2[1:]])

    # Measurments
    fly_time_meas = (N_fly_2+np.argmax(gN_vecc_extra[N_fly_2:]<=1e-5))*dt
    y_meas = x_p[1:]
    d1_meas = max(x_b) - Hb  # disturbance on height
    d2_meas = fly_time_meas - T_fly  # disturbance on ball flight time
    u_real_dist_vec[j, :] = np.squeeze(y_meas)-y_des[1:].T

    # LEARN THROW
    y_des, u_ff, ub_throw = my_ilc.learnThrowStep(ub_throw=ub_throw, x_catch=x_catch, T_throw=T_throw, 
                                                  u_ff_old=u_ff, y_meas=y_meas, 
                                                  d1_meas=d1_meas, d2_meas=d2_meas)

    # 5. Collect data for plotting
    dup_vec[j, :] = np.squeeze(my_ilc.kf_dpn.d)
    x_p_vec[j, :] = np.squeeze(x_p)
    x_b_vec[j, :] = np.squeeze(x_b_extra)
    u_p_vec[j, :] = np.squeeze(u_p)
    u_des_vec[j, :] = np.squeeze(u_ff)
    u_d2_vec[j] = d2_meas  # ub_throw
    u_b0_vec[j] = ub_throw
    u_T_fly_vec[j] = fly_time_meas
    print("ITERATION: " + str(j+1)  # noqa
          + ", Ub0: " + str(ub_throw) + ", " + str(u_p[-1])  # flake8: W503
          + ", Fly time: " + str(fly_time_meas)  # flake8: W503
          + ", Error on fly time: " + str(d2_meas))  # noqa: W503

  x_b = np.vstack([x_b, xb_1, xb_2])
  u_b = np.vstack([u_b, u_bb_1, u_bb_2])
  x_p = np.vstack([x_p, x_pp_1, x_pp_2])
  u_p = np.vstack([u_p, u_pp_1, u_pp_2])
  dP_N_vec = np.vstack([dP_N_vec, dP_NN_1, dP_NN_2])
  gN_vec = np.vstack([gN_vec, gN_vecc_1, gN_vecc_2])
  F_vec = np.vstack([F_vec, F_vecc_1, F_vecc_2])

  # print(y_meas.shape, y_des.shape)
  
  # %%
  # Plot the stuff
  if plot:
    plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, vertical_lines={T_throw: "T_throw", T_throw+T_fly: "T_catch"})
    # u_real_dist_vec[-1, :] = np.squeeze(my_ilc.kf_dpn.d)
    # x_p_vec[-1, :] = y_des
    # plotIterations(u_real_dist_vec.T, "measured dist", dt, every_n=3)
    # plotIterations(dup_vec.T, "dup", dt, every_n=3)
    # plotIterations(x_p_vec.T, "x_p", dt, every_n=1)
    plotIterations(u_b0_vec, "ub0", every_n=3)
    # plotIterations(x_b_vec.T, "x_b", dt, every_n=3)
    # plotIterations(u_T_fly_vec, "T_fly", every_n=3)
    plotIterations(u_d2_vec, "d2", every_n=3)
    plt.show()
  
  return u_ff, ub_throw, 0.0, T_fly, T_throw


if __name__ == "__main__":
  # Init state
  x_ruhe = -0.4
  E = 0.25
  tau = 0.53
  dwell_ration = 0.68
  catch_throw_ratio = 0.5
  T_throw, T_hand, ub_catch, ub_throw, T_empty, H, z_catch = calc(tau, dwell_ration, catch_throw_ratio, E)
  T_fly = 2*T_empty + T_hand
  
  learnThrow(T_fly=T_fly, T_throw=T_throw, Hb=H, ub_throw=ub_throw, x_ruhe=x_ruhe, x_catch=z_catch, ILC_it=115, plot=True)