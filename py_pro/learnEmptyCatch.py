# %%
# #!/usr/bin/env python2
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import get_min_jerk_trajectory, get_minjerk_trajectory
from learnThrow import learnThrow
from juggling_apollo.JugglingPlanner import calc

def learnEmpty(uff_throw, T_fly, T_throw, u_b_throw, ub_catch, x_throw, x_catch, x_ruhe, ILC_it=15):
  # %%
  print("juggling_apollo")

  # Throw and catch point
  N_throw = steps_from_time(T_throw, dt)-1  # size of our vectors(i.e. length of the interval)
  N_fly = steps_from_time(T_fly, dt)-1  # size of our vectors(i.e. length of the interval)

  # %%
  kf_dpn_params = {
    'M': 0.1*np.eye(N_fly, dtype='float'),
    'd0': np.zeros((N_fly, 1), dtype='float'),
    'P0': 0.1*np.eye(N_fly, dtype='float'),
    'epsilon0': 0.3,
    'epsilon_decrease_rate': 1
  }
  # Init state
  # x0_throw = [[x_ruhe, H], [x_ruhe, 0.0], 0, 0]
  x0_throw = [x_ruhe, x_ruhe, 0, 0]
  x0 = [0.0, 0.0, ub_throw, ub_throw+0.01]
  my_ilc = ILC(dt, kf_dpn_params=kf_dpn_params, x_0=x0, t_f=T_fly)
  sim = Simulation(x0=x0_throw, input_is_force=False, air_drag=True, plate_friction=True)
  sim.reset()
  
  # Learn Throw
  # reset ilc
  my_ilc.initILC(N_1=N_fly, impact_timesteps=[True]*N_fly)  # ignore ball
  t1 = T_throw;    t2 = t1+T_empty;   t3 = t2+T_hand;   t4 = t3+T_empty; 
  x1 = x0[0];      x2 = z_catch;      x3 = x1;          x4 = z_catch;    
  u1 = x0[3];      u2 = ub_catch;     u3 = ub_throw;    u4 = ub_catch;   
  y_des, v, a, j = get_minjerk_trajectory(dt=dt, tt=[t1, t2, t3, t4], xx=[x1, x2, x3, x4], uu=[u1, u2, u3, u4])
  u_ff = my_ilc.quad_input_optim.calcDesiredInput(my_ilc.kf_dpn.d, np.array(y_des[1:], dtype='float').reshape(-1, 1))

  # collect: dup, x_p, x_b, u_p
  x_b_vec = np.zeros([ILC_it, N_fly+1], dtype='float')
  dup_vec = np.zeros([ILC_it, my_ilc.kf_dpn.d.size], dtype='float')
  x_p_vec = np.zeros([ILC_it+1, N_fly+1], dtype='float')
  u_p_vec = np.zeros([ILC_it, N_fly+1], dtype='float')
  u_des_vec = np.zeros([ILC_it, N_fly], dtype='float')
  u_b0_vec = np.zeros([ILC_it, 1], dtype='float')
  u_d2_vec = np.zeros([ILC_it, 1], dtype='float')
  u_T_fly_vec = np.zeros([ILC_it, 1], dtype='float')
  u_real_dist_vec = np.zeros([ILC_it+1, my_ilc.kf_dpn.d.size], dtype='float')

  # ILC Loop
  y_meas = None
  fly_time_meas = 0
  # disturbance to be learned
  period = 0.02/dt
  disturbance_throw = 100*np.sin(2*np.pi/period*np.arange(N_throw), dtype='float')  # disturbance on the plate position(0:my_ilc.N_throw-1)
  disturbance = 100*np.sin(2*np.pi/period*np.arange(N_fly), dtype='float')  # disturbance on the plate position(0:my_ilc.N_fly-1)
  
  # Visualization
  visual=False
  sim.simulate_one_iteration(dt=dt, T=T_throw, x0=x0_throw, u=uff_throw, d=disturbance_throw, visual=visual, slow=5)
  x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec = sim.simulate_one_iteration(dt=dt, T=T_fly, u=u_ff, d=disturbance, visual=visual, slow=5)
  # plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, x_p_des=y_des)


  for j in range(ILC_it):

    # Extra simulation (the throw)
    x_b_throw, u_b_throw, x_p_throw, u_p_throw, dP_N_throw, gN_throw, F_throw = \
    sim.simulate_one_iteration(dt=dt, T=T_throw, x0=x0_throw, u=uff_throw, d=disturbance_throw)

    # Main Simulation (after-throw)
    x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec = sim.simulate_one_iteration(dt=dt, T=T_fly, u=u_ff, d=disturbance)

    # LEARN THROW
    y_meas = x_p[1:]
    u_ff = my_ilc.learnEmptyStep(u_ff_old=u_ff, y_meas=y_meas, y_des=y_des)

    # 5. Collect data for plotting
    x_b_vec[j, :] = np.squeeze(x_b)
    dup_vec[j, :] = np.squeeze(my_ilc.kf_dpn.d)
    x_p_vec[j, :] = np.squeeze(x_p)
    u_p_vec[j, :] = np.squeeze(u_p)
    u_des_vec[j, :] = np.squeeze(u_ff)
    u_T_fly_vec[j] = fly_time_meas
    print("ITERATION: " + str(j+1)
          + ", Ub0: " + str(0) + ", " + str(u_p[-1])
          + ", Fly time: " + str(fly_time_meas))

  x_b = np.vstack([x_b_throw, x_b])
  u_b = np.vstack([u_b_throw, u_b])
  x_p = np.vstack([x_p_throw, x_p])
  u_p = np.vstack([u_p_throw, u_p])
  dP_N_vec = np.vstack([dP_N_throw, dP_N_vec])
  gN_vec = np.vstack([gN_throw, gN_vec])
  F_vec = np.vstack([F_throw, F_vec])

  # print(y_meas.shape, y_des.shape)

  # sim.simulate_one_iteration(dt=dt, T=T_throw, x0=x0_throw, u=uff_throw, visual=True)
  # sim.simulate_one_iteration(dt=dt, T=T_fly, x0=x000, u=u_ff, d=disturbance, visual=True)
  # %%
  # Plot the stuff
  plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, x_p_des=np.append(x_p_throw, y_des),
                  vertical_lines={T_throw: "T_throw", T_fly: "T_catch"}, 
                  horizontal_lines={x_catch: "Z_catch"})
  
  
  # u_real_dist_vec[-1, :] = np.squeeze(my_ilc.kf_dpn.d)
  # x_p_vec[-1, :] = y_des
  # plotIterations(u_real_dist_vec.T, "measured dist", dt, every_n=3)
  # plotIterations(dup_vec.T, "dup", dt, every_n=3)
  # plotIterations(x_p_vec.T, "x_p", dt, every_n=1)
  # plotIterations(u_b0_vec, "ub0", every_n=3)
  # plotIterations(x_b_vec.T, "x_b", dt, every_n=3)
  # plotIterations(u_T_fly_vec, "T_fly", every_n=3)
  # plotIterations(u_d2_vec, "d2", every_n=3)
  plt.show()
  
  return u_ff

if __name__ == "__main__":
  
  # Init state
  E = 0.25
  tau = 0.53
  dwell_ration = 0.68
  catch_throw_ratio = 0.5
  T_throw, T_hand, ub_catch, ub_throw, T_empty, H,  z_catch = calc(tau, dwell_ration, catch_throw_ratio, E)
  T_fly = 2*T_empty + T_hand
  

  x_ruhe = -0.4
  uff_throw, u_b_throw, x_throw, T_fly, T_throw = learnThrow(T_fly=T_fly, T_throw=T_throw, Hb=H, 
                                                             ub_throw=ub_throw, x_ruhe=x_ruhe, x_catch=z_catch, ILC_it=15, plot=False)
  uff_catch = learnEmpty(uff_throw=uff_throw, ILC_it=1, T_fly=T_fly, T_throw=T_throw, 
                         u_b_throw=u_b_throw, ub_catch=ub_catch,
                         x_throw=x_throw, x_catch=z_catch, x_ruhe=x_ruhe)
