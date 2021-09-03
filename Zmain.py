#!/usr/bin/env python2
from juggling_apollo.Simulation import Simulation

# Constants
g = 9.80665     # [m/s^2]
dt = 0.004      # [s] discretization time step size

# Params
m_b = 0.1       # [kg]
m_p = 10        # [kg]
k_c = 10        # [1/s]  time-constant of velocity controller


def main():
  print("Py2")
  input_is_force = False
  air_drag = True

  h_b_max = 1;                                        % [m] maximal height the ball achievs
    [Tb, ~] = plan_ball_trajectory(h_b_max, 0, 0);      % [s] flying time of the ball
    Tsim= Tb*2*5;                                       % [s] simulation time
    N = Simulation.steps_from_time(Tsim, dt);           % number of steps for one iteration (maybe use floor)

  sim = Simulation(m_b, m_p, k_c, g, input_is_force, air_drag)
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = \
      sim.simulate_one_iteration(dt, Tsim, x_b0, x_p0, u_b0, u_p0, F_p, 1, dist)


if __name__ == "__main__":
    main()
