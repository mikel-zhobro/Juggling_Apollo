#!/usr/bin/env python2
import __add_path__
import numpy as np
from ApolloILC.utils import plan_ball_trajectory, steps_from_time
from ApolloILC.settings import dt, m_p
from Simulation import SimulationPaddleBall
from Simulation.SimulationPaddleBall import Simulation, plot_simulation


def test1():
  input_is_force = False
  air_drag = True

  # Design params
  h_b_max = 1                                        # [m] maximal height the ball achievs
  Tb, _ = plan_ball_trajectory(h_b_max, 0, 0)        # [s] flying time of the ball
  Tsim= Tb*2*5                                       # [s] simulation time
  N = steps_from_time(Tsim, dt)           # number of steps for one iteration (maybe use floor)

  # Initial Conditions
  x_b0 = 0       # [m]   starting ball position
  x_p0 = 0       # [m]   starting plate position
  u_p0 = 0       # [m/s] starting plate velocity
  u_b0 = u_p0    # [m/s] starting ball velocity
  x0 = [x_b0, x_p0, u_b0, u_p0]
  # %% Simulation Example
  # Input
  A = 0.3                                            # [m] amplitude
  timesteps = dt * np.arange(N)                             # [s,s,..] timesteps
  F_p = 100 * m_p * A*np.sin(np.pi/Tb *timesteps)          # [N] input force on the plate
  dist = 0.1 * np.sin(timesteps)
  input_is_force = True

  # Simulation for N steps
  sim = Simulation(input_is_force, air_drag, x0)
  [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt, Tsim, F_p, x0, repetitions=1, d=dist)

  # Plotting for Simulation Example
  plot_simulation(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)


def test_all():
  test1()


if __name__ == "__main__":
  test_all()
