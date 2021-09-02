import numpy as np
from utils import plot_intervals, plt


# MINJERKTRAJECTORY Used to do min-jerk trajectory computation
# Since any free start- or end-state puts a constraint on the constate
# the equations stay the same and only the coefficients change.
# This allows us to call get_trajectories() to create paths of
# different constraints.
def get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=None, a_tb=None):
  # Input:
  #   x_ta, u_ta, (optional: a.ta): conditions at t=ta
  #   x_tb, u_tb, (optional: a.tb): conditions at t=tb
  #   a: is set to [] if start and end acceleration are free
  # Output:
  #   xp_des(t) = [x(t)       u(t)         a(t)            u(t)]
  #             = [position   velocity     acceleration    jerk]

  # Get polynom parameters for different conditions
  T = tb-ta
  if a_ta:
    # 1. set start acceleration
    if a_tb:
      # a. set end acceleration
      c1, c2, c3, c4, c5, c6 = set_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)
    else:
      # b.free end acceleration
      c1, c2, c3, c4, c5, c6 = set_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a_ta)
  else:
    # 2. free start acceleration
    if a_tb:
      # a. set end acceleration
      c1, c2, c3, c4, c5, c6 = free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a_tb)
    else:
      # b.free end acceleration
      c1, c2, c3, c4, c5, c6 = free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb)

  # Trajectory values ta->tb
  t = np.arange(0, T, dt)  # 0:dt:T
  j, a, v, x = get_trajectories(t, c1, c2, c3, c4, c5, c6)
  return x, v, a, j


# Get  values from polynom parameters
def get_trajectories(t, c1, c2, c3, c4, c5, c6):
  # print(c1)
  t_5 = t**5
  t_4 = t**4
  t_3 = t**3
  t_2 = t**2
  j = c1*t_2/2   - c2*t       + c3                              # jerk
  a = c1*t_3/6   - c2*t_2/2  + c3*t     + c4                    # acceleration
  v = c1*t_4/24  - c2*t_3/6  + c3*t_2/2 + c4*t      + c5        # velocity
  x = c1*t_5/120 - c2*t_4/24 + c3*t_3/6 + c4*t_2/2 + c5*t + c6  # position
  return [j, a, v, x]


# 1) Acceleration is set at t=0 (a(0)=a0 => c4=a0)
def set_start_acceleration(T, x0, xT, u0, uT, a0=None, aT=None):
  T_5 = T**5
  T_4 = T**4
  T_3 = T**3
  T_2 = T**2
  if aT is None:
      # free end acceleration u(T)=0
      M = np.array([[320/T_5, -120/T_4, -20/(3*T_2)],
                    [200/T_4, -72/T_3, -8/(3*T)],
                    [40/T_3, -12/T_2, -1/3]])
      c = np.array([-(a0*T_2)/2 - u0*T - x0 + xT, uT - u0 - T*a0, 0])
  else:
      # set end acceleration a(T)=aT
      M = np.array([[720/T_5, -360/T_4, 60/T_3],
                    [360/T_4, -168/T_3, 24/T_2],
                    [60/T_3, -24/T_2, 3/T]])
      c = np.array([-(a0*T_2)/2 - u0*T - x0 + xT, uT - u0 - T*a0, aT - a0])

  c123 = M.dot(c.T)
  c1 = c123[0]
  c2 = c123[1]
  c3 = c123[2]
  c4 = a0
  c5 = u0
  c6 = x0
  return c1, c2, c3, c4, c5, c6


# 2) Acceleration is free at t=0 (u(0)=0 => c3=0)
def free_start_acceleration(T, x0, xT, u0, uT, aT=None):
  T_5 = T**5
  T_4 = T**4
  T_3 = T**3
  T_2 = T**2
  if aT is None:
      # free end acceleration u(T)=0
      M = np.array([[120/T_5, -60/T_4, -5/T_2],
                    [60/T_4, -30/T_3, -3/(2*T)],
                    [5/T_2, -3/(2*T), -T/24]])
      c = np.array([xT - x0 - T*u0, uT - u0, 0])
  else:
      # set end acceleration a(T)=aT
      M = np.array([[320/T_5, -200/T_4, 40/T_3],
                    [120/T_4, -72/T_3, 12/T_2],
                    [20/(3*T_2), -8/(3*T), 1/3]])
      c = np.array([xT - x0 - T*u0, uT - u0, aT])

  c123 = M.dot(c.T)
  c1 = c123[0]
  c2 = c123[1]
  c4 = c123[2]
  c3 = 0
  c5 = u0
  c6 = x0
  return c1, c2, c3, c4, c5, c6


def plotMinJerkTraj(x, v, a, j, dt, title, intervals=None, colors=None):
  """Plots the x,v,a,j trajectories together with possible intervals and colors

  Args:
      x ([List(double)]): position vector
      v ([List(double)]): velocity vector
      a ([List(double)]): acceleration vector
      j ([List(double)]): jerk vector
      dt ([double]): time step
      title ([String]): tittle of the plot
      intervals ([set((a,b))], optional): {(0.1, 0.2), (0.42,0.55), ..}
      colors ([tuple], optional): ('gray', 'blue', ..)
  """
  if colors is None:
    colors = []
  fig, axs = plt.subplots(4, 1)
  timesteps = np.arange(0, x.size) * dt  # (1:length(x))*dt

  axs[0].plot(timesteps, x, 'b', label='Plate position')
  axs[1].plot(timesteps, v, 'b', label='Plate velocity')
  axs[2].plot(timesteps, a, 'b', label='Plate acceleration')
  axs[3].plot(timesteps, j, 'b', label='Plate jerk')
  if intervals is not None:
    for ax in axs:
      ax.legend(loc=1)
      ax = plot_intervals(ax, intervals, dt, colors)
  fig.suptitle(title)
  plt.show()


def main():
  dt = 0.002
  ta = 0; tb = 0.9032/2
  x_ta=-0.425; x_tb=0; u_ta=0; u_tb=4.4287; a_ta=None; a_tb=None

  x1, v1, a1, j1 = get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb)
  # plotMinJerkTraj(x1, v1, a1, j1, dt, 'probe')

  tb2=tb+0.9032
  x_ta=0; x_tb=0; u_ta=4.4287; u_tb=-u_ta/99
  a_ta = a1[-1]
  x2, v2, a2, j2 = get_min_jerk_trajectory(dt, tb, tb2, x_ta, x_tb, u_ta, u_tb, a_ta)

  tb3=tb2+0.9032/2
  x_ta=0; x_tb=-0.4
  u_ta=u_tb; u_tb=0
  a_ta = a2[-1]
  a_tb = 0
  x3, v3, a3, j3 = get_min_jerk_trajectory(dt, tb2, tb3, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)

  intervals = {(ta/dt, tb/dt), (tb/dt, tb2/dt), (tb2/dt, tb3/dt)}
  colors = ('gray', 'gray', 'red')
  plotMinJerkTraj(np.concatenate((x1, x2, x3)),
                  np.concatenate((v1, v2, v3)),
                  np.concatenate((a1, a2, a3)),
                  np.concatenate((j1, j2, j3)), dt, "Plate Free Motion", intervals, colors)


if __name__ == "__main__":
  main()
