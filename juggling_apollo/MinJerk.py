# MINJERKTRAJECTORY Used to do min-jerk trajectory computation
# Since any free start- or end-state puts a constraint on the constate
# the equations stay the same and only the coefficients change.
# This allows us to call get_trajectories() to create paths of
# different constraints.

import numpy as np
from utils import plot_intervals, plt, steps_from_time, plot_lines_coord


def get_minjerk_xyz(dt, tt, xx, uu, smooth_acc=False, i_a_end=None, only_pos=True):
  """Computes a multi-interval minjerk trajectory in 3 dimension(xyz)

  Args:
      dt ([list]): [[double] x nr_intervals] x 3
      tt ([list]): [[double] x nr_intervals] x 3
      xx ([list]): [[double] x nr_intervals] x 3
      uu ([list]): [[double] x nr_intervals] x 3
      smooth_acc (bool, optional): Whether the acceleartion between intervals should be smooth.
      i_a_end ([type], optional): If not None shows the number of the interval, whose end-acceleration should be used for the last interval.

  Returns:
      [lists]: xx returns only a list of position trajectories(not velocities and acceleration)
  """
  if only_pos:
    xxx = [get_minjerk_trajectory(dt, tt, xx[i], uu[i], smooth_acc=smooth_acc, i_a_end=i_a_end, only_x=True) for i in range(len(xx))]
    return xxx
  else:
    xxx = [None] * len(xx)
    vvv = [None] * len(xx)
    aaa = [None] * len(xx)
    jjj = [None] * len(xx)
    for i in range(len(xx)):
      xxx[i],vvv[i],aaa[i],jjj[i] = get_minjerk_trajectory(dt, tt, xx[i], uu[i], smooth_acc=smooth_acc, i_a_end=i_a_end, only_x=False)
    return xxx, vvv, aaa, jjj


def get_minjerk_trajectory(dt, tt, xx, uu, smooth_acc=False, i_a_end=None, only_x=False, extra_at_end=None):
  """Computes a multi-interval minjerk trajectory in 1 dimension

  Args:
      dt ([list]): [double] x nr_intervals
      tt ([list]): [double] x nr_intervals
      xx ([list]): [double] x nr_intervals
      uu ([list]): [double] x nr_intervals
      smooth_acc (bool, optional): Whether the acceleartion between intervals should be smooth.
      i_a_end ([type], optional): If not None shows the number of the interval, whose start-acceleration should be used for the last interval.
      extra_at_end([type], optional): If not None shows the number of times the last value of position should be repeated

  Returns:
      [lists]: x, v, a, j
  """
  # Initialization
  T_whole = tt[-1] - tt[0]
  N_Whole = steps_from_time(T_whole, dt)
  x_ret = np.zeros(N_Whole, dtype='double')
  v_ret = np.zeros(N_Whole, dtype='double')
  a_ret = np.zeros(N_Whole, dtype='double')
  j_ret = np.zeros(N_Whole, dtype='double')

  N = len(tt)

  t_last = tt[0]  # last end-time
  x_last = xx[0]
  u_last = uu[0]
  a_last = None
  a_ende = None
  a_end = None
  n_last = 0  # last end-index
  for i in range(N-1):
    t0 = t_last; t1 = tt[i+1]
    x0 = x_last; x1 = xx[i+1]
    u0 = u_last; u1 = uu[i+1]
    x, v, a, j = get_min_jerk_trajectory(dt, t0, t1, x0, x1, u0, u1, a_ta=a_last, a_tb=a_ende)

    len_x = len(x)
    x_ret[n_last: n_last+len_x] = x
    v_ret[n_last: n_last+len_x] = v
    a_ret[n_last: n_last+len_x] = a
    j_ret[n_last: n_last+len_x] = j

    t_last = t0 + (len_x-1)*dt
    n_last += len_x-1  # (since last end-value == new first-value we overlay them and take only 1)
    x_last = x[-1]
    u_last = v[-1]
    a_last = a[-1] if smooth_acc else None
    if i_a_end is not None and i == i_a_end:
      a_end = a[0]
    if i==N-3:
      a_ende = a_end
  x_ret[n_last:] = x[-1]
  v_ret[n_last:] = v[-1]
  a_ret[n_last:] = a[-1]
  j_ret[n_last:] = j[-1]

  if extra_at_end is not None:
    repeat = [1]* N_Whole
    repeat[-1] = extra_at_end
    x_ret = np.hstack((x_ret, x_ret[1:extra_at_end]))
    v_ret = np.hstack((v_ret, v_ret[1:extra_at_end]))
    a_ret = np.hstack((a_ret, a_ret[1:extra_at_end]))
    j_ret = np.hstack((j_ret, j_ret[1:extra_at_end]))
  if only_x:
    return x_ret
  else:
    return x_ret, v_ret, a_ret, j_ret

def get_min_jerk_xyz(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=[None]*3, a_tb=[None]*3, lambdas=False):
  xx = get_min_jerk_trajectory(dt, ta, tb, x_ta[0], x_tb[0], u_ta[0], u_tb[0], a_ta=a_ta[0], a_tb=a_tb[0], lambdas=lambdas)
  yy = get_min_jerk_trajectory(dt, ta, tb, x_ta[1], x_tb[1], u_ta[1], u_tb[1], a_ta=a_ta[1], a_tb=a_tb[1], lambdas=lambdas)
  zz = get_min_jerk_trajectory(dt, ta, tb, x_ta[2], x_tb[2], u_ta[2], u_tb[2], a_ta=a_ta[2], a_tb=a_tb[2], lambdas=lambdas)
  def tmp(t):
    x, vx, ax, jx = xx(t)
    y, vy, ay, jy = yy(t)
    z, vz, az, jz = zz(t)
    xxx = np.vstack((x,y,z)).T
    vvv = np.vstack((vx,vy,vz)).T
    aaa = np.vstack((ax,ay,az)).T
    jjj = np.vstack((jx,jy,jz)).T
    return xxx, vvv, aaa, jjj
  if lambdas:
    return tmp
  else:
    xxx = np.vstack((xx[0],yy[0],zz[0])).T
    vvv = np.vstack((xx[1],yy[1],zz[1])).T
    aaa = np.vstack((xx[2],yy[2],zz[2])).T
    jjj = np.vstack((xx[3],yy[3],zz[3])).T
    return xxx, vvv, aaa, jjj


def get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=None, a_tb=None, lambdas=False):
  """Computes a minjerk trajectory with set or free start and end conditions.

  Args:
      dt ([float]): timestep
      ta ([float]): start time of the interval
      tb ([float]): end time of the interval
      a: is set to [] if start and end acceleration are free
      x_ta, u_ta, (optional: a_ta): conditions at t=ta
      x_tb, u_tb, (optional: a_tb): conditions at t=tb
  Returns:
      xp_des(t) = [x(t)       u(t)         a(t)            u(t)]
                = [position   velocity     acceleration    jerk]
  """
  # Get polynom parameters for different conditions
  T = tb-ta
  if a_ta is not None:
    # 1. set start acceleration
    if a_tb is not None:
      # a. set end acceleration
      c1, c2, c3, c4, c5, c6 = set_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)
    else:
      # b.free end acceleration
      c1, c2, c3, c4, c5, c6 = set_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a_ta)
  else:
    # 2. free start acceleration
    if a_tb is not None:
      # a. set end acceleration
      c1, c2, c3, c4, c5, c6 = free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a_tb)
    else:
      # b.free end acceleration
      c1, c2, c3, c4, c5, c6 = free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb)

  if lambdas:
    return lambda t: get_trajectories(t-ta, c1, c2, c3, c4, c5, c6)
  else:
    # Trajectory values ta->tb
    # N_Whole = steps_from_time(T, dt)
    # t = np.linspace(0, T, N_Whole)
    t = np.arange(ta, tb, dt)  # 0:dt:T ceil((stop - start)/step)
    x, v, a, j = get_trajectories(t-ta, c1, c2, c3, c4, c5, c6)
  return x, v, a, j


# Get  values from polynom parameters
def get_trajectories(t, c1, c2, c3, c4, c5, c6):
  """Given 5th order polynomial coeficients it returns values corresponing to timesteps t.
  """
  t_5 = t**5
  t_4 = t**4
  t_3 = t**3
  t_2 = t**2
  j = c1*t_2/2   - c2*t      + c3                               # jerk
  a = c1*t_3/6   - c2*t_2/2  + c3*t     + c4                    # acceleration
  v = c1*t_4/24  - c2*t_3/6  + c3*t_2/2 + c4*t      + c5        # velocity
  x = c1*t_5/120 - c2*t_4/24 + c3*t_3/6 + c4*t_2/2 + c5*t + c6  # position
  return x, v, a, j


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
                    [40/T_3, -12/T_2, -1.0/3.0]])
      c = np.array([-(a0*T_2)/2 - u0*T - x0 + xT, uT - u0 - T*a0, 0])
  else:
      # set end acceleration a(T)=aT
      M = np.array([[720/T_5, -360/T_4, 60/T_3],
                    [360/T_4, -168/T_3, 24/T_2],
                    [60/T_3, -24/T_2, 3/T]])
      c = np.array([xT - x0 - T*u0 - (a0*T_2)/2, uT - u0 - T*a0, aT - a0])

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
                    [20/(3*T_2), -8/(3*T), 1.0/3.0]])
      c = np.array([xT - x0 - T*u0, uT - u0, aT])

  c123 = M.dot(c.T)
  c1 = c123[0]
  c2 = c123[1]
  c4 = c123[2]
  c3 = 0
  c5 = u0
  c6 = x0
  return c1, c2, c3, c4, c5, c6


def plotMinJerkTraj(x, v, a, j, dt, title, intervals=None, colors=None, tt=None, xx=None, uu=None):
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
  for ax in axs:
    ax.set_xlim(xmin=0,xmax=timesteps[-1])
  axs[0].plot(timesteps, x, 'b', label='Plate position')
  axs[0].legend(loc=1)
  axs[1].plot(timesteps, v, 'b', label='Plate velocity')
  axs[1].legend(loc=1)
  axs[2].plot(timesteps, a, 'b', label='Plate acceleration')
  axs[2].legend(loc=1)
  axs[3].plot(timesteps, j, 'b', label='Plate jerk')
  axs[3].legend(loc=1)
  if intervals is not None:
    for ax in axs:
      ax = plot_intervals(ax, intervals, dt, colors)

  if tt is not None:
    if xx is not None:
      plot_lines_coord(axs[0], tt, xx)
    else:
      for t in tt:
        ax[0].axvline(t, linestyle='--')
    if uu is not None:
      plot_lines_coord(axs[1], tt, uu)
    else:
      for t in tt:
        ax[1].axvline(t, linestyle='--')
    for ax in axs[2:]:
      for t in tt:
        ax.axvline(t, linestyle='--')
  fig.suptitle(title)
  plt.show(block = True)


def plotMJ(dt, tt, xx, uu, smooth_acc=False, xvaj = None, i_a_end=0):
  print(
  "\n X: " +str(xx) +
  "\n T: " +str(tt) +
  "\n U: " +str(uu)
      )
  title = "Min-Jerk trajectory with " +  ("" if smooth_acc else "non") +"-smoothed acceleration."
  if xvaj is None:
    x, v, a, j = get_minjerk_trajectory(dt, tt=tt, xx=xx, uu=uu, smooth_acc=smooth_acc, i_a_end=i_a_end)
  else:
    x, v, a, j = xvaj
  plotMinJerkTraj(x, v, a, j, dt, title, tt=tt[0:4], xx=xx[0:4], uu=uu[0:4])


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  dt = 0.004
  smooth_acc = True
  tt=[ 0.0,    0.2,         0.6,             0.8,              1.0 ]
  xx=[[0.0,    0.2,        -0.1,             0.3,              1.0 ],
      [0.0,    0.6,         0.0,             0.2,              0.0 ]]

  uu=[[0.0,    3.4,    -0.2,          1.1,         0.0],
      [0.0,    2.2,    0.0,           2.0,         0.0]]

  xxx, vvv, aaa, jjj = get_minjerk_xyz(dt, tt, xx, uu, smooth_acc, only_pos=False)

  plotMinJerkTraj(xxx[0], vvv[0], aaa[0], jjj[0], dt, "Free acceleartion")
  plt.show()
