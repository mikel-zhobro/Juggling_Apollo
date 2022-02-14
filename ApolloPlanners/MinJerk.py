# MINJERKTRAJECTORY Used to do min-jerk trajectory computation
# Since any free start- or end-state puts a constraint on the constate
# the equations stay the same and only the coefficients change.
# This allows us to call get_trajectories() to create paths of
# different constraints.

import numpy as np
import matplotlib.pyplot as plt

from utils import plot_intervals, steps_from_time, plot_lines_coord


##### Helper Functions for multi-interval multi-dimensional minjerk

### 1.Straight forward approach
def get_multi_interval_minjerk_xyz(dt, tt, xx, uu, smooth_acc=False, smooth_start=False, i_a_end=None, only_pos=False):
  """ Generates a multi-interval multi-dimensional minjerk trajectory

  Args:
      dt ([np.array]): (nr_intervals, Dim)
      tt ([np.array]): (nr_intervals, Dim)
      xx ([np.array]): (nr_intervals, Dim)
      uu ([np.array]): (nr_intervals, Dim)
      smooth_acc (bool, optional): Whether the acceleartion between intervals should be smooth.
      i_a_end ([type], optional): If not None shows the number of the interval, whose end-acceleration should be used for the last interval.

  Returns:
      xxx, vvv, aaa, jjj: np.arrays of size (nr_intervals, Dim)
  """
  N = xx.shape[1]
  if only_pos:
    xxx = np.array([get_multi_interval_minjerk_1D(dt, tt, xx[:,i], uu[:,i],
                                                  smooth_acc=smooth_acc, smooth_start=smooth_start,
                                                  i_a_end=i_a_end, only_x=True)
                    for i in range(N)]).T
    return xxx
  else:
    xxx = [None] * N
    vvv = [None] * N
    aaa = [None] * N
    jjj = [None] * N
    for i in range(N):
      xxx[i],vvv[i],aaa[i],jjj[i] = get_multi_interval_minjerk_1D(dt, tt, xx[:,i], uu[:,i],
                                                                  smooth_acc=smooth_acc, smooth_start=smooth_start,
                                                                  i_a_end=i_a_end, only_x=False)
    return np.array(xxx).T[:,:,np.newaxis], np.array(vvv).T[:,:,np.newaxis], np.array(aaa).T[:,:,np.newaxis], np.array(jjj).T[:,:,np.newaxis]

def get_multi_interval_minjerk_1D(dt, tt, xx, uu, smooth_acc=False, smooth_start=False, i_a_end=None, only_x=False, extra_at_end=None):
  """Generates a multi-interval minjerk trajectory in 1 dimension

  Args:
      dt ([float]):
      tt ([np.array]): (nr_intervals, )
      xx ([np.array]): (nr_intervals, )
      uu ([np.array]): (nr_intervals, )
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
  a_last = None if not smooth_start else 0.0
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


### 2.Functional approach
def get_multi_interval_multi_dim_minjerk(dt, tt, xx, uu, smooth_acc=False, smooth_start=False, i_a_end=None, only_pos=True, lambdas=False):
  """ Generates a multi-interval multi-dimensional minjerk trajectory
      The advantage of this function is that it is able to return conditional functions which can then be evaluated
      at arbitrary times or time intervals.

    Args:
      dt ([list]): float
      tt ([np.array]): (nr_intervals,)
      xx ([np.array]): (nr_intervals, Dim)
      uu ([np.array]): (nr_intervals, Dim)
      lambdas (bool, optional): Whether to return lambdas . Defaults to False.

      [np.array]: (4, N_times, Dims) where 4 are position, velocity, acceleration and jerk
      or
      [lambda]:  tmp(t) -> np.array(4, N_times, Dims) where 4 are x(t), v(t), a(t) and j(t)

  """
  D = xx.shape[1]
  N = len(tt)  # nr of minjerk intervals
  xxs = np.array([None]* N)  # (N, )

  a_last = [None if not smooth_start else 0.0]*D
  a_ende = [None]*D
  a_end = None
  for i, (tb, xb, ub) in enumerate(zip(tt[1:], xx[1:], uu[1:])):
    xxs[i] = get_multi_dim_minjerk(dt, tt[i], tb, xx[i], xb, uu[i], ub, a_ta=a_last, a_tb=a_ende, lambdas=True)
    # smooth_acc
    a_last = xxs[i](tb)[2].squeeze() if smooth_acc else [None]*D
    # i_a_end
    if i_a_end is not None and i == i_a_end:
      a_end = xxs[i](tt[i])[2].squeeze()
    if i == N-3:
      a_ende = a_end

  def tmp(t):
    t = np.asarray(t).reshape(-1,1)
    ret = np.zeros((4, len(t), D))

    for i, _t in enumerate(tt[1:]):
      mask = (tt[i] <= t) & ( (t <= _t) if i==len(tt)-2 else (t < _t))
      ret[:, mask.nonzero()[0], :] = xxs[i](t[mask])
    return ret[:,:,:,np.newaxis]

  if lambdas:
    return tmp
  else:
    ttt = np.linspace(tt[0], tt[-1], 1 + int((tt[-1]-tt[0])/dt))
    return tmp(ttt)

def get_multi_dim_minjerk(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=None, a_tb=None, lambdas=False):
  """Generates a multi-dimensional minjerk trajectory

  Args:
      dt ([float]):
      ta, tb ([float]): start and end time
      x_ta, x_tb ([np.array]): (Dim, ), start and end position
      u_ta, u_tb ([np.array]): (Dim, ), start and end velocity
      a_ta, a_tb ([np.array]): (Dim, ), start and end acceleration. Defaults to None.
      lambdas (bool, optional): Whether to return lambdas . Defaults to False.

  Returns:
      [np.array]: (4, N_times, Dims) where 4 are position, velocity, acceleration and jerk
      or
      [lambda]:  tmp(t) -> np.array(4, N_times, Dims) where 4 are x(t), v(t), a(t) and j(t)
  """
  a_ta = [None] * len(x_ta) if a_ta is None else a_ta
  a_tb = [None] * len(x_ta) if a_tb is None else a_tb
  xxs = [get_min_jerk_trajectory(dt, ta, tb, x_ta[i], x_tb[i], u_ta[i], u_tb[i], a_ta=a_ta[i], a_tb=a_tb[i], lambdas=True)
         for i in range(len(x_ta))]

  def tmp(t):
    t = np.asarray(t).reshape(-1,1)
    ret = np.zeros((4, len(t), len(xxs)))
    for i, xx in enumerate(xxs):
      ret[:, :, i:i+1] = xx(t)
    return ret

  if lambdas:
    return tmp
  else:
    ttt = np.linspace(ta, tb, 1 + int((tb-ta)/dt))
    return tmp(ttt)



##### Main Functions

def get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=None, a_tb=None, lambdas=False):
  """Generates a minjerk trajectory with set or free start and end conditions.

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
    N_Whole = steps_from_time(T, dt)
    t = np.linspace(ta, tb, N_Whole)
    # t = np.arange(ta, tb, dt)  # 0:dt:T ceil((stop - start)/step)
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



##### Plotting functions

def plotMinJerkTraj(x, v, a, j, dt, title, intervals=None, colors=None, tt=None, xx=None, uu=None, block=True):
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
  fsize = 12
  if colors is None:
    colors = []
  fig, axs = plt.subplots(4, 1, figsize=(16, 11))
  timesteps = np.arange(0, x.size) * dt  # (1:length(x))*dt
  for ax in axs:
    ax.grid(True)
    ax.set_xlim(xmin=0,xmax=timesteps[-1])
  axs[0].plot(timesteps, x, 'b', label='Position')
  axs[0].legend(loc=1, fontsize=fsize)
  axs[1].plot(timesteps, v, 'b', label='Velocity')
  axs[1].legend(loc=1, fontsize=fsize)
  axs[2].plot(timesteps, a, 'b', label='Acceleration')
  axs[2].legend(loc=1, fontsize=fsize)
  axs[3].plot(timesteps, j, 'b', label='Jerk')
  axs[3].legend(loc=1, fontsize=fsize)
  if intervals is not None:
    for ax in axs:
      ax = plot_intervals(ax, intervals, dt, colors)

  if tt is not None:
    if xx is not None:
      plot_lines_coord(axs[0], tt, xx, typ=None)
    else:
      for t in tt:
        axs[0].axvline(t, linestyle='--')
    if uu is not None:
      plot_lines_coord(axs[1], tt, uu, typ=None)
    else:
      for t in tt:
        axs[1].axvline(t, linestyle='--')
    for ax in axs[2:]:
      for t in tt:
        ax.axvline(t, linestyle='--')
  fig.suptitle(title)
  plt.show(block = block)


def plotMJ(dt, tt, xx, uu, smooth_acc=False, xvaj = None, i_a_end=0):
  print(
  "\n X: " +str(xx) +
  "\n T: " +str(tt) +
  "\n U: " +str(uu)
      )
  title = "Min-Jerk trajectory with " +  ("" if smooth_acc else "non") +"-smoothed acceleration."
  if xvaj is None:
    x, v, a, j = get_multi_interval_minjerk_1D(dt, tt=tt, xx=xx, uu=uu, smooth_acc=smooth_acc, i_a_end=i_a_end)
  else:
    x, v, a, j = xvaj
  plotMinJerkTraj(x, v, a, j, dt, title, tt=tt[0:4], xx=xx[0:4], uu=uu[0:4])


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  dt = 0.004
  smooth_acc = False
  smooth_start = False
  i_a_end = -1
  tt=[ 0.0,    0.5,         0.6,             0.8,              1.0 ]
  xx=np.array([[0.0,    0.0,        -0.1,             0.3,              1.0 ],
               [0.0,    0.6,         0.0,             0.2,              0.0 ]]).T

  uu=np.array([[0.0,    3.4,    -0.2,          1.1,         0.0],
               [0.0,    2.2,    0.0,           2.0,         0.0]]).T

  # 1. straight forward
  xxx, vvv, aaa, jjj = get_multi_interval_minjerk_xyz(dt, tt, xx, uu, smooth_acc, smooth_start, only_pos=False, i_a_end=i_a_end)

  # 2. functional approach values
  xxx1, vvv1, aaa1, jjj1 = get_multi_interval_multi_dim_minjerk(dt, tt, xx, uu, smooth_acc, smooth_start, i_a_end=i_a_end)

  # 3. functional approach lambdas
  ttt = np.linspace(tt[0], tt[-1], 1 + int((tt[-1]-tt[0])/dt))
  func = get_multi_interval_multi_dim_minjerk(dt, tt, xx, uu, smooth_acc, smooth_start, i_a_end=i_a_end, lambdas=True)
  xxx2, vvv2, aaa2, jjj2 =  func(ttt)

  print(np.linalg.norm(xxx2 - xxx1)/xxx1.size)
  print(np.linalg.norm(xxx - xxx1)/xxx.size)
  print(np.linalg.norm(vvv - vvv1)/vvv.size)
  print(np.linalg.norm(aaa - aaa1)/vvv.size)
  print(np.linalg.norm(jjj - jjj1)/vvv.size)


  plt.plot(ttt, aaa[:,0], 'b')
  plt.plot(ttt, aaa1[:,0], 'r')
  plt.title('Differences: straight forward vs functional approach')
  plotMinJerkTraj(xxx[:,0], vvv[:,0], aaa[:,0], jjj[:,0], dt, "straight forward approach", block=False, tt=tt, xx=xx[:,0], uu=uu[:,0])
  plotMinJerkTraj(xxx1[:,0], vvv1[:,0], aaa1[:,0], jjj1[:,0], dt, "functional approach values", block=False, tt=tt, xx=xx[:,0], uu=uu[:,0])
  plotMinJerkTraj(xxx1[:,0], vvv2[:,0], aaa2[:,0], jjj2[:,0], dt, "functional approach lambdas", block=False, tt=tt, xx=xx[:,0], uu=uu[:,0])
  plt.savefig('MJ_proposal.pdf')
  plt.show()
