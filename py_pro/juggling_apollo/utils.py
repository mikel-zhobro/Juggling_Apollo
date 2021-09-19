import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from settings import g
import numpy as np
# mpl.use('TkAgg')


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def plan_ball_trajectory(hb, d1=0, d2=0):
    ub_0 = np.sqrt(2*g*(hb - d1))  # velocity of ball at throw point
    Tb = 2*ub_0/g + d2  # flying time of the ball
    return Tb, ub_0


def flyTime2HeightAndVelocity(Tfly):
  ub_0 = g*Tfly/2
  Hb = ub_0**2 /(2*g)
  return Hb, ub_0


def steps_from_time(T, dt):
    """ Method to find nr of timesteps

    Args:
        T (double): Time in seconds
        dt (double): Timestep in seconds

    Returns:
        int: Number of timesteps in T
    """
    return int(np.ceil(T / dt))  # np.arange(0,T,dt)


def find_continuous_intervals(gN_vec):
    """ Find connected intervals of values
    TODO: vectorize for multiple ball

    Args:
        indices ([int]): List of indexes of true intervals [timesteps x N]

    Returns:
        set: A 2xN set where N is nr of intevals found and 2 represents start and end indexes.
    """
    if len(gN_vec.shape) > 1:
      temp = np.logical_or.reduce(gN_vec <= 1e-5, axis=1)
    else:
      temp = gN_vec <= 1e-5

    indices = 1 + np.argwhere(temp)

    intervals = tuple()
    indices = np.squeeze(indices)
    if indices.size != 0:
        # Find intervals where gN<=0
        last = indices[0]
        start = last

        for i in indices:
            if i - last > 1 or i == indices[-1]:
                intervals = intervals + ([start - 1, last],)
                start = i
            last = i
    return intervals


def plot_intervals(ax, intervals, dt, colors=None):
    if colors is None or len(colors) != len(intervals):
        colors = np.repeat('gray', len(intervals))  # #2ca02c
    for i, col in zip(intervals, colors):
        ax.axvspan(dt * i[0], dt * i[1], facecolor=col, alpha=0.3)
    return ax


def plotIterations(y, title, dt=1, every_n=1):

  f = plt.figure()
  n_x = y.shape[0]
  n_i = y.shape[1]
  timesteps = np.arange(n_x)*dt
  for i in np.arange(0, n_i, every_n):
    label = "iteration "+str(i+1) if n_i>1 else ""
    plt.plot(timesteps, y[:, i], label=label)
  plt.legend()
  plt.title(title)
  plt.xlabel("ITERATIONS")
  plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.show(block=False)

def plot_lines_coord(ax, tt, xx):
  assert len(tt) == len(xx)
  xmin, xmax, ymin, ymax = ax.axis()
  # Draw lines connecting points to axes
  ax.scatter(tt, xx)
  for t, x in zip(tt, xx):
    ax.plot([t, t], [ymin, x], ls='--', lw=1.5, alpha=0.5)
    ax.plot([xmin, t], [x, x], ls='--', lw=1.5, alpha=0.5)