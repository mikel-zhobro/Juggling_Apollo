import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from settings import g, ABS
import numpy as np
from matplotlib.transforms import Bbox

# mpl.use('TkAgg')
np.set_printoptions(precision=3, suppress=True)

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def plan_ball_trajectory(hb, d1=0, d2=0):
    ub_0 = np.sqrt(2*g*(hb - d1))  # velocity of ball at throw point
    Tb = 2*ub_0/g + d2  # flying time of the ball
    return Tb, ub_0


def flyTime2HeightAndVelocity(Tfly):
  ub_0 = g*Tfly/2
  Hb = ub_0**2 /(2*g)
  return Hb, ub_0


def time_from_step(N, dt):
  return rtime((N-1.0)*dt, dt)

def steps_from_time(T, dt):
    """ Method to find nr of timesteps

    Args:
        T (double): Time in seconds
        dt (double): Timestep in seconds

    Returns:
        int: Number of timesteps in T
    """
    # assert T % dt < 1e-8
    return int(T/dt) + 1  # np.arange(0,T+dt,dt)

def rtime(T, dt):
    """ Method to find nr of timesteps

    Args:
        T (double): Time in seconds
        dt (double): Timestep in seconds

    Returns:
        int: Number of timesteps in T
    """
    return dt*(T // dt) + 1e-10  # np.arange(0,T,dt)

def find_continuous_intervals(gN_vec):
    """ Find connected intervals of values
    TODO: vectorize for multiple ball

    Args:
        indices ([int]): List of indexes of true intervals [timesteps x N]

    Returns:
        set: A 2xN set where N is nr of intevals found and 2 represents start and end indexes.
    """
    if len(gN_vec.shape) > 1:
      temp = np.logical_or.reduce(gN_vec <= ABS, axis=1)
    else:
      temp = gN_vec <= ABS

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

def plot_lines_coord(ax, tt, xx, typ=None):

  einheit = ['m', r'$\frac{m}{s}$', r'$\frac{m}{s^2}$']
  ei = einheit[typ] if typ is not None else ''
  assert len(tt) == len(xx)
  xmin, xmax, ymin, ymax = ax.axis()
  # Draw lines connecting points to axes
  ax.scatter(tt, xx)
  for t, x in zip(tt,xx):
    txt = r'({}s, {} {})'.format(t, x, ei)
    ax.text(t, x, txt, size=9, color='k', zorder=10, weight='normal')

#   for t, x in zip(tt, xx):
#     ax.plot([t, t], [ymin, x], ls='--', lw=1.5, alpha=0.5)
#     ax.plot([xmin, t], [x, x], ls='--', lw=1.5, alpha=0.5)

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])