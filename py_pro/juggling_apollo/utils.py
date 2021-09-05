import matplotlib as mpl
import matplotlib.pyplot as plt
from .settings import g
import numpy as np
mpl.use('TkAgg')


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def plan_ball_trajectory(hb, d1=0, d2=0):
    ub_0 = np.sqrt(2*g*(hb - d1))  # velocity of ball at throw point
    Tb = 2*ub_0/g + d2  # flying time of the ball
    return Tb, ub_0


def steps_from_time(T, dt):
    """ Method to find nr of timesteps

    Args:
        T (double): Time in seconds
        dt (double): Timestep in seconds

    Returns:
        int: Number of timesteps in T
    """
    return int(np.floor(T / dt) + 1)


def find_continuous_intervals(indices):
    """ Find connected intervals of values

    Args:
        indices ([int]): List of indexes of true intervals

    Returns:
        set: A 2xN set where N is nr of intevals found and 2 represents start and end indexes.
    """

    intervals = set()
    indices = np.squeeze(indices)
    if indices.size != 0:
        # Find intervals where gN<=0
        last = indices[0]
        start = last

        for i in indices:
            if i - last > 1 or i == indices[-1]:
                intervals.add((start - 1, last))
                start = i
            last = i
    return intervals


def plot_intervals(ax, intervals, dt, colors=None):
    if colors is None or len(colors) != len(intervals):
        colors = np.repeat('gray', len(intervals))  # #2ca02c

    for i, col in zip(intervals, colors):
        ax.axvspan(dt * i[0], dt * i[1], facecolor=col, alpha=0.3)
    return ax
