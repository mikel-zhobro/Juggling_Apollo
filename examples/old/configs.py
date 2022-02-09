from pickle import FALSE
import numpy as np

from utils import DotDict
from juggling_apollo.utils import steps_from_time, plt, rtime
from ApolloPlanners import JugglingPlanner, MinJerk
from juggling_apollo import MinJerk


def get_minjerk_config(dt, end_repeat, plot=False):
    mj = DotDict()

    # 1. --------------- Compute juggling params for given E, tau and ---------------
    # A) Orientation
    #    World: x-left, y-forward, z-up
    #    TCP:   x-down, y-right,   z-forward
    # B) Position
    #    only z changes, x and y stay contant for 1D case
    Ex = 0.15                                                                 # Ex = z_catch - z_throw
    tau = 0.5                                                                 # tau = mj.T_hand + mj.T_empty (length of the repeatable part)
    dwell_ration = 0.6                                                        # what part of tau is used for mj.T_hand
    T_hand, T_empty, ub_throw, H, z_catch = JugglingPlanner.calc(tau, dwell_ration, Ex, slower=3.51)  # 2.0 gives error

    mj.T_hand = rtime(T_hand, dt)
    mj.T_empty = rtime(T_empty, dt)
    # Times
    mj.T_throw_first = rtime(mj.T_hand*0.5, dt)                                                # Time to use for the first throw from home position
    mj.T_fly = mj.T_hand + 2.0*mj.T_empty
    T_FULL = mj.T_empty + mj.T_hand
    mj.T_FULL = T_FULL + end_repeat*dt
    N_1 =               steps_from_time(mj.T_FULL, dt)                         # size of our vectors(i.e. length of the learning interval)
    mj.N_throw =        steps_from_time(mj.T_throw_first, dt)                     # timestep where throw must happen
    mj.N_throw_empty =  steps_from_time(mj.T_throw_first+mj.T_empty, dt)              # timestep where catch must happen
    mj.N_repeat_point = steps_from_time(mj.T_throw_first, dt)                     # timestep from where the motion should be repeated

    # ---------------           ---------------
    # --------------- Min Jerk ---------------
    smooth_acc = False
    ub_catch = -ub_throw*0.9
    i_a_end = 0
    tt=[0.0,      mj.T_throw_first,     mj.T_throw_first+mj.T_empty,   T_FULL  ]

    yy=[0.0,      z_catch/4.0,       z_catch/2.0,             0.0     ]
    zz=[0.0,      0.0,               z_catch,                 0.0     ]
    uuyy=[0.0,      ub_throw/12.0,     ub_catch/12.0,           0.0     ]
    uuzz=[0.0,      ub_throw/12.0,     ub_catch/12.0,           0.0     ]

    z_des, velo, accel, jerk = MinJerk.get_multi_interval_minjerk_1D(dt, smooth_acc=smooth_acc, i_a_end=i_a_end, tt=tt, xx=zz, uu=uuzz, extra_at_end=end_repeat+1)  # Min jerk trajectories (out of the loop since trajectory doesn't change)
    y_des, velo2, accel2, jerk2 = MinJerk.get_multi_interval_minjerk_1D(dt, smooth_acc=smooth_acc, i_a_end=i_a_end, tt=tt, xx=yy, uu=uuyy, extra_at_end=end_repeat+1)  # Min jerk trajectories (out of the loop since trajectory doesn't change)


    # Cartesian -> JointSpace                   <------------------------------------------------------------------------------------------ Min Jerk Trajectory (CARTESIAN AND JOINT SPACE)
    thetas                   = np.zeros_like(z_des)
    xyz_traj_des             = np.zeros((thetas.size, 3))
    xyz_traj_des[:,2]        = z_des
    xyz_traj_des[:,1]        = y_des

    if plot:
        print(z_catch)
        MinJerk.plotMJ(dt, tt, zz, uuzz, smooth_acc, (z_des, velo, accel, jerk))
        MinJerk.plotMJ(dt, tt, yy, uuyy, smooth_acc, (y_des, velo2, accel2, jerk2))
        plt.show()
    if plot:
        from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(xyz_traj_des[:,0], xyz_traj_des[:,1], xyz_traj_des[:,2], 'gray')
        plt.show()

    return N_1, xyz_traj_des, thetas, mj

