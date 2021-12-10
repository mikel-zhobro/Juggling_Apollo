import math
import numpy as np
import matplotlib.pyplot as plt

import MinJerk
from utils import g, plot_A


def plan(dt, T_home, IK, J, h=0.3, throw_height=0.15, swing_size=0.15, slower=1.0, rep=1, verbose=False):
    """
    Args:
        dt ([type]): [description]
        IK ([type]): function to calc IK
        J ([type]): function that calc jacobi given joint state
        h (float, optional): Height the ball should achieve
        throw_height (float, optional): at what height we perform the throw
        swing_size (float, optional): how much we can swing
        slower (float, optional):
        rep (int, optional): [description]. Defaults to 1.
    """
    dt = dt/slower

    tf = 2.0*math.sqrt(2.0*(h-throw_height)/g)  # time of flight
    t_swing = 0.2
    tf_stop = t_swing + tf/4.
    tf_catch_start = tf_stop + tf/4.
    tf_catch_end = tf_stop + tf
    t_start = tf_catch_end + 2.*t_swing
    ts = [0., t_swing, tf_stop, tf_catch_end, t_start]

    # Cartesian positions
    stop_height = throw_height+(h-throw_height)/4.
    catch_height = throw_height/4
    x0 = np.zeros((3,1))
    x1 = np.array([0., swing_size, throw_height]).reshape(3,1)
    x2 = np.array([0., swing_size, stop_height]).reshape(3,1)
    x3 = np.array([0., swing_size, catch_height]).reshape(3,1)
    x4 = np.zeros_like(x0)
    xs = [x0, x1, x2, x3, x4]

    # Cartesian velocities
    v_throw = 0.5*g*tf # in z direction
    v_catch = -0.5*v_throw
    v0 = np.zeros((3,1))
    v1 = np.array([0., 0., v_throw]).reshape(3,1)
    v2 = np.zeros((3,1))
    v3 = np.array([0., 0., v_catch]).reshape(3,1)
    v4 = np.zeros((3,1))
    vs = [v0, v1, v2, v3, v4]

    # Joint Positions
    q_s = np.zeros((len(xs),7,1))
    Tmp = T_home.copy()
    for i in range(len(q_s)):
        Tmp[:3,3:4] = T_home[:3, 3:4] + xs[i]
        q_s[i] = IK(Tmp)

    # Joint Velocities
    qv_s = np.zeros((len(xs),7,1))
    for i in range(len(qv_s)):
        Ji = J(q_s[i])[:3,:]
        qv_s[i] = np.linalg.pinv(Ji).dot(vs[i])

    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_minjerk_xyz(dt, ts, q_s.transpose(1,0,2), qv_s.transpose(1,0,2), smooth_acc=True, only_pos=False)
    q_traj, qv_traj, qa_traj, qj_traj = np.asarray(q_traj).T, np.asarray(qv_traj).T, np.asarray(qa_traj).T, np.asarray(qj_traj).T

    if verbose:
        plot_A(180./np.pi*q_traj.reshape(1,-1,7,1), dt=dt)
        plt.suptitle("Angle Positions")
        plot_A(180./np.pi*qv_traj.reshape(1,-1,7,1))
        plt.suptitle("Angle Velocities")
        plot_A(180./np.pi*qa_traj.reshape(1,-1,7,1))
        plt.suptitle("Angle Accelerations")
        plt.show()
    return q_traj.reshape(-1,7,1)