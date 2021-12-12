import math
import numpy as np
import matplotlib.pyplot as plt

import MinJerk
from utils import g, plot_A, set_axes_equal


def plan(dt, T_home, IK, J, seqFK, h=0.5, throw_height=0.2, swing_size=0.2, slower=1.0, rep=1, verbose=False):
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
    t1 = 0.4                        # swing time
    t2 = t1 + tf/4.                 # stop time
    t3 = t1 + 0.7*tf                    # catch time
    t4 = t2 + 2.*tf                  # wait time
    t5 = t4 + 4.*t1                 # back home time
    # ts = [0., t1, t2, t4, t5]
    ts = [0., t1, t3, t4]

    # Cartesian positions
    stop_height = throw_height+(h-throw_height)/4.
    catch_height = throw_height/4
    x0 = np.zeros((3,1))                                            
    x1 = np.array([0., swing_size, throw_height]).reshape(3,1)      # swing position
    x2 = np.array([0., swing_size, stop_height]).reshape(3,1)       # stop position
    x3 = np.array([0., swing_size, catch_height]).reshape(3,1)      # catch position
    x4 = np.array([0., swing_size, catch_height]).reshape(3,1)      # catch-wait position
    x5 = np.zeros_like(x0)                                          # home position
    # xs = [x0, x1, x2, x3, x4]
    xs = [x0, x1, x3, x3]

    # Cartesian velocities
    v_throw = 0.5*g*tf # in z direction
    v_catch = -0.0*v_throw
    v0 = np.zeros((3,1))
    v1 = np.array([0., 0., v_throw]).reshape(3,1)
    v2 = np.zeros((3,1))
    v3 = np.array([0., 0., 0.]).reshape(3,1)
    v4 = np.zeros((3,1))
    # vs = [v0, v1, v2, v3, v4]
    vs = [v0, v1, v3, v3]

    a_s = [None, None, ]
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

    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_minjerk_xyz(dt, ts, q_s.transpose(1,0,2), qv_s.transpose(1,0,2), smooth_acc=False, only_pos=False)
    q_traj, qv_traj, qa_traj, qj_traj = np.asarray(q_traj).T, np.asarray(qv_traj).T, np.asarray(qa_traj).T, np.asarray(qj_traj).T
    
    q_traj = q_traj.reshape(-1,7,1)

    T_traj = seqFK(q_traj)

    if verbose:
        plot_A(180./np.pi*q_traj.reshape(1,-1,7,1), dt=dt)
        plt.suptitle("Angle Positions")
        plot_A(180./np.pi*qv_traj.reshape(1,-1,7,1))
        plt.suptitle("Angle Velocities")
        # plot_A(180./np.pi*qa_traj.reshape(1,-1,7,1))
        # plt.suptitle("Angle Accelerations")
        plt.show()
        
    if verbose:
        from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(T_traj[:,0, -1], T_traj[:,1, -1], T_traj[:,2, -1], 'gray')
        ax.scatter(*T_traj[0, 0:3, -1], label="start")
        ax.scatter(*T_traj[int(t1/dt), 0:3, -1], label="throw")
        ax.scatter(*T_traj[int(t3/dt), 0:3, -1], label="catch")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        set_axes_equal(ax)
        plt.legend()
        plt.show()

    return q_traj, T_traj