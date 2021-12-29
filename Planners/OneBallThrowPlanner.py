import math
import numpy as np
import matplotlib.pyplot as plt

import MinJerk
from utils import g, plot_A, set_axes_equal


def plan(dt, T_home, IK, J, seqFK, h=0.75, throw_height=0.35, swing_size=0.46, slower=1.0, rep=1, verbose=False):
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
    t1 = 0.8                        # throw time
    t2 = 2*t1                       # catch time
    t3 = t2 + 0.8*tf                # home time
    ts = [0., t1, t2,  t3]

    # Cartesian positions
    stop_height = throw_height+(h-throw_height)/4.
    catch_height = throw_height/4.
    x0 = np.zeros((3,1))
    x1 = np.array([0., swing_size, throw_height]).reshape(3,1)      # swing position
    x2 = np.array([0., swing_size, catch_height]).reshape(3,1)   # catch-wait position
    xs = [x2, x0, x1, x2]

    # Cartesian velocities
    alpha = 0.
    v_throw = 0.5*g*tf # in z direction
    v_catch = -0.2*v_throw
    v0 = np.zeros((3,1))
    v1 = np.array([0.,np.sin(alpha)*v_throw, np.cos(alpha)*v_throw]).reshape(3,1)
    v2 = np.zeros((3,1))
    vs = [v2, v0, v1, v2]

    a_s = [None, None, ]
    # Joint Positions
    q_s = np.zeros((len(xs),7,1))
    Tmp = T_home.copy()
    R = Tmp[:3,:3]
    for i in range(len(q_s)):
        # if i ==1:
        #     alpha = -0.2
        #     rx = -np.array([-np.sin(alpha), 0, np.cos(alpha)]).reshape(3,1)
        #     # rx = -vs[i]/np.linalg.norm(vs[1])
        #     ry = T_home[:3,1:2]
        #     # ry = np.cross(rz, rx, axis=0)
        #     rz = np.cross(rx, ry, axis=0)
        #     # rx = -np.cross(rz, T_home[:3,1:2], axis=0)
        #     R = np.hstack((rx, ry, rz))
        # else:
        #     R = T_home[:3,:3]
        Tmp[:3,:3] = R
        Tmp[:3,3:4] = T_home[:3, 3:4] + xs[i]
        print(Tmp)
        q_s[i] = IK(Tmp)

    # Joint Velocities
    W = np.eye(7)
    W[3:,3:] *= 1.
    W[-2,-2] = 20.
    W[-4,-4] = 24.

    H = np.zeros((10,10))
    H[:7,:7] = W
    b = np.zeros((10,1))
    qv_s = np.zeros((len(xs),7,1))
    for i in range(len(qv_s)):
        Ji = J(q_s[i])[:3,:]
        H[7:,:7] = -Ji
        H[:7,7:] = Ji.T
        # qv_s[i] = np.linalg.pinv(Ji).dot(vs[i])
        b[7:] = -vs[i]
        qv_s[i] = np.linalg.inv(H).dot(b)[:7]

    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_multi_interval_minjerk_xyz(dt, ts, q_s, qv_s, smooth_acc=False, only_pos=False, i_a_end=0)

    q_traj = q_traj.reshape(-1,7,1)

    T_traj = seqFK(q_traj)

    if verbose:
        A = 180./np.pi
        A = 1.
        plot_A(A*q_traj.reshape(1,-1,7,1), dt=dt)
        plt.suptitle("Angle Positions")
        plot_A(A*qv_traj.reshape(1,-1,7,1))
        plt.suptitle("Angle Velocities")
        # plot_A(180./np.pi*qa_traj.reshape(1,-1,7,1))
        # plt.suptitle("Angle Accelerations")
        # plt.show()

        from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(T_traj[:,0, -1], T_traj[:,1, -1], T_traj[:,2, -1], 'gray')
        ax.scatter(*T_traj[0, 0:3, -1], label="start")
        ax.scatter(*T_traj[int(t1/dt), 0:3, -1], label="back")
        # ax.scatter(*T_traj[int(t2/dt), 0:3, -1], label="stop")
        ax.scatter(*T_traj[int(t2/dt), 0:3, -1], label="throw")
        ax.scatter(*T_traj[int(t3/dt), 0:3, -1], label="home")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        set_axes_equal(ax)
        plt.legend()
        plt.show()

    return q_traj, T_traj
