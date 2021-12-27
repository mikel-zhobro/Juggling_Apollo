import math
import numpy as np
import matplotlib.pyplot as plt
import SiteSwapPlanner

import MinJerk
from utils import g, plot_A, set_axes_equal


def plan(dt, T_home, T_dhtcp_tcp, IK, J, seqFK, h=0.75, throw_height=0.35, swing_size=0.46, slower=1.0, rep=1, verbose=False):
    """
    TODO: incorporate it in the siteswap planner (or no)
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
    # dt = dt/slower

    jp = SiteSwapPlanner.JugglingPlanner()
    pattern=(3,); h=0.3; r_dwell=0.6; throw_height=0.15; swing_size=0.2; w=0.3; slower=1.0; rep=1
    plan = jp.plan(dt, 2, pattern=pattern, h=h, r_dwell=r_dwell, throw_height=throw_height, swing_size=swing_size, w=w, slower=slower, rep=rep)

    # plan.plot()
    cts  = plan.hands[0].ct_period
    for ct in cts:
        ts = ct.traj.tt
        xs = [xtmp.T.reshape(3,1) for xtmp in ct.traj.xx]
        vs = [vtmp.T.reshape(3,1) for vtmp in ct.traj.vv]

    N, x0, v0, a0, j0, rot_traj_des = plan.hands[0].get(get_thetas=True)  # get plan for hand0

    # Joint Positions
    q_s = np.zeros((len(xs),7,1))
    Tmp = T_home.copy()
    R = Tmp[:3,:3]
    for i in range(len(q_s)):
        Tmp[:3,:3] = R
        Tmp[:3,3:4] = T_home[:3, 3:4] + xs[i]
        q_s[i] = IK(Tmp.dot(T_dhtcp_tcp.T))

    # Joint Velocities
    W = np.eye(7)
    W[3:,3:] *= 1.

    H = np.zeros((10,10))
    H[:7,:7] = W
    b = np.zeros((10,1))
    qv_s = np.zeros((len(xs),7,1))
    for i in range(len(qv_s)):
        Ji = J(q_s[i])[:3,:]
        H[7:,:7] = -Ji
        H[:7,7:] = Ji.T
        qv_s[i] = np.linalg.pinv(Ji).dot(vs[i])
        b[7:] = -vs[i]
        # qv_s[i] = np.linalg.inv(H).dot(b)[:7]

    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_multi_interval_minjerk_xyz(dt, ts, q_s, qv_s, smooth_acc=False, only_pos=False, i_a_end=0)
    q_traj = q_traj.reshape(-1,7,1)

    T_traj = seqFK(q_traj)

    if verbose:
        A = 180./np.pi
        # A = 1.
        plot_A(A*q_traj.reshape(1,-1,7,1), dt=dt)
        plt.suptitle("Angle Positions")
        # plt.savefig('Joint_Angle_Traj_joint.pdf')
        plot_A(A*qv_traj.reshape(1,-1,7,1), dt=dt)
        plt.suptitle("Angle Velocities")
        # plt.savefig('Joint_Angle_Vel_Traj_joint.pdf')
        # plot_A(180./np.pi*qa_traj.reshape(1,-1,7,1))
        # plt.suptitle("Angle Accelerations")
        # plt.show()

        # 3D plot
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # prepare
        balltraj_throw = cts[0].ballTraj.xxx[:] + T_home[:3, -1]
        balltraj_catch = cts[0].ct_c.ballTraj.xxx[:] + T_home[:3, -1]
        xXx = x0-x0[0]+T_home[:3, -1]
        yYy = T_traj[:, 0:3, -1]

        a = ax.plot3D(balltraj_throw[:,0], balltraj_throw[:,1], balltraj_throw[:,2], linestyle='--', label='throw ball traj')
        tss = [0, len(balltraj_throw)//2, -1]
        ax.quiver(balltraj_throw[tss,0], balltraj_throw[tss,1], balltraj_throw[tss,2],
                  cts[0].ballTraj.vvv[tss,0], cts[0].ballTraj.vvv[tss,1], cts[0].ballTraj.vvv[tss,2],
                  length=0.07, normalize=True, color=a[0].get_color())


        a = ax.plot3D(balltraj_catch[:,0], balltraj_catch[:,1], balltraj_catch[:,2], linestyle='--', label='catch ball traj')
        tss = [0, len(balltraj_catch)//2, -1]
        ax.quiver(balltraj_catch[tss,0], balltraj_catch[tss,1], balltraj_catch[tss,2],
                  cts[0].ct_c.ballTraj.vvv[tss,0], cts[0].ct_c.ballTraj.vvv[tss,1], cts[0].ct_c.ballTraj.vvv[tss,2],
                  length=0.07, normalize=True, color=a[0].get_color(),)

        ax.plot3D(yYy[:,0], yYy[:,1], yYy[:,2], 'blue', label='joint space plan')
        ax.plot3D(xXx[:,0], xXx[:,1], xXx[:,2], 'red', label='cartesian space plan')

        ax.scatter(*yYy[0, 0:3], color ='k') #label="start")
        ax.scatter(*yYy[int(ts[1]/dt), 0:3], color ='k') # label="throw")
        # ax.scatter(*yYy[int(ts[2]/dt), 0:3], label="catch")

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        set_axes_equal(ax)
        plt.title('Right hand trajectory')
        plt.legend()
        plt.show()

    return q_traj, T_traj
