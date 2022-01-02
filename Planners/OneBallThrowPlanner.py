import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import MinJerk
from utils import g, plot_A, set_axes_equal, utilities


T_home = np.array([[0.0, -1.0, 0.0,  0.3],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.9],
                   [-1.0, 0.0, 0.0, -0.5],
                   [0.0,  0.0, 0.0,  1.0 ]], dtype='float')

def plan(dt, kinematics, h=0.65, throw_height=0.25, swing_size=0.46, slower=1.0, rep=1, verbose=False):
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
    t1 = 1.4                        # throw time
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
        q_s[i] = kinematics.IK(Tmp)

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
        Ji = kinematics.J(q_s[i])[:3,:]
        H[7:,:7] = -Ji
        H[:7,7:] = Ji.T
        # qv_s[i] = np.linalg.pinv(Ji).dot(vs[i])
        b[7:] = -vs[i]
        qv_s[i] = np.linalg.inv(H).dot(b)[:7]

    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_multi_interval_minjerk_xyz(dt, ts, q_s, qv_s, smooth_acc=False, only_pos=False, i_a_end=0)

    q_traj = q_traj.reshape(-1,7,1)

    T_traj = kinematics.seqFK(q_traj)

    if verbose:
        plot_A(q_traj.reshape(1,-1,7,1), dt=dt, limits=kinematics.limits)
        plt.suptitle("Joint angles")
        plot_A(qv_traj.reshape(1,-1,7,1), dt=dt, limits=kinematics.vlimits, index_labels=[r"$\dot{\theta}_%d$" %i for i in range(7)])
        plt.suptitle("Joint angle velocities")
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



def plan2(dt, kinematics, h=0.5, throw_height=0.0, swing_size=0.46, slower=1.0, rep=1, verbose=False):
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
    d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186

    dt = dt/slower

    tf = 2.0*math.sqrt(2.0*(h-throw_height)/g)  # time of flight
    t1 = 0.4                     # throw time
    t2 = t1 + 1.*tf              # home time
    ts = [0., t1, t2]
    print('TCatch', t2)

    # Cartesian positions
    x0 = np.array([0., 0., 0]).reshape(3,1)   # home/catch position
    x1 = np.array([0., 0., 0-throw_height]).reshape(3,1)   # throw position
    xs = [x0, x1, x0]

    # Cartesian velocities
    alpha = 0.
    v_throw = 0.5*g*tf # in z direction
    v0 = np.zeros((3,1))
    v1 = np.array([0.,np.sin(alpha)*v_throw, np.cos(alpha)*v_throw]).reshape(3,1)   # throw velocity
    v2 = np.zeros((3,1))     # catch velocity
    vs = [v0, v1, v2]

    # Joint Positions
    q_s = np.zeros((len(xs),7,1))
    Tmp = T_home.copy()
    R = Tmp[:3,:3]
    for i in range(len(q_s)):
        Tmp[:3,:3] = R
        Tmp[:3,3:4] = T_home[:3, 3:4] + xs[i]
        q_s[i] = kinematics.IK(Tmp)

    # Joint Velocities
    W = np.diag([0.1, 1., 1, 44, 44, 24, 144])
    H = np.zeros((10,10)); H[:7,:7] = W

    b = np.zeros((10,1))
    qv_s = np.zeros((len(xs),7,1))
    for i in range(len(qv_s)):
        Ji = kinematics.J(q_s[i])[:3,:]
        H[7:,:7] = -Ji
        H[:7,7:] = Ji.T
        # qv_s[i] = np.linalg.pinv(Ji).dot(vs[i])
        # qv_s[i,:4] = np.linalg.pinv(Ji[:, :4]).dot(vs[i])
        b[7:] = -vs[i]
        qv_s[i] = np.linalg.inv(H).dot(b)[:7]

        # findBestThrowPosition(kinematics.FK, kinematics.J, q_s[i], qv_s[i], vs[i], T_home[:3,:3])

    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_multi_interval_multi_dim_minjerk(dt, ts, q_s, qv_s, smooth_acc=False, only_pos=False, i_a_end=0)
    T_traj = kinematics.seqFK(q_traj)

    if verbose:
        plot_A(q_traj.reshape(1,-1,7,1), dt=dt, limits=kinematics.limits)
        plt.suptitle("Joint angles")
        plot_A(qv_traj.reshape(1,-1,7,1), dt=dt, limits=kinematics.vlimits, index_labels=[r"$\dot{\theta}_%d$" %i for i in range(7)])
        plt.suptitle("Joint angle velocities")
        # plot_A(180./np.pi*qa_traj.reshape(1,-1,7,1))
        # plt.suptitle("Angle Accelerations")
        # plt.show()

        from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(T_traj[:,0, -1], T_traj[:,1, -1], T_traj[:,2, -1], 'gray')
        ax.scatter(*T_traj[0, 0:3, -1], label="start")
        ax.scatter(*T_traj[int(t1/dt), 0:3, -1], label="throw")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        set_axes_equal(ax)
        plt.legend()
        plt.show()

    return q_traj, T_traj



def constrained_optim(J, q_init, vgoal, jac=None):
    con = lambda i: lambda qdot: J[i, :].dot(qdot) - vgoal[i]
    cons = (
            # {'type':'eq', 'fun': con(0)},
            # {'type':'eq', 'fun': con(1)},
            # {'type':'eq', 'fun': con(2)},
            {'type':'ineq', 'fun': lambda qdot: 0.5 - abs(J[0, :].dot(qdot))},
            {'type':'ineq', 'fun': lambda qdot: 0.5 - abs(J[1, :].dot(qdot))},
            )

    bounds = [utilities.JOINTS_V_LIMITS[j] for j in utilities.R_joints]

    def fun(q_dot):
        qd = np.asarray(q_dot).reshape(7,1).copy()
        return q_dot.T.dot(q_dot)

    def fun(qdot):
        return - J[2, :].dot(qdot)

    # def fun(qdot):
    #     v = J[:3, :].dot(qdot)
    #     return - v.dot(v)

    result = minimize(fun, q_init, method="SLSQP", bounds=bounds, constraints=cons)
    if not result.success:
        print("optim was unseccussfull")
        return q_init
    return result.x



def findBestThrowPosition(FK, J, q_init, qdot_init, vgoal, R_des, jac=None):
    con = lambda i: lambda qqd: J(qqd[:7])[i, :].dot(qqd[7:]) - vgoal[i]
    con_R = lambda i: lambda qqd: FK(qqd[:7])[i,i] - R_des[i,i]

    cons = tuple({'type':'eq', 'fun': con(i)} for i in range(2)) + tuple({'type':'eq', 'fun': con_R(i)} for i in range(3))


    bounds =  [utilities.JOINTS_LIMITS[j] for j in utilities.R_joints] + [utilities.JOINTS_V_LIMITS[j] for j in utilities.R_joints]

    def fun(qqd):
        return - J(qqd[:7])[2, :].dot(qqd[7:])

    result = minimize(fun, (q_init, qdot_init) , method="SLSQP", bounds=bounds, constraints=cons)
    qqd = result.x
    if not result.success:
        print("optim was unseccussfull")
        return q_init
    return result.x