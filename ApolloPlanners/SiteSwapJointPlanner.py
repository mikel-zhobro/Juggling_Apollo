#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   SiteSwapJointPlanner.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   transforms plans for usage in Apollo
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import MinJerk
import SiteSwapPlanner
import utils


def vel_from_position(qtraj, qv0, dt):
    """ Compute velocities from positions using a simple discretization approach.

    Args:
        qtraj (np.array(N,7,1)): joint positions
        qv0 (np.array(7,1)): initial velocity
        dt (float): time step

    Returns:
        qvtraj (np.array(N,7,1)): joint velocities
    """
    qvtraj = np.zeros_like(qtraj)

    qvtraj[0] = qv0.reshape(-1,1)

    for i in range(1, len(qtraj)):
        qvtraj[i] = ((qtraj[i] - qtraj[i-1])/dt  + qvtraj[i-1]) /2.
    return qvtraj


def plan(dt, kinematics, joint_space=True, pattern=(3,), h=0.6, throw_height=0.25, swing_size=0.16, slower=1.0, rep=1, verbose=False, r_dwell=0.45, w=0.4):
    """

    check `SiteSwapPlanner.JugglingPlanner().plan()` for a better documentation of the parameters.

    Args:
        dt (float): time step
        kinematics (ApolloKinematics): kinematics for apollo to compute IK, seqIK, FK, seqFK, Jacobian..
        verbose (bool): whether to plan in joint-space or in cartesian-space
        pattern (tuple): siteswap pattern
        h (float): height of the standard 3-throw
        throw_height (float): at what height we perform the throw
        swing_size (float): how much we can swing
        slower (float): how many times slower we want to perform the juggling trajectory
        rep (int): number of repitation of the juggling period
        verbose (bool): whether to print/plot verbose information
        r_dwell (float): dwell ratio
        w (float): side length of the regular polygon the hands are positioned at

    Returns:
        q_traj (np.array(N,7,1)): joint position trajectory
        qv_traj (np.array(N,7,1)): joint velocity trajectory
        T_traj (np.array(N,4,4)):  the corresponding cartesian trajectory
    """
    jp = SiteSwapPlanner.JugglingPlanner()
    plan = jp.plan(dt, 2, pattern=pattern, h=h, r_dwell=r_dwell, throw_height=throw_height, swing_size=swing_size, w=w, rep=rep)
    # plan.plot()

    # Get CTs(edge conditions) for the first hand
    cts  = plan.hands[0].ct_period
    ts = []
    xs = []
    vs = []
    for ct in cts:
        ts += [t_ * slower for t_ in ct.traj.tt]
        xs += [xtmp.T.reshape(3,1) for xtmp in ct.traj.xx]
        vs += [vtmp.T.reshape(3,1)/slower for vtmp in ct.traj.vv]


    # A. Joint space plan
    ########################################################################################################
    # a. Find best position to perform the throw at by solving nonlinear optim problem `findBestThrowPosition`
    # dt = dt/slower
    q_init = np.array([0.2975, -0.9392, -0.5407,  1.4676,  1.35  , -0.4971, -0.4801]).reshape(7,1)
    T_home = kinematics.FK(q_init)
    # q_init = np.array([0.1193, -1.0512, -0.1186,  1.1475, -0.4451,  0.2153,  1.3167]).reshape(7,1)
    R_des = np.eye(3)
    R_des[:,2:3] = vs[1]/ np.linalg.norm(vs[1])
    R_des[:,0] = np.cross(R_des[:,1], R_des[:,2])
    q_init, qdot_init = findBestThrowPosition(FK=kinematics.FK, J=kinematics.J, q_init=q_init, qdot_init=np.zeros((7,1)), vgoal=vs[1], R_des=R_des)


    # b. Compute the joint space correspondigs of the CTs
    #    used for joint-space planning
    offset = xs[1]
    q_s = np.zeros((len(xs),7,1))
    Tmp = T_home.copy()
    R = Tmp[:3,:3]
    for i in range(len(q_s)):
        Tmp[:3,:3] = R
        Tmp[:3,3:4] = T_home[:3, 3:4] + xs[i] - offset
        q_s[i] = kinematics.IK(Tmp)

    # Joint Velocities
    # Needed if we use the weighted pinv of jacobian
    # W = np.eye(7)
    # W[3:,3:] *= 1.
    # H = np.zeros((10,10))
    # H[:7,:7] = W
    # b = np.zeros((10,1))

    qv_s = np.zeros((len(xs),7,1))
    for i in range(len(qv_s)):
        Ji = kinematics.J(q_s[i])[:3,:]
        # 1. Pinv of jacobian
        # qv_s[i] = np.linalg.pinv(Ji).dot(vs[i])

        # 2. Weighted pinv of jacobian
        # b[7:] = -vs[i]
        # H[7:,:7] = -Ji
        # H[:7,7:] = Ji.T

        # qv_s[i] = np.linalg.inv(H).dot(b)[:7]
        # 3. Constrained optimization
        qv_s[i], vw = constrained_optim(Ji, np.zeros((7,1)), vs[i])
        print(i, vs[i].T, vw.T)
    q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_multi_interval_multi_dim_minjerk(dt, ts, q_s, qv_s, smooth_acc=True, only_pos=False, i_a_end=True)
    # q_traj, qv_traj, qa_traj, qj_traj = MinJerk.get_multi_interval_minjerk_xyz(dt, ts, q_s, qv_s, smooth_acc=False, only_pos=False, i_a_end=0)
    T_traj = kinematics.seqFK(q_traj)
    ########################################################################################################

    # B. Cartesian space plan
    ########################################################################################################
    N, x0, v0, a0, j0, rot_traj_des = plan.hands[0].get(get_thetas=True)  # get plan for hand0
    xXx = x0-offset.squeeze()+T_home[:3, -1]
    T_traj_cartesian = utils.utilities.pR2T(xXx, rot_traj_des)
    q_cartesian, _, _ = kinematics.seqIK(T_traj_cartesian, considered_joints=[])
    qv_cartesian = vel_from_position(q_cartesian, qv_traj[0], dt)
    joint_list = [0,1,2,3, 4, 5,6]
    ########################################################################################################


    if verbose:
        plan.plot(orientation=True)
        joint_list = [0,1,2,3,4,5,6]

        # 1.Joint trajectories for the joint-space and cartesian-space planning
        # a. Jointspace Plan
        utils.plot_A(q_traj.reshape(1,-1,7,1), indexes_list=joint_list, dt=dt, limits=kinematics.limits, xlabel=r"$t$ [s]", ylabel=r"angle [$grad$]", scatter_times=ts)
        plt.suptitle("Angle Positions [Joint space plan]")
        # plt.savefig('Joint_Angle_Traj_joint.pdf')
        utils.plot_A(qv_traj.reshape(1,-1,7,1), indexes_list=joint_list, dt=dt, limits=kinematics.vlimits, index_labels=[r"$\dot{\theta}_%d$" %(i+1) for i in range(7)],
            xlabel=r"$t$ [s]", ylabel=r"[$\frac{grad}{s}$]", scatter_times=ts)
        plt.suptitle("Angle Velocities [Joint space plan]")
        # plt.savefig('Joint_Angle_Vel_Traj_joint.pdf')
        # utils.plot_A(180./np.pi*qa_traj.reshape(1,-1,7,1))
        # plt.suptitle("Angle Accelerations")
        # plt.show()

        # b. Cartesian Plan
        utils.plot_A(q_cartesian.reshape(1,-1,7,1), indexes_list=joint_list, dt=dt, limits=kinematics.limits, xlabel=r"$t$ [s]", ylabel=r"angle [$grad$]", scatter_times=ts)
        plt.suptitle("Angle positions [Cartesian space plan]")
        # plt.savefig('Joint_Angle_Traj_joint.pdf')
        utils.plot_A(qv_cartesian.reshape(1,-1,7,1), indexes_list=joint_list, dt=dt, limits=kinematics.vlimits, index_labels=[r"$\dot{\theta}_%d$" %(i+1) for i in range(7)],
                xlabel=r"$t$ [s]", ylabel=r"angle [$grad$]", scatter_times=ts)
        plt.suptitle("Angle velocities [Cartesian space plan]")

        plt.show()


        # 1. 3D cartesian trajectories for the joint-space and cartesian-space planning

        from mpl_toolkits.mplot3d import axes3d, Axes3D # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # prepare
        yYy = T_traj[:, 0:3, -1]

        balltraj_throw = cts[0].ballTraj.xxx[:] + T_home[:3, -1] -offset.squeeze()
        a = ax.plot3D(balltraj_throw[:,0], balltraj_throw[:,1], balltraj_throw[:,2], linestyle='--', label='throw ball traj')
        tss = [0, len(balltraj_throw)//2, -1]
        ax.quiver(balltraj_throw[tss,0], balltraj_throw[tss,1], balltraj_throw[tss,2],
                  cts[0].ballTraj.vvv[tss,0], cts[0].ballTraj.vvv[tss,1], cts[0].ballTraj.vvv[tss,2],
                  length=0.07, normalize=True, color=a[0].get_color())

        ax.plot3D(yYy[:,0], yYy[:,1], yYy[:,2], 'blue', label='joint space plan')
        ax.plot3D(xXx[:,0], xXx[:,1], xXx[:,2], 'red', label='cartesian space plan')

        ax.scatter(*yYy[0, 0:3], color ='k') #label="start")
        ax.scatter(*yYy[int(ts[1]/dt), 0:3], color ='k') # label="throw")

        ax.scatter(*(T_home[:3, 3:4] + xs[0] - offset), color ='purple') #label="start")
        ax.scatter(*(T_home[:3, 3:4] + xs[1] - offset), color ='purple') #label="start")
        ax.scatter(*(T_home[:3, 3:4] + xs[2] - offset), color ='purple') #label="start")
        # ax.scatter(*yYy[int(ts[1]/dt), 0:3], color ='purple') # label="throw")
        # ax.scatter(*yYy[int(ts[2]/dt), 0:3], label="catch")

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        utils.set_axes_equal(ax)
        plt.title('Right hand trajectory')
        plt.legend()
        plt.show()

    if joint_space:
        return q_traj, qv_traj, T_traj
    else:
        return q_cartesian, qv_cartesian, T_traj_cartesian


def constrained_optim(J, q_init, vgoal, jac=None):
    con = lambda i: lambda qdot: J[i, :].dot(qdot) - vgoal[i]
    cons = (
            {'type':'eq', 'fun': con(0)},
            {'type':'eq', 'fun': con(1)},
            {'type':'eq', 'fun': con(2)},
            # {'type':'ineq', 'fun': lambda qdot: 0.5 - abs(J[0, :].dot(qdot))},
            # {'type':'ineq', 'fun': lambda qdot: 0.5 - abs(J[1, :].dot(qdot))},
            )

    bounds = [utils.utilities.JOINTS_V_LIMITS[j] for j in utils.utilities.R_joints]

    def fun(q_dot):
        # qd = np.asarray(q_dot).reshape(7,1).copy()
        return q_dot.T.dot(q_dot)

    # def fun(qdot):
    #     return J[2, :].dot(qdot)

    # def fun(qdot):
    #     v = J[:3, :].dot(qdot)
    #     return - v.dot(v)

    result = minimize(fun, q_init, method="SLSQP", bounds=bounds, constraints=cons)
    if not result.success:
        print("optim was unseccussfull")
        # return q_init
    v_ach = J[:3, :].dot(result.x.reshape(7,1))
    return result.x.reshape(7,1),  v_ach


def findBestThrowPosition(FK, J, q_init, qdot_init, vgoal, R_des, jac=None):
    con = lambda i: lambda qqd: J(qqd[:7])[i, :].dot(qqd[7:]) - vgoal[i]
    con_R = lambda i, j: lambda qqd: FK(qqd[:7])[:3,:3].T.dot(R_des)[i,i]

    cons = tuple({'type':'eq', 'fun': con(i)} for i in range(3)) + tuple({'type':'eq', 'fun': con_R(i, 2)} for i in range(3)) # for j in range(3))


    bounds =  [utils.utilities.JOINTS_LIMITS[j] for j in utils.utilities.R_joints] + [utils.utilities.JOINTS_V_LIMITS[j] for j in utils.utilities.R_joints]

    def fun(qqd):
        return J(qqd[:7])[2, :].dot(qqd[7:])

    def fun(qqd):
        return qqd[7:].T.dot(qqd[7:])

    # def fun(qqd):
    #     return np.linalg.norm(FK(qqd[:7])[:3,:3] - R_des)

    result = minimize(fun, (q_init, qdot_init) , method="SLSQP", bounds=bounds, constraints=cons)
    qqd = result.x
    if not result.success:
        print("Cannot find home position")
        # return q_init
    return result.x[:7], result.x[7:]
