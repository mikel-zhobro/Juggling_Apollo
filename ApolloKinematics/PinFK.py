#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   PinFK.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   defines class for Pinocchio based forward and inverse kinematics
'''

import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from utilities import modrad, reduce_model
from settings import R_joints, L_joints, TCP, WORLD, BASE, FILENAME, JOINTS_LIMITS

np.set_printoptions(precision=4, suppress=True)
pin.switchToNumpyArray()  # https://github.com/stack-of-tasks/pinocchio/issues/802


# inv kinematics params
IT_MAX     = 10200
eps_pos    = 1e-3
eps_orient = 1e-2
DT         = 10e-4
damp       = 1e-12

class PinRobot():
    def __init__(self, r_arm=True):
        """ Initialize pinocchio dependent models/variables: model, data, joint_states
            filename ([str]): path to the urdf file
        """
        self.joints_list = R_joints if r_arm else L_joints
        self.model = reduce_model(FILENAME, jointsToUse=self.joints_list)
        self.data = self.model.createData()  # information that changes according to joint configuration etc
                                             # Only used as internal state(still all functions should be called with certain joint_state as input)

        # Use "BASE" instead of 'universe' as base coordinate frame (Set the BASE Frame)
        self.SE3_base_origin = self.FK_f2f(pin.neutral(self.model), BASE, WORLD, homog=False)

    def clip_limit_joints(self, Q):
        """ Clipps Q according to the joint limits

        Args:
            Q (np.array(7,1)): joint angles

        Returns:
            Q_clipped (np.array(7,1)): clipped joint angles
        """
        for i, name in enumerate(self.joints_list):
            Q[i, 0] = np.clip(Q[i, 0], *JOINTS_LIMITS[name])
        return Q

    def FK(self, q, frameName=TCP, homog=True):
        """
        returns the SE3_base_frame(R,p) of frameName in base coordinates
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q.reshape(7,1))
        ret = self.SE3_base_origin * pin.updateFramePlacement(self.model, self.data, frameId)
        return ret.homogeneous if homog else ret  # updates data and returns T_base_frame

    def FK_f2f(self, q, baseName=BASE, frameName=TCP, homog=True):
        """
        returns the SE3_baseName_frameName(R,p) of frameName in baseName coordinates
        """
        baseId = self.model.getFrameId(baseName)
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        T_origin_base = pin.updateFramePlacement(self.model, self.data, baseId)
        T_origin_frame = pin.updateFramePlacement(self.model, self.data, frameId)
        return (T_origin_base.inverse() * T_origin_frame).homogeneous if homog else T_origin_base.inverse() * T_origin_frame

    def J(self, q, frameName=TCP):
        """
        returns SE3(R,p) and J of TCP in BASE frame
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        SE3_origin_tcp = pin.updateFramePlacement(self.model, self.data, frameId)  # computes frame relevant info
        J_local = pin.computeFrameJacobian(self.model, self.data, q, frameId)
        return J_local, self.SE3_base_origin * SE3_origin_tcp

    def ik_apollo(self, Q_start, goal_p, goal_R=None, frameName=TCP, plot=False):
        """ Inverse Kinematics

        Args:
            goal_p ([float, float, float]): Endeffector xyz position in BASE frame
            goal_R ([np.array((3,3))], optional): Endeffector's orientation in BASE frame. If not specified it will be vertical.
            Q_i ([float]*N_JointUsed], optional): Whether to plot the error.
            plot (bool, optional): Whether to plot the error.
            frameName (str, optional): Name of TCP

        Returns:
            [list]: joint configuration to achieve desired endeffector position/orientation
        """

        # Desired TCP cartesian position
        goal_p = np.array(goal_p).reshape(3,1)
        goal_R = goal_R if goal_R is not None else np.eye(3)[:,[2,0,1]]
        SE3_base_goal = pin.SE3(goal_R, goal_p)
        print("GOAL")
        print(SE3_base_goal)

        def get_se3_error(SE3_base_tcp_i):
            dMi = SE3_base_goal.actInv(SE3_base_tcp_i)
            err = pin.log(dMi).vector
            return err

        i=0
        errs = []
        Q_i = Q_start.copy()
        while True:

            # Calc T_base_tcp and J_base_tcp for the new joint_states
            J_base_tcp, SE3_base_tcp_i  = self.J(Q_i, frameName)

            # Calc cartesian errors
            err = get_se3_error(SE3_base_tcp_i)

            # 1 Calc qoint velocities
            qv = - J_base_tcp.T.dot(solve(J_base_tcp.dot(J_base_tcp.T) + damp * np.eye(6), err))

            # 2
            # J_invj = np.linalg.pinv(J_base_tcp)
            # qv = -np.matmul(J_invj, err)

            # Update joint_states
            Q_i = pin.integrate(self.model, Q_i, qv*DT)
            Q_i = modrad(Q_i)
            # Q_i = self.clip_limit_joints(Q_i)

            i += 1
            pos_norm_err = np.linalg.norm(err[:3])
            orient_norm_err = np.linalg.norm(err[3:])
            converged = pos_norm_err<eps_pos and orient_norm_err<eps_orient
            if converged or i >= IT_MAX:
                break
            if not i % 10:
                print('\n {}:'.format(i) +' final error: %s' % err.T + '\t pos.norm(error): %s' % pos_norm_err + '\t orient.norm(error): %s' % orient_norm_err)
            if plot:
                errs.append(norm(err))

        if converged:
            print("Convergence achieved!")
        else:
            print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

        print('\n {}:'.format(i) +' final error: %s' % err.T + '\t pos.norm(error): %s' % pos_norm_err + '\t orient.norm(error): %s' % orient_norm_err)

        if plot:
            plt.plot(errs)
            plt.show()
        return Q_i.copy()


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    pin_rob = PinRobot(True)

    # # Try out IK
    # home_new = np.array([np.pi/8, -0.4, 0.0, 3*np.pi/8, 0.0, 0.0, 0.0]).reshape(-1, 1)
    # home_new = np.random.rand(7,1)*np.pi
    # SE3_w_tcp = pin_rob.FK(home_new)
    # delta_R = R.from_euler("xyz", [0.1, 0.1, 0.1]).as_dcm()
    # # q_goal = pin_rob.ik_apollo(home_new, SE3_w_tcp.translation-0.3, SE3_w_tcp.rotation[:,[1,2,0]], plot=False)
    # q_goal = pin_rob.ik_apollo(home_new, SE3_w_tcp.translation-0.3, delta_R.dot(SE3_w_tcp.rotation), plot=False)
    # print(q_goal.T)
    # print()
    # print(home_new.T)


    home_pose = np.array([0.0, -0.1, -np.pi/6, 0.0, 0.0, np.pi/2, 0.0]).reshape(-1,1)
    home_pose = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    home_pose = np.array([0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)


    def print_FK(base, frame):
        print(base, frame) # base -> frame
        print(pin_rob.FK_f2f(home_pose, baseName=base, frameName=frame))

    # Find DH params
    if False:
        print_FK("R_SFE", "R_EB")  # Shoulder -> Elbow
        print_FK("R_EB", "R_WFE")  # Elbow -> Wrist
        print_FK("R_WFE", TCP)     # Wrist -> TCP
        print_FK(BASE, TCP)        # Base -> TCP



    print_FK("universe", BASE) # World -> TCP
    print_FK(BASE, "R_BASE")   # Base  -> TCP

    print_FK("universe", TCP) # World  -> TCP
    print_FK(BASE, TCP)       # R_Base -> TCP
    print_FK("R_BASE", TCP)   # R_Base -> TCP