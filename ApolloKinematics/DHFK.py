#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   DHFK.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   defines class for Denavit-Hartenberg forward kinematics
'''

import numpy as np
from Sets import ContinuousSet
from utilities import JOINTS_LIMITS, pR2T, invT
from math import sin, cos
np.set_printoptions(precision=4, suppress=True)


spi6 = 0.5
cpi6 = cos(np.pi/6)

class DH_revolut():
    """ Implementation of DH kinematics and other methods
        to compute FK, Jacobian and Hessian matrix at given joint configuraton.

        The main methods are unit-tested in:
            1. tests/check_jac.py  (test jacobian and hessian computations)
            2. tests/check_ik_real_dh.py (test get to (dh)base and get to (dh)TCP frame)
    """
    T_base_rbase = np.array([  # Sets the Arm to the correct(main) base frame
        [-spi6,  cpi6,  0.0, 0.0],
        [ cpi6,  spi6,  0.0, 0.0],
        [ 0.0,   0.0,  -1.0, 0.0],
        [ 0.0,   0.0,   0.0, 1.0]], dtype = 'float')

    T_rbase_dhbase = np.array([  # Sets the Arm to the right base frame
        [ 0.0, -1.0, 0.0, 0.0],
        [ 0.0,  0.0, 1.0, 0.0],
        [-1.0,  0.0, 0.0, 0.0],
        [ 0.0,  0.0, 0.0, 1.0]], dtype = 'float')
    T_dhtcp_tcp = np.array([[ 0.0,  0.0, -1.0, -0.04],  # uppword orientation(cup is up)
                            [-1.0,  0.0,  0.0,  0.0],
                            [ 0.0,  1.0,  0.0,  0.01],
                            [ 0.0,  0.0,  0.0,  1.0 ]], dtype='float')
                # = invT(np.array([[0.0, -1.0, 0.0,  0.0],
                #                  [0.0,  0.0, 1.0,  0.01],
                #                  [-1.0, 0.0, 0.0, -0.04],
                #                  [0.0,  0.0, 0.0,  1.0]], dtype='float'))
    class Joint():
        def __init__(self, a, alpha, d, theta, index, limit=(-np.pi, np.pi), vlimit=(-10., 10.), name="", offset=0.0):
            """ Class that keeps the DH parameters for a single joint.
            Args:
                a , alpha, d, theta (float):            DH params
                index, name (int, string):              index and name of the joint
                limit, vlimit (tuple(float, float)):    Angle and velocity limits
                offset (float):                         offset angle of the joint
            """
            self.a = a
            self.alpha = alpha
            self.d = d
            self.theta = theta
            self.name = name
            self.offset = offset
            self.index = index
            self.limit_range = ContinuousSet(limit[0], limit[1], False, False)
            self.vlimit_range = ContinuousSet(vlimit[0], vlimit[1], False, False)

        def __repr__(self):
            return '{}. {}: a({}), b({}), d({}), theta({})'.format(self.index, self.name, self.a, self.alpha, self.d, self.theta)

    def __init__(self):
        self.n_joints = 0
        self.joints = list()

    def joint(self, i):
        return self.joints[i]

    def add_joint(self, a, alpha, d, theta, limit=(-np.pi, np.pi), vlimit=(-10., 10.), name="", offset=0.0):
        self.joints.append(self.Joint(a, alpha, d, theta, self.n_joints, limit, vlimit, name, offset))
        self.n_joints += 1

    def getT(self, j, theta):
        """Returns the transformation matrix related to joint j and q_j=theta, according to the DH table.

        Args:
            j (Joint): joint
            theta (float): joint angle

        Returns:
            [np.array(4,4)]: j_T_j+1
        """
        c_th = cos(theta + j.theta + j.offset); s_th = sin(theta + j.theta + j.offset)
        if j.index==6:
            return np.array(
                [[s_th,   c_th,    0.0,  0.0],
                [-c_th,   s_th,    0.0,  0.0],
                [0.0,     0.0,     1.0,  j.d],
                [0.0,     0.0,     0.0,  1.0]], dtype='float').dot(self.T_dhtcp_tcp)
        else:
            sig = float(np.sign(j.alpha))
            return np.array(
                [[c_th,   0.0,      s_th*sig,  0.0],
                [s_th,   0.0,      -c_th*sig,  0.0],
                [0.0,    sig,       0.0,       j.d],
                [0.0,    0.0,       0.0,       1.0]], dtype='float')

    def FK(self, Q, rbase_frame=False):
        """ Computes the forward kinematics

        Args:
            Q (np.array(7,1)):  Joint configuration
            rbase_frame (bool): Whether we want to use the right base frame as reference frame.

        Returns:
            T0_7 (np.array(7,1)): homogenous transformation for the endeffector
        """
        Q = Q.copy().reshape(-1, 1)
        T0_7 = np.eye(4)
        for j, theta_j in zip(self.joints, Q):
            T0_7 = T0_7.dot(self.getT(j, theta_j))
        # dh_base -> r_base
        T0_7 = self.T_rbase_dhbase.dot(T0_7)
        if not rbase_frame:  # dont enter if we want fk in rbase frame
            # r_base -> base
            T0_7 = self.T_base_rbase.dot(T0_7)
        return T0_7

    def get_i_R_j(self, i, j, Qi_j):
        """ Computes the relative rotation between two joints

        Args:
            i, j (int, int): Joint indexes between which we compute the relative rotation
            Qi_j (np.array(j-i, 1)): angles of joints from i to j

        Returns:
            i_R_j (np.array(3, 3)): rotation matrix
        """
        # Qi_j: angles of joints from i to j
        assert i != j, 'i:{} and j:{} cannot be equal'.format(i,j)
        transpose_at_end = False
        if i>j:
            transpose_at_end = True
            a  = j
            j = i
            i = a

        i_R_j = np.eye(3)
        for joint, th_j in zip(self.joints[i:j], Qi_j):
            i_R_j = i_R_j.dot(self.getT(joint, th_j)[:3, :3])

        if transpose_at_end:
            i_R_j = i_R_j.T
        return i_R_j

    def get_i_T_j(self, i, j, Qi_j):
        """ Computes the relative transformation between two joints

        Args:
            i, j (int, int): Joint indexes between which we compute the relative rotation
            Qi_j (np.array(j-i, 1)): angles of joints from i to j

        Returns:
            i_T_j (np.array(4, 4)): homogenous tranformation matrix
        """
        assert i < j, 'i:{} and j:{} cannot be equal'.format(i,j)

        i_T_j = np.eye(4)
        for joint, th_j in zip(self.joints[i:j], Qi_j):
            i_T_j = i_T_j.dot(self.getT(joint, th_j))
        return i_T_j

    def get_goal_in_dh_base_frame(self, p_base_x, R_base_x):
        """ Makes up for a different base than the one used for DH model.
            Used to bring the homogenous transformation
            from the base(origin) to the (dh)base frame of DH.

        Args:
            p_base (np.array(3,1)): _description_
            R_base (np.array(3,3)): _description_

        Returns:
            R_dhbase_x, p_dhbase_x: rotation and position in (dh)base frame
        """
        T_base_x = pR2T(p_base_x, R_base_x).squeeze()
        # base-> rbase
        T_rbase_base = invT(self.T_base_rbase)
        T_rbase_x = T_rbase_base.dot(T_base_x)

        # rbase -> dh_base
        T_dhbase_rbase = invT(self.T_rbase_dhbase)
        T_dhbase_x = T_dhbase_rbase.dot(T_rbase_x)

        return T_dhbase_x[:3,3:4], T_dhbase_x[:3,:3]

    def get_goal_in_dhtcp_frame(self, p_0_tcp, R_0_tcp):
        """ Makes up for different tcp frame than the one used for DH model.
            Used to compute the (dh)TCP homogenous transformation from
            the real TCP(endeffector) homogenous transformation.

        Args:
            p_0_tcp (np.array(3,1)): position of TCP
            R_0_tcp (np.array(3,3)): rotation matrix of TCP

        Returns:
           T_0_dhtcp np.array(4,4): homogenous transformation matrix for (dh)TCP
        """
        T_0_tcp = pR2T(p_0_tcp, R_0_tcp).squeeze()
        T_0_dhtcp = T_0_tcp.dot(invT(self.T_dhtcp_tcp))  # T_0_7
        return T_0_dhtcp[:3,3:4], T_0_dhtcp[:3,:3]

    def i_J_j(self, i, j, Qi_j):
        """ Computes Jacobian between two joints

        Args:
            i, j (int, int): joint indexes
            Qi_j (np.array(j-i, 1)): : angles of joints from i to j

        Returns:
            i_J_j (np.array(6, j-i)): Jacobian matrix between joints i and j
        """
        # J = [zi(x)delta(pi), .., zk(x)delta(pk), .., zj(x)delta(pj)
        #      zi            , .., zk            , .., zj            ] where delta(pk) = pj -pk
        assert i < j, 'i:{} and j:{} cannot be equal'.format(i,j)
        N = j-i
        i_T_k = np.eye(4)
        z_s = [None]*(N+1); z_s[0] = i_T_k[:3, 2:3]
        p_s = np.zeros((3, N+1)); p_s[:,0] = i_T_k[:3, -1]
        for n, joint_k, th_k in zip(range(N), self.joints[i:j], Qi_j):
            i_T_k = i_T_k.dot(self.getT(joint_k, th_k))
            z_s[n+1]    = i_T_k[:3, 2:3]
            p_s[:, n+1] = i_T_k[:3, 3]

        J_s = []
        for n in range(N):
            J_s.append(np.cross(z_s[n], p_s[:, N:N+1]-p_s[:, n:n+1], axis=0))

        JP = np.hstack(J_s)
        JO = np.hstack(z_s[:-1])

        J = np.vstack([JP, JO])
        return J

    def J(self, q, rbase_frame=False):
        """ Computes the jacobian matrix given a joint configuration.
            Jacobian expresses the cartesian velocities caused by joint velocities.
            We want the cartesian velocities in the right base.
            In order to tranfsorm the jacobian in the right base frame we have to premultiply with
            [R_base_base_dh    0
            0                 R_base_base_dh]

        Args:
            q (np.array(7,1)): joint angles
            rbase_frame (bool): Whether we want to use the right base frame as reference frame.

        Returns:
            J (np.array(6,7)) : Jacobian matrix
        """


        # dh_base -> rbase
        T_base_dhbase = self.T_rbase_dhbase.copy()
        # rbase -> base
        if not rbase_frame:  # dont enter if we want to work with reference to rbase
            T_base_dhbase = self.T_base_rbase.dot(T_base_dhbase)

        TMP = np.eye(6)
        TMP[:3,:3] = T_base_dhbase[:3,:3]
        TMP[3:,3:] = T_base_dhbase[:3,:3]
        return TMP.dot(self.i_J_j(0, 7, q))

    def H(self, q, rbase_frame=False):
        """ Computes the Hessian matrix for a given joint configurtion.
            Implementation from paper: "A Systematic Approach to Computing the Manipulator Jacobian and Hessian
                                        using the Elementary Transform Sequence"
            Jesse Haviland, Peter Corke, Fellow, IEEE, https://arxiv.org/pdf/2010.08696.pdf

        Args:
            q (np.array(7,1)): joint angles
            rbase_frame (bool): Whether we want to use the right base frame as reference frame.

        Returns:
            H (np.array(6,7,7)): Hessian matrix
        """
        n = self.n_joints

        H = np.zeros((6, n, n))
        J0 = self.J(q, rbase_frame)

        for j in range(n):
            for i in range(j, n):
                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]
        return H

if __name__ == "__main__":
    from ApolloKinematics.utilities import R_joints
    pi2 = np.pi/2
    th3_offset = np.pi/6
    d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186
    a_s            = [0.0] * 7
    alpha_s        = [pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
    d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
    theta_s = [0.0, -pi2, pi2, 0.0, -pi2, 0.0, 0.0]
    offsets = [0.0, 0.0, th3_offset, 0.0, 0.0, 0.0, 0.0]

    # Create Robot
    my_fk_dh = DH_revolut()
    for a, alpha, d, theta, name, offset in zip(a_s, alpha_s, d_s, theta_s, R_joints, offsets):
        my_fk_dh.add_joint(a, alpha, d, theta, JOINTS_LIMITS[name], name, offset)



    # Joint configuration
    home_pose = np.array([0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    base_T_tcp = my_fk_dh.FK(home_pose, False)
    print(base_T_tcp)  # base_T_tcp
    print(my_fk_dh.get_i_T_j(0,7, home_pose))  # basedh_T_tcp
