#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ApolloKinematics.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   defines class with Apollo's kinematics(fk, ik, seq_ik, seq_fk, etc.)
'''

import numpy as np
import matplotlib.pyplot as plt

from DHFK import DH_revolut
from utilities import R_joints, L_joints, JOINTS_LIMITS, JOINTS_V_LIMITS
from utilities import invT
from PinFK import PinRobot
from AnalyticalIK import IK_anallytical, IK_heuristic2, IK_heuristic3, IK_find_psi_and_GCs


class ApolloArmKinematics():
    def __init__(self, r_arm=True, noise=None):
        self.r_arm = r_arm
        self.dh_rob = self.init_dh(noise)
        try:
            self.pin_rob = PinRobot(r_arm=r_arm)
        except:
            pass

    def init_dh(self, noise=None):
        """ Initializes the dh model.
        Args:
            noise (float, optional): If not none gives the amplitude of distance noise for the 4 distances of LWR. Defaults to None.
                                     Allow noiy parameters of forward kinematics
        """
        joints2Use = R_joints if self.r_arm else L_joints
        pi2 = np.pi/2
        d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186
        a_s            = [0.0] * 7
        alpha_s        = [pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
        d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
        theta_s = [0.0, -pi2, pi2, 0.0, -pi2, 0.0, 0.0]
        offsets = [0.0, 0.0, np.pi/6, 0.0, 0.0, 0.0, 0.0]

        # Create DH Robot
        n_ds = np.zeros(7)
        if noise is not None:
            for i in [0, 2, 4, 6]:
                n_ds[i] = noise*(np.random.rand()-0.5)
            print('Noise', n_ds)
        dh_rob = DH_revolut()
        for a, alpha, d, theta, name, offset, n_d in zip(a_s, alpha_s, d_s, theta_s, joints2Use, offsets, n_ds):
            dh_rob.add_joint(a, alpha, d+n_d, theta, JOINTS_LIMITS[name], JOINTS_V_LIMITS[name], name, offset)
        if noise is not None:
            print('Noise', dh_rob.joints)
        return dh_rob

    def FK(self, q):
        """
        returns the Transformationsmatrix base_T_tcp
        """
        # return self.pin_rob.FK(q.reshape(7, 1)).homogeneous
        return self.dh_rob.FK(q)

    def J(self, q):
        """
        returns the Transformationsmatrix base_T_tcp
        """
        return self.dh_rob.J(q.reshape(7, 1))

    def seqFK(self, qs):
        assert len(qs.shape)>2, "make sure you give a seq of joint states as input"
        return np.array([self.FK(q) for q in qs]).reshape(-1, 4, 4)

    def IK(self, T_d, q_init=None, for_seqik=False, considered_joints=list(range(7))):
        # Returns joint configuration that correspond to the IK solutions with biggest PSI feasible set
        # PSI is choosen from the middle of the set
        if q_init is None:
            GC2, GC4, GC6, feasible_set, solu =  IK_heuristic3(p07_d=T_d[:3, 3:4], R07_d=T_d[:3, :3], DH_model=self.dh_rob, considered_joints=considered_joints) # decide branch of solutions(GC2=1)
            assert not feasible_set.empty, "Not possible to calculate IK"
            feasible_set = feasible_set.max_range()
            psi = feasible_set.middle                     # Choose psi for start configuration
            q_joints = solu(psi)                          # Start configuration
        else:
           psi, GC2, GC4, GC6, feasible_set, solu = IK_find_psi_and_GCs(p07_d=T_d[:3, 3:4], R07_d=T_d[:3, :3], q_init=q_init, DH_model=self.dh_rob) #, considered_joints=considered_joints) # decide branch based on q_init
           q_joints = solu(psi)

        if for_seqik:
            return q_joints, GC2, GC4, GC6, psi, solu, feasible_set
        else:
            return q_joints

    def seqIK(self, T_dhTCP_traj, q_init=None, considered_joints=list(range(7)), verbose=False):
        """
        Args:
            T_traj_TCP           ([np.array((N, 4,4))]): relative movements from start_position
            position_traj        ([np.array((N, 3))]): relative movements from start_position
            thetas_traj          ([np.array(N)])     : relative rotations around z axis from start orientation of TCP (x-down, y-left, z-forward) <- upright position
            T_start              ([np.array((4, 4))]): decides start position/orientation from where everything unfolds
            verbose (bool, optional)                 : Ploting for verbose reasons. Defaults to False.

        Returns:
            [np.array((N, 7))]: Joint trajectories
        """
        mu = 0.02


        # Init lists
        N = T_dhTCP_traj.shape[0]
        joint_trajs = np.zeros((N,7,1))
        psis = np.zeros((N))
        psi_mins = np.zeros((N))
        psi_maxs = np.zeros((N))

        # Find the solution branch we shall follow in this sequence and starting psi
        q_joint_state_start, GC2, GC4, GC6, psi, _, _ = self.IK(T_dhTCP_traj[0], for_seqik=True, considered_joints=considered_joints)   # Start configuration
        # Add start configuration
        joint_trajs[0] = q_joint_state_start
        for i, T_i in enumerate(T_dhTCP_traj[1:]): # position_traj and thetas should start from 0
            # Calc IK for new pose
            solu, feas_set, root_funcs = IK_anallytical(T_i[:3,3:4], T_i[:3,:3], self.dh_rob, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False, considered_joints=considered_joints)

            # Choose psi for the new joint configuration
            feas_set = feas_set.range_of(psi)
            psi = (1-mu)*psi + mu*feas_set.middle  # TODO: Find range closest to previous one, not just max_range

            # Fill verbose lists
            joint_trajs[i+1] = solu(psi)
            psis[i+1] = psi
            psi_mins[i+1] = feas_set.a
            psi_maxs[i+1] = feas_set.b

        if verbose:
            self.plot(joint_trajs, psis, psi_mins, psi_maxs)

        return joint_trajs, q_joint_state_start, (psis, psi_mins, psi_maxs)

    def transform_in_dh_frames(self, T_dhTCP_TCP, T_TCP):
        # T can be a trajectory of homogenous transformations (N, 4, 4) or one single homogenous transformation
        # T is gives T_base_TCP, i.e. the transformation for the TCP. Where the connection to the dhTCP is given by T_dhTCP_TCP.
        T_shape = T_TCP.shape
        TT = T_TCP.reshape(-1, 4, 4 ).copy()
        T_TCP_dhTCP = invT(T_dhTCP_TCP)
        for i, Ti in enumerate(T_TCP):
            TT[i] =  Ti.dot(T_TCP_dhTCP)
        return TT.reshape(T_shape)

    @property
    def limits(self):
        return [j.limit_range for j in self.dh_rob.joints]

    @property
    def vlimits(self):
        return [j.vlimit_range for j in self.dh_rob.joints]

    def plot(self, joint_trajs=None, psis=None, psi_mins=None, psi_maxs=None, dt=1, rad=False):
        if psis is None:
            self.plot_joints(joint_trajs, dt, rad=rad)
        elif joint_trajs is None:
            self.plot_psis(psis, psi_mins, psi_maxs, dt=dt, rad=rad)
        else:
            fig, axs = plt.subplots(8,1, figsize=(16,12))
            self.plot_psis(psis, psi_mins, psi_maxs, dt=dt, ax=axs[0], rad=rad)
            self.plot_joints(joints_traj=joint_trajs, dt=dt, axs=axs[1:], rad=rad)
            plt.show()


    def plot_psis(self, psis, psi_mins, psi_maxs, dt=1.0, ax=None, rad=False):
        noax = ax is None
        if noax:
            fig, ax = plt.subplots(1, figsize=(16,12))

        fac = 1.0 if rad else 180.0/np.pi
        times = np.arange(0, len(psis)) * dt
        ax.plot(times, fac*psis, '-', color='k', label="psi")
        ax.fill_between(times, fac*psi_mins, fac*psi_maxs, color='gray', alpha=0.2, label="psimin<->psimax")
        ax.legend()

        if noax:
            plt.show()
        return ax

    def plot_joints(self, joints_traj, dt=1.0, axs=None, rad=False):
        colors = ["r", 'g', 'b', 'k', 'c', 'm', 'y']
        noax = axs is None
        if noax:
            fig, axs = plt.subplots(7,1, figsize=(16,12))

        fac = 1.0 if rad else 180.0/np.pi
        times = np.arange(0,joints_traj.shape[0]) * dt

        lines = [axs[iii].plot(times, fac* joints_traj.reshape(-1, 7)[:,iii], color=colors[iii], label=r"$\theta_{}$".format(iii+1))[0] for iii in range(7)]
        for iii in range(7):  # limits
            if True:
                axs[iii].axhline(fac*self.limits[iii].a, color=lines[iii].get_color(), linestyle='dashed')
                axs[iii].axhline(fac*self.limits[iii].b, color=lines[iii].get_color(), linestyle='dashed')
                axs[iii].axhspan(fac*self.limits[iii].a, fac*self.limits[iii].b, color=lines[iii].get_color(), alpha=0.3, label='feasible set')
                axs[iii].legend(loc=1)

        if noax:
            plt.show()
        return axs