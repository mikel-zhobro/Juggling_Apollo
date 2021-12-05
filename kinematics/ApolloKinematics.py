import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, acos, sqrt, atan2, asin

from DH import DH_revolut
from utilities import R_joints, L_joints, JOINTS_LIMITS
from utilities import skew, vec, clip_c
from fk_pin_local import PinRobot
from AnalyticalIK import IK_anallytical, IK_heuristic2, IK_heuristic3

class ApolloArmKinematics():
    def __init__(self, r_arm=True, noise=None):
        self.r_arm = r_arm
        self.dh_rob = self.init_dh(noise)
        try:
            self.pin_rob = PinRobot(r_arm=r_arm)
        except:
            pass

    def init_dh(self, noise=None):
        """[summary]

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
        n_ds = [0.0]*7
        if noise is not None:
            for i in [0, 2, 4, 6]:
                n_ds[i] = noise*(np.random.rand()-0.5)
        dh_rob = DH_revolut()
        for a, alpha, d, theta, name, offset, n_d in zip(a_s, alpha_s, d_s, theta_s, joints2Use, offsets, n_ds):
            dh_rob.add_joint(a, alpha, d+n_d, theta, JOINTS_LIMITS[name], name, offset)
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

    def IK(self,p07_d, R07_d, GC2=1.0, GC4=1.0, GC6=1.0):
        return IK_anallytical(p07_d, R07_d, self.dh_rob, GC2=GC2, GC4=GC4, GC6=GC6)

    def IK_best(self, T_d, for_seqik=False):
        # Returns joint configuration that correspond to the IK solutions with biggest PSI feasible set
        # PSI is choosen from the middle of the set
        GC2, GC4, GC6, feasible_set, solu =  IK_heuristic3(p07_d=T_d[:3, 3:4], R07_d=T_d[:3, :3], DH_model=self.dh_rob) # decide branch of solutions
        assert not feasible_set.empty, "Not possible to calculate IK"
        feasible_set = feasible_set.max_range()
        psi = feasible_set.middle                     # Choose psi for start configuration
        q_joints = solu(feasible_set.a+1e-4)          # Start configuration
        q_joints = solu(psi)                          # Start configuration

        if for_seqik:
            return q_joints, GC2, GC4, GC6, psi
        else:
            return q_joints

    def seqIK(self, position_traj, thetas_traj, T_start, verbose=False):
        """[summary]

        Args:
            position_traj        ([np.array((N, 3))]): relative movements from start_position
            thetas_traj          ([np.array(N)])     : relative rotations around z axis from start orientation of TCP (x-down, y-right, z-forward)
            T_start              ([np.array((4, 4))]): decides start position/orientation from where everything unfolds
            verbose (bool, optional)                 : Ploting for verbose reasons. Defaults to False.

        Returns:
            [np.array((N, 7))]: Joint trajectories
        """
        assert np.all(position_traj[0] == 0.0) and thetas_traj[0]==0.0, "Make sure both position_traj&thetas_traj start with 0s."
        mu = 0.02
        # Find the solution branch we shall follow in this sequence and starting psi
        R_start = T_start[:3, :3]
        p_start = T_start[:3, 3:4]
        q_joint_state_start, GC2, GC4, GC6, psi = self.IK_best(T_start, for_seqik=True)   # Start configuration

        # Init lists
        N = position_traj.shape[0]
        cartesian_traj = np.zeros((N, 4, 4))
        joint_trajs = np.zeros((N,) + q_joint_state_start.shape)
        psis = np.zeros((N))
        psi_mins = np.zeros((N))
        psi_maxs = np.zeros((N))

        # Add start configuration
        for i, (pos_goal_i, theta_i) in enumerate(zip(position_traj, thetas_traj)): # position_traj and thetas should start from 0

            # Compute next orientation from theta
            s = sin(theta_i); c = cos(theta_i)
            start_R_i = np.array([[c,  -s,   0.0,],
                                  [s,   c,   0.0,],
                                  [0.0, 0.0, 1.0,]], dtype='float')
            R_i = R_start.dot(start_R_i)
            p_i = p_start+pos_goal_i.reshape(3,1)


            cartesian_traj[i, :3,:3] = R_i
            cartesian_traj[i, :3,3:4] = p_i
            # Calc IK for new pose
            solu, feas_set = IK_anallytical(p_i, R_i, self.dh_rob, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)

            # Choose psi for the new joint configuration
            feas_set = feas_set.range_of(psi)
            psi = (1-mu)*psi + mu*feas_set.middle  # TODO: Find range closest to previous one, not just max_range

            # Fill verbose lists
            joint_trajs[i] = solu(psi)
            psis[i] = psi
            psi_mins[i] = feas_set.a
            psi_maxs[i] = feas_set.b

        if verbose:
            self.plot(joint_trajs, psis, psi_mins, psi_maxs)

        return cartesian_traj, joint_trajs, q_joint_state_start, (psis, psi_mins, psi_maxs)

    @property
    def limits(self):
        return [j.limit_range for j in self.dh_rob.joints]

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


if __name__ == "__main__":
    from apollo_interface.Apollo_It import ApolloInterface
    dt = 0.004      # [s] discretization time step size
    home_pose = np.array([0.0, -1.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

    rArmInterface = ApolloInterface(r_arm=True)
    rArmKinematics = ApolloArmKinematics(r_arm=True)

    N = 200
    T_start = rArmKinematics.FK(home_pose)
    T_start[:3, :3] = np.array([[0.0, -1.0, 0.0,],  # uppword orientation(cup is up)
                                [0.0,  0.0, 1.0,],
                                [-1.0, 0.0, 0.0,]], dtype='float')

    thetas_traj = 0.0* (np.cos(np.arange(N)*dt) * np.pi/4 - np.pi/4)
    position_traj = np.zeros((N, 3))  # (x-left, y-forward, z-up)
    position_traj[:, 0] = -np.sin(np.arange(N)*dt)*0.1
    # position_traj[:, 1] = np.sin(np.arange(N)*dt)*0.1
    # position_traj[:, 2] = np.sin(np.arange(N)*dt)*0.1

    cartesian_traj_des, joints_traj, q_start, _ = rArmKinematics.seqIK(position_traj, thetas_traj, T_start, verbose=True)


    rArmInterface.go_to_home_position(q_start, 2000)
    for i in range(N):
        q_i = joints_traj[i,:].reshape(-1, 1)
        rArmInterface.go_to_posture_array(joints_traj[i], int(dt*1000), False)
    rArmInterface.go_to_home_position(q_start, 2000)


    # plt.plot([np.linalg.norm(position_traj[i] + T_start[:3, -1] - rArmKinematics.FK(joints_traj[i])[:3, -1] ) for i in range(N)])
    # plt.title("ERROR on IK")
    # plt.show()
