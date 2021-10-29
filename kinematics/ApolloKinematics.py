import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sin, cos, acos, sqrt, atan2, asin

from DH import DH_revolut
from utilities import R_joints, L_joints, JOINTS_LIMITS
from utilities import skew, vec, clip_c
from fk_pin_local import PinRobot
from AnalyticalIK import IK_anallytical, IK_heuristic2

class ApolloArmKinematics():
    def __init__(self, r_arm=True):
        self.r_arm = r_arm
        self.dh_rob = self.init_dh()
        self.pin_rob = PinRobot(r_arm=r_arm)

    def init_dh(self):
        joints2Use = R_joints if self.r_arm else L_joints
        pi2 = np.pi/2
        d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186
        a_s            = [0.0] * 7
        alpha_s        = [pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
        d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
        theta_s = [0.0, -pi2, pi2, 0.0, -pi2, 0.0, 0.0]
        offsets = [0.0, 0.0, np.pi/6, 0.0, 0.0, 0.0, 0.0]

        # Create DH Robot
        dh_rob = DH_revolut()
        for a, alpha, d, theta, name, offset in zip(a_s, alpha_s, d_s, theta_s, joints2Use, offsets):
            dh_rob.add_joint(a, alpha, d, theta, JOINTS_LIMITS[name], name, offset)
        return dh_rob

    def FK(self, q):
        """
        returns the Transformationsmatrix base_T_tcp
        """
        return self.pin_rob(q).homogeneous

    def IK(self,p07_d, R07_d, DH_model, GC2=1.0, GC4=1.0, GC6=1.0):
        return IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6)

    def seqIK(self, position_traj, thetas_traj, q_joints_state_start, verbose=False):
        """[summary]

        Args:
            position_traj        ([np.array((N, 3))]): relative movements from start_position
            thetas_traj          ([np.array(N)])     : relative rotations around z axis from start orientation(x-down, y-right, z-forward)
            q_joints_state_start ([np.array((7, 1))]): decides start position/orientation from where everything unfolds
            verbose (bool, optional)                 : Ploting for verbose reasons. Defaults to False.

        Returns:
            [np.array((N, 7))]: Joint trajectories
        """
        mu = 0.3
        # Find a solution satisfying the constraints for start pose
        T_start = self.dh_rob.FK(q_joints_state_start)
        GC2, GC4, GC6, feasible_set, solu =  IK_heuristic2(p07_d=T_start[:3,3:4], R07_d=T_start[:3,:3], DH_model=self.dh_rob) # decide branch of solutions

        # Choose psi for start configuration
        feasible_set = feasible_set.max_range()
        assert not feasible_set.empty, "Not possible to calculate IK"
        psi = feasible_set.middle
        psi_min = feasible_set.a
        psi_max = feasible_set.b
        q_joint_state_i = solu(psi)

        # Init lists
        N = position_traj.shape[0]
        psis = np.zeros((N))
        psi_mins = np.zeros((N))
        psi_maxs = np.zeros((N))
        joints_traj = np.zeros((N,) + q_joint_state_i.shape)
        joints_traj[0] = q_joint_state_i
        psis[0] = psi
        psi_mins[0] = psi_min
        psi_maxs[0] = psi_max

        R_start = np.array([[0.0, -1.0, 0.0,],  # uppword orientation(cup is up)
                            [0.0,  0.0, 1.0,],
                            [-1.0, 0.0, 0.0,]], dtype='float')
        p_start = T_start[:3, 3:4]

        for i, (pos_goal_i, theta_i) in enumerate(zip(position_traj[1:], thetas_traj[1:])): # position_traj and thetas should start from 0

            # Compute next orientation from theta
            s = sin(theta_i); c = cos(theta_i)
            start_R_i = np.array([[c,  -s,   0.0,],
                                  [s,   c,   0.0,],
                                  [0.0, 0.0, 1.0,]], dtype='float')
            R_i = R_start.dot(start_R_i)

            # Calc IK for new pose
            solu, feas_set = IK_anallytical(p_start+pos_goal_i.reshape(3,1), R_i, self.dh_rob, GC2=GC2, GC4=GC4, GC6=GC6, verbose=True)

            # Choose psi for the new joint configuration
            feas_set = feas_set.range_of(psi)
            psi = (1-mu)*psi + mu*feas_set.middle  # TODO: Find range closest to previous one, not just max_range

            # Fill verbose lists
            joints_traj[i+1,:] = solu(psi)
            psis[i+1] = psi
            psi_mins[i+1] = feas_set.a
            psi_maxs[i+1] = feas_set.b

        if verbose:
            self.plot(psis, psi_mins, psi_maxs, joints_traj)
        return joints_traj

    def plot(self, psis, psi_mins, psi_maxs, joint_trajs, dt=1):
        times = np.arange(0, len(psis)) * dt
        plt.figure()
        plt.plot(times, psis, '-', color='k', label="psi")
        plt.fill_between(times, psi_mins, psi_maxs, color='gray', alpha=0.2, label="psimin<->psimax")
        plt.legend()


        plt.figure()
        plt.plot(times, joint_trajs.copy().squeeze())
        plt.show()




if __name__ == "__main__":
    dt = 0.06
    myApollo = ApolloArmKinematics()
    home_pose = np.array([0.0, -1.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

    N = 12
    thetas_traj = np.zeros((N,))
    position_traj = np.zeros((N, 3))
    position_traj[:, 0] = np.sin(np.arange(N)*dt)

    myApollo.seqIK(position_traj, thetas_traj, home_pose, verbose=True)
