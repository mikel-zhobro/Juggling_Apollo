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
    
    def IK_heuristic2(self, p07_d, R07_d):
        """
        Does a search over all branches and returns the one with the biggest feasible set.
        """
        solution_function, feasible_set = IK_heuristic2(p07_d, R07_d, self.dh_rob)
        psi_middle = feasible_set.middle
        psi_min = feasible_set.a
        psi_max = feasible_set.b
        assert not feasible_set.empty, "Not possible to calculate IK"
        return solution_function(psi_middle), psi_min, psi_max
    
    def seqIK(self, position_traj, thetas, q_joints_state_start):
        N = position_traj.shape[0]
        psi_mins = np.zeros((N))
        psi_maxs = np.zeros((N))
        joints_traj = np.zeros((N, q_joints_state_start.size))
        joints_traj[0] = q_joints_state_start
        psi_mins[0] = q_joints_state_start-1.0
        psi_maxs[0] = q_joints_state_start+1.0

        q_joint_state_i = q_joints_state_start.copy()
        for i, (pos_goal_i, theta_i) in enumerate(zip(position_traj[1:], thetas[1:])):
            s = sin(theta_i); c = cos(theta_i)
            R_goal_i = np.array([[ s,   c,  0.0,],
                                [0.0, 0.0, 1.0,],
                                [-c,   s,  0.0,]], dtype='float')
            q_joint_state_i, psi_min, psi_max = self.IK(pos_goal_i.reshape(3,1), R_goal_i, q_joint_state_i.copy(), verbose=False)
            joints_traj[i+1,:] = q_joint_state_i.T
            psi_mins[i+1] = psi_min
            psi_maxs[i+1] = psi_max

        return joints_traj
        
    def CartesianMinJerk2JointSpace(self, position_traj, thetas, q_joints_state_start):
        """ Calculates the joint space trajectories that correspond to cartesian trajectories

        Args:
            position_traj ([np.array(dtype='float')]): [N,3] a xyz relative trajectory of length N from the start position specified by q_joints_state_start
            thetas ([np.array(dtype='float')]): [N,] describing the orientation of the cup in the 2D plane(deviation from upright position)
                                                z_cup = [0, -1, 0] is constant as we want to rotate only in the 2D plane
                                                R = [[ s, 0, c],
                                                    [ 0,-1, 0],
                                                    [-c, 0, s]]
            q_joints_state_start ([type]): starting joint configuration which should correspond to the first pose
        """
        # 1. Make sure the home position of the hand matches the start theta of the trajectory
        base_T_tcp_start = self.FK(q_joints_state_start.copy())
        s = sin(thetas[0]); c = cos(thetas[0])
        R_goal_start = np.array([[ s,   c,  0.0,],
                                [0.0, 0.0, 1.0,],
                                [-c,   s,  0.0,]], dtype='float')
        assert np.allclose(base_T_tcp_start[:3, :3], R_goal_start), "Home position must match the start theta of the trajectory"
        assert np.allclose(base_T_tcp_start[:3, -1], position_traj[0,:]), "Home position must match the start position of the trajectory"
        ## If needed to corrects
        ## a. Update joint_start so that the home position/orientation matches the first position/orientation of the trajectory
        # q_joints_state_start = IK(base_T_tcp_start[:3, 3:], R_goal_start, q_joints_state_start.copy())

        N = position_traj.shape[0]
        psi_mins = np.zeros((N))
        psi_maxs = np.zeros((N))
        joints_traj = np.zeros((N, q_joints_state_start.size))
        joints_traj[0] = q_joints_state_start
        psi_mins[0] = q_joints_state_start-1.0
        psi_maxs[0] = q_joints_state_start+1.0

        q_joint_state_i = q_joints_state_start.copy()
        for i, (pos_goal_i, theta_i) in enumerate(zip(position_traj[1:], thetas[1:])):
            s = sin(theta_i); c = cos(theta_i)
            R_goal_i = np.array([[ s,   c,  0.0,],
                                [0.0, 0.0, 1.0,],
                                [-c,   s,  0.0,]], dtype='float')
            q_joint_state_i, psi_min, psi_max = self.IK(pos_goal_i.reshape(3,1), R_goal_i, q_joint_state_i.copy(), verbose=False)
            joints_traj[i+1,:] = q_joint_state_i.T
            psi_mins[i+1] = psi_min
            psi_maxs[i+1] = psi_max
        return joints_traj
    

    def plot(self, psi_mins, psi_maxs, dt=1):
        times = np.arange(0, len(psi_maxs)) * dt
        plt.plot(times, (psi_mins+psi_maxs)/2, '-', color='k', label="psi")
        plt.fill_between(times, psi_mins, psi_maxs, color='gray', alpha=0.2, label="psimin-psimax")
        plt.legend()
        plt.show()
       
       
       

if __name__ == "__main__":
    myApollo = ApolloArmKinematics()