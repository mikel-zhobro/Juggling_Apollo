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
        return self.pin_rob.FK(q).homogeneous
        # return self.dh_rob.FK(q)
    
    def IK(self,p07_d, R07_d, DH_model, GC2=1.0, GC4=1.0, GC6=1.0):
        return IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6)

    def seqIK(self, position_traj, thetas_traj, T_start, verbose=False):
        """[summary]

        Args:
            position_traj        ([np.array((N, 3))]): relative movements from start_position
            thetas_traj          ([np.array(N)])     : relative rotations around z axis from start orientation(x-down, y-right, z-forward)
            q_joints_state_start ([np.array((7, 1))]): decides start position/orientation from where everything unfolds
            verbose (bool, optional)                 : Ploting for verbose reasons. Defaults to False.

        Returns:
            [np.array((N, 7))]: Joint trajectories
        """
        mu = 0.01
        # Find the solution branch we shall follow in this sequence and starting psi
        R_start = T_start[:3, :3]
        p_start = T_start[:3, 3:4]
        GC2, GC4, GC6, feasible_set, solu =  IK_heuristic2(p07_d=p_start, R07_d=R_start, DH_model=self.dh_rob) # decide branch of solutions
        assert not feasible_set.empty, "Not possible to calculate IK"
        feasible_set = feasible_set.max_range()
        psi = feasible_set.middle     # Choose psi for start configuration
        q_joint_state_i = solu(psi)   # Start configuration

        # Init lists
        N = position_traj.shape[0]
        joints_traj = np.zeros((N,) + q_joint_state_i.shape)
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

            # Calc IK for new pose
            solu, feas_set = IK_anallytical(p_start+pos_goal_i.reshape(3,1), R_i, self.dh_rob, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)

            # Choose psi for the new joint configuration
            feas_set = feas_set.range_of(psi)
            psi = (1-mu)*psi + mu*feas_set.middle  # TODO: Find range closest to previous one, not just max_range

            # Fill verbose lists
            joints_traj[i] = solu(psi)
            psis[i] = psi
            psi_mins[i] = feas_set.a
            psi_maxs[i] = feas_set.b

        if verbose:
            self.plot(psis, psi_mins, psi_maxs, joints_traj)
        return joints_traj, q_joint_state_i
    
    def plot(self, psis, psi_mins, psi_maxs, joint_trajs, dt=1):
        times = np.arange(0, len(psis)) * dt
        fig, axs = plt.subplots(2,1)

        axs[0].plot(times, psis, '-', color='k', label="psi")
        axs[0].fill_between(times, psi_mins, psi_maxs, color='gray', alpha=0.2, label="psimin<->psimax")
        axs[0].legend()
        
        
        js = axs[1].plot(times[1:], 180.0/np.pi*joint_trajs[1:,:].copy().squeeze())
        axs[1].legend(js, [r'$\theta_{}$'.format(i+1) for i in range(7)])
        plt.show()




if __name__ == "__main__":
    from apollo_interface.Apollo_It import MyApollo, plot_simulation
    dt = 0.004      # [s] discretization time step size
    home_pose = np.array([0.0, -1.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

    r_arm_interface = MyApollo(r_arm=True)
    myApollo = ApolloArmKinematics()
    
    N = 5000
    T_start = myApollo.dh_rob.FK(home_pose)
    T_start[:3, :3] = np.array([[0.0, -1.0, 0.0,],  # uppword orientation(cup is up)
                                [0.0,  0.0, 1.0,],
                                [-1.0, 0.0, 0.0,]], dtype='float')

    thetas_traj = 0.0* (np.cos(np.arange(N)*dt) * np.pi/4 - np.pi/4)
    position_traj = np.zeros((N, 3))  # (x-left, y-forward, z-up)
    position_traj[:, 0] = -np.sin(np.arange(N)*dt)*0.1
    # position_traj[:, 1] = np.sin(np.arange(N)*dt)*0.1
    # position_traj[:, 2] = np.sin(np.arange(N)*dt)*0.1
    
    joints_traj, q_start = myApollo.seqIK(position_traj, thetas_traj, T_start, verbose=True)
    
    
    r_arm_interface.go_to_home_position(q_start, 2000)
    for i in range(N):
        q_i = joints_traj[i,:].reshape(-1, 1)
        # print(np.linalg.norm(position_traj[i] + T_start[:3, -1] - myApollo.FK(q_i)[:3, -1] ))
        r_arm_interface.go_to_posture_array(joints_traj[i], int(dt*1000), False)
        
    r_arm_interface.go_to_home_position(q_start, 2000)
    
    
