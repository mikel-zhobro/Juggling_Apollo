import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from utilities import modrad, reduce_model
from settings import R_joints, L_joints, TCP, FILENAME, JOINTS_LIMITS

np.set_printoptions(precision=3, suppress=True)
pin.switchToNumpyMatrix()  # https://github.com/stack-of-tasks/pinocchio/issues/802

# inv kinematics params
IT_MAX = 10200
eps_pos    = 1e-3
eps_orient    = 1e-2
DT     = 10e-4
damp   = 1e-12
    
class PinRobot():
    def __init__(self, r_arm=True):
        """ Initialize pinocchio dependent models/variables: model, data, joint_states
            filename ([str]): path to the urdf file
        """
        self.joints_list = R_joints if r_arm else L_joints

        
        self.model = reduce_model(FILENAME, jointsToUse=self.joints_list)
        self.data = self.model.createData()  # information that changes according to joint configuration etc
                                             # Only used as internal state(still all functions should be called with certain joint_state as input)
        
        # Use "BASE" instead of "ORIGIN" as world coordinate frame
        self.SE3_world_origin = pin.SE3(np.eye(3), np.zeros((3,1)))
        self.SE3_world_origin = self.FK(pin.neutral(self.model), "BASE").inverse()

    def FK(self, q, frameName=TCP):
        """
        returns the SE3_world_frame(R,p) of frameName in world coordinates
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        return self.SE3_world_origin * pin.updateFramePlacement(self.model, self.data, frameId) # updates data and returns T_world_frame

    def J(self, q, frameName=TCP):
        """
        returns SE3(R,p) and J of TCP in world coordinates
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        SE3_world_tcp = pin.updateFramePlacement(self.model, self.data, frameId)  # computes frame relevant info
        J_local = pin.computeFrameJacobian(self.model, self.data, q, frameId)
        return J_local, self.SE3_world_origin*SE3_world_tcp

    def ik_apollo(self, Q_start, goal_p, goal_R=None, frameName=TCP, plot=False):
        """ Inverse Kinematics

        Args:
            goal_p ([float, float, float]): Endeffector xyz position in WORLD frame
            goal_R ([np.array((3,3))], optional): Endeffector's orientation in WORLD frame. If not specified it will be vertical.
            Q_i ([float]*N_JointUsed], optional): Whether to plot the error.
            plot (bool, optional): Whether to plot the error.
            frameName (str, optional): Name of TCP

        Returns:
            [list]: joint configuration to achieve desired endeffector position/orientation
        """
        
        # Desired TCP cartesian position
        goal_p = np.array(goal_p).reshape(3,1)
        goal_R = goal_R if goal_R is not None else np.eye(3)[:,[2,0,1]]
        SE3_world_goal = pin.SE3(goal_R, goal_p)
        print("GOAL")
        print(SE3_world_goal)
        
        def get_se3_error(SE3_world_tcp_i):
            dMi = SE3_world_goal.actInv(SE3_world_tcp_i)
            err = pin.log(dMi).vector
            return err


        i=0
        errs = []
        Q_i = Q_start.copy()
        while True:

            # Calc T_world_tcp and J_world_tcp for the new joint_states
            J_world_tcp, SE3_world_tcp_i  = self.J(Q_i, frameName)

            # Calc cartesian errors
            err = get_se3_error(SE3_world_tcp_i)
            
            # 1 Calc qoint velocities
            qv = - J_world_tcp.T.dot(solve(J_world_tcp.dot(J_world_tcp.T) + damp * np.eye(6), err))
            
            # 2
            # J_invj = np.linalg.pinv(J_world_tcp)
            # qv = -np.matmul(J_invj, err)
            
            # Update joint_states
            Q_i = pin.integrate(self.model, Q_i, qv*DT)
            Q_i = modrad(Q_i)


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
    pin_rob = PinRobot()
    home_new = np.array([np.pi/8, -0.4, 0.0, 3*np.pi/8, 0.0, 0.0, 0.0]).reshape(-1, 1)
    SE3_w_tcp = pin_rob.FK(home_new)
    q_goal = pin_rob.ik_apollo(home_new, SE3_w_tcp.translation-0.3, SE3_w_tcp.rotation[:,[1,2,0]], plot=True)
    
    print(q_goal.T)
    print()
    print(home_new.T)
    

    # pin_rob = PinRobot(False)
    # home_pose = np.array([np.pi/8, 0.0, 0.0, 3*np.pi/8, 0.0, 0.0, 0.0]).reshape(-1,1)
    # home_pose = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)

    # frameId = pin_rob.model.getFrameId(TCP)
    # pin.forwardKinematics(pin_rob.model, pin_rob.data, home_pose)
    # pin.updateFramePlacement(pin_rob.model, pin_rob.data, frameId)




    # q = pin.randomConfiguration(pin_rob.model,-np.ones(pin_rob.model.nq),np.ones(pin_rob.model.nq))
    # v = np.random.rand(pin_rob.model.nv)

    # frame_name = TCP
    # frame_id = pin_rob.model.getFrameId(frame_name)
    # frame = pin_rob.model.frames[frame_id]
    # frame_placement = frame.placement
    # parent_joint = frame.parent

    # # 0
    # # pin_rob.fk(q, TCP)
    # pin.forwardKinematics(pin_rob.model, pin_rob.data, q)
    # pin.updateFramePlacement(pin_rob.model, pin_rob.data, frame_id) # updates data
    # # pin.updateFramePlacements(pin_rob.model,pin_rob.data)
    # J = pin.computeFrameJacobian(pin_rob.model, pin_rob.data, q, frame_id)
    
    # print(J)
    # print()
    
    # # 1
    # pin.forwardKinematics(pin_rob.model,pin_rob.data,q,v)
    # pin.computeJointJacobians(pin_rob.model,pin_rob.data,q)
    # pin.updateFramePlacement(pin_rob.model,pin_rob.data, frame_id)
    # frame_J = pin.getFrameJacobian(pin_rob.model, pin_rob.data, frame_id, pin.ReferenceFrame.WORLD)
    # J_dot_v = pin.Motion(frame_J.dot(v).reshape(6,1))
    # print(frame_J)