from numpy.core.defchararray import join
import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from utilities import modrad, reduce_model
from settings import R_joints, L_joints, TCP, WORLD, BASE, FILENAME, JOINTS_LIMITS

np.set_printoptions(precision=4, suppress=True)
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

        # Use "BASE" instead of 'universe' as base coordinate frame (Set the BASE Frame)
        self.SE3_base_origin = self.FK_f2f(pin.neutral(self.model), BASE, WORLD).inverse()

    def limit_joints(self, Q):
        for i, name in enumerate(self.joints_list):
            Q[i, 0] = np.clip(Q[i, 0], *JOINTS_LIMITS[name])
        return Q

    def frames(self):
        pass

    def FK(self, q, frameName=TCP):
        """
        returns the SE3_base_frame(R,p) of frameName in base coordinates
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        return self.SE3_base_origin * pin.updateFramePlacement(self.model, self.data, frameId) # updates data and returns T_base_frame

    def FK_f2f(self, q, baseName=BASE, frameName=TCP):
        """
        returns the SE3_baseName_frameName(R,p) of frameName in baseName coordinates
        """
        baseId = self.model.getFrameId(baseName)
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        T_origin_base = pin.updateFramePlacement(self.model, self.data, baseId)
        T_origin_frame = pin.updateFramePlacement(self.model, self.data, frameId)
        return T_origin_base.inverse() * T_origin_frame # returns T_baseName_frameName

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
            # Q_i = self.limit_joints(Q_i)

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

    # # Try out IK
    # home_new = np.array([np.pi/8, -0.4, 0.0, 3*np.pi/8, 0.0, 0.0, 0.0]).reshape(-1, 1)
    # home_new = np.random.rand(7,1)*np.pi
    # SE3_w_tcp = pin_rob.FK(home_new)
    # q_goal = pin_rob.ik_apollo(home_new, SE3_w_tcp.translation-0.3, SE3_w_tcp.rotation[:,[1,2,0]], plot=False)
    # print(q_goal.T)
    # print()
    # print(home_new.T)


    # pin_rob = PinRobot(False)
    # home_pose = np.array([np.pi/8, 0.0, 0.0, 3*np.pi/8, 0.0, 0.0, 0.0]).reshape(-1,1)
    home_pose = np.array([0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    # print(pin_rob.FK(home_pose, "R_BASE"))
    # print(pin_rob.FK(home_pose+2.0, "R_BASE"))
    # print(pin_rob.FK(home_pose+1.0, "R_BASE"))
    # print(pin_rob.FK_f2f(home_pose, baseName=BASE, frameName="R_BASE"))  # base-> R_base
    # print(pin_rob.FK_f2f(home_pose, baseName='universe', frameName="R_BASE"))
    # print(pin_rob.FK_f2f(home_pose, baseName='universe', frameName="R_BASE"))


    # Find DH params
    prevJoint = BASE
    prevJoint = "R_BASE"
    print(-1, BASE, "R_BASE")
    print(pin_rob.FK_f2f(home_pose, baseName=BASE, frameName="R_BASE"))

    for i, jointName in enumerate(R_joints):
        print(i, prevJoint, jointName)
        print(pin_rob.FK_f2f(home_pose, baseName=prevJoint, frameName=jointName))
        prevJoint = jointName

    print(i+1, prevJoint, TCP)
    print(pin_rob.FK_f2f(home_pose, baseName=prevJoint, frameName=TCP))

    print(i+2, "R_SFE", "R_EB")  # Shoulder -> Elbow
    print(pin_rob.FK_f2f(home_pose, baseName="R_SFE", frameName="R_EB"))

    print(i+3, "R_EB", "R_WFE")  # Elbow -> Wrist
    print(pin_rob.FK_f2f(home_pose, baseName="R_EB", frameName="R_WFE"))

    print(i+4, "R_WFE", TCP)  # Wrist -> TCP
    print(pin_rob.FK_f2f(home_pose, baseName="R_WFE", frameName=TCP))

    print(i+5, "R_BASE", TCP)
    print(pin_rob.FK_f2f(home_pose, baseName="R_BASE", frameName=TCP))

    print(i+6, BASE, TCP)
    print(pin_rob.FK_f2f(home_pose, baseName=BASE, frameName=TCP))



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