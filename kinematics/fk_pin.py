import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from utilities import modrad, euler_to_quaternion
np.set_printoptions(precision=3, suppress=True)
pin.switchToNumpyMatrix()  # https://github.com/stack-of-tasks/pinocchio/issues/802
jointsToUse = ["R_SFE", "R_SAA", "R_HR", "R_EB", "R_WR", "R_WFE", "R_WAA"]
N_JointUsed = len(jointsToUse)

# endeffector:
#   -
#     name: "R_HAND"
#     link: "R_PALM"
#   -
#     name: "L_HAND"
#     link: "L_PALM"

PLAYFUL_CARTESIAN = None
_IK = None
# inv kinematics params
IT_MAX = 10200
eps    = 1e-3
DT     = 10e-4
damp   = 1e-12
JOINT_ID = 7  # 29
TCP = "R_WAA"
TCP = "R_KUKA_TOOL"
TCP = "R_Arm_PalmBlueComponentJoint"
OFF = [0.65, 0.95, -0.57]
FILENAME = "/home/apollo/Software/workspace/src/catkin/robots/apollo_robot_model/target.urdf"
FILENAME = "/home/apollo/Software/workspace/src/catkin/playful/playful-kinematics/urdf/apollo.urdf"

def reduce_model(model, initialJointConfig):
    all_joints = [n for n in model.names]
    jointsToLock = list(set(all_joints) - set(["universe"] +jointsToUse))

    # Get the ID of all existing joints
    jointsToLockIDs = []
    for jn in jointsToLock:
        jointsToLockIDs.append(model.getJointId(jn))
    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)

    joint_index_dict = {jn: model_reduced.getJointId(jn) for jn in jointsToUse}
    sorted_jn_index = sorted(joint_index_dict.items(), key=lambda kv: kv[0])
    print("Free Joints and their corresponding indexes in Pinocchio:")
    print("    " + str(sorted_jn_index))
    return model_reduced, joint_index_dict

class PinRobot():
    def __init__(self, filename=FILENAME):
        """ Initialize pinocchio dependent models/variables: model, data, joint_states
            filename ([str]): path to the urdf file
        """
        rpy = np.array([0,0,-3.14], dtype='float').reshape(3,1)
        t = np.array([0,0,-1.4])
        rpy = np.array([0.0, 0.0, 0.0], dtype='float').reshape(3,1)
        t = np.array([0.0, 0.0, 0.0])
        R = pin.rpy.rpyToMatrix(rpy)
        self.T_BASE_ORIGIN = pin.SE3(R, t)
        
        model = pin.buildModelFromUrdf(filename)
        self.model, self.joint_index_dict = reduce_model(model, pin.neutral(model))  # model, only the constant information(that never changes)
        self.data = self.model.createData()  # information that changes according to joint configuration etc
                                             # Only used as internal state(still all functions should be called with certain joint_state as input)
        
        # self.print_data(self.data)
        # self.print_joint_configuration(pin.neutral(self.model))
        print('-'*45)

    def FK(self, q, frameName=TCP):
        """
        returns the SE3_world_frame(R,p) of frameName in world coordinates
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        return pin.updateFramePlacement(self.model, self.data, frameId) # updates data and returns T_world_frame

    def J(self, q, frameName=TCP):
        """
        returns SE3(R,p) and J of TCP in world coordinates
        """
        frameId = self.model.getFrameId(frameName)
        pin.forwardKinematics(self.model, self.data, q)
        SE3_world_tcp = pin.updateFramePlacement(self.model, self.data, frameId)
        R_R_world_frame = SE3_world_tcp.action
        J_local = pin.computeFrameJacobian(self.model, self.data, q, frameId)
        return R_R_world_frame.dot(J_local), SE3_world_tcp

    def ik_apollo(self, Q_start, goal_p, goal_R=None, frameName=TCP, plot=False):
        """ Inverse Kinematics

        Args:
            goal_p ([float, float, float]): Endeffector xyz position
            goal_R ([np.array((3,3))], optional): Endeffector's orientation. If not specified it will be vertical.
            Q_i ([float]*N_JointUsed], optional): Whether to plot the error.
            plot (bool, optional): Whether to plot the error.
            frameName (str, optional): Name of TCP

        Returns:
            [list]: joint configuration to achieve desired endeffector position/orientation
        """
        # Get Frame Id
        frameId = self.model.getFrameId(frameName)
        
        # Desired TCP cartesian position
        goal_p = np.array(goal_p).reshape(3,1)
        goal_R = goal_R if goal_R is not None else np.eye(3)[:,[2,0,1]]
        SE3_world_goal = self.T_BASE_ORIGIN.inverse() * pin.SE3(goal_R, goal_p)
        print("GOAL")
        print(SE3_world_goal)
        
        def get_se3_error(SE3_world_tcp_i):
            # dMi = SE3_world_goal.actInv(SE3_world_tcp_i)
            # err = SE3_world_goal.action * pin.log(dMi).vector
            # return err
            return SE3_world_tcp_i.action * pin.log(SE3_world_tcp_i.actInv( SE3_world_goal)).vector

        i=0
        errs = []
        Q_i = Q_start.copy()
        while True:


            # Calc T_world_tcp and J_world_tcp for the new joint_states
            J_world_tcp, SE3_world_tcp_i  = self.J(Q_i, frameName)

            # Calc cartesian errors
            err = get_se3_error(SE3_world_tcp_i)
            
            # 1 Calc qoint velocities
            # qv = - J_world_tcp.T.dot(solve(J_world_tcp.dot(J_world_tcp.T) + damp * np.eye(6), err))
            
            # 2
            J_invj = np.linalg.pinv(J_world_tcp)
            qv = np.matmul(J_invj, err)
            
            # Update joint_states
            Q_i = pin.integrate(self.model, Q_i, qv*DT)


            i += 1
            print('\nfinal error: %s' % err.T + '\t norm(error): %s' % norm(err))
            if norm(err) < eps or i >= IT_MAX:
                success = norm(err) < eps
                break
            if not i % 10:
                print('%d: error = %s' % (i, err.T))
            if plot:
                errs.append(norm(err))

        if success:
            print("Convergence achieved!")
        else:
            print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

        # self.print_joint_configuration(Q_i)
        # self.print_data(SE3_world_tcp_i)
        print('\nfinal error: %s' % err.T + '\t norm(error): %s' % norm(err))

        if plot:
            plt.plot(errs)
            plt.show()
        return Q_i.copy()

    # ======================== Some printing functions ========================
    # =========================================================================
    def print_data(self, data, frameName=None, xyz=True, rpy=True):
        """
            data ([type]): print rotation and translation of the frameName in data
            xyz/rpy (bool, optional): True if translation/rotation should be print
        """
        print("\nCartesan coordinates according to FK:")
        if frameName is None:
            frameName = TCP
        frameId = self.model.getFrameId(frameName)
        data2 = self.T_BASE_ORIGIN * data
        print("    {}:".format(frameName))
        if xyz:
            print(("\t{: .3f} {: .3f} {: .3f} [xyz]".format(*data2.translation.T.flat)))
        if rpy:
            print(("\t{: .3f} {: .3f} {: .3f} {: .3f} [rpy] \n".format(*euler_to_quaternion(pin.rpy.matrixToRpy(data2.rotation).T.flat) )))
            print((data2.rotation))

    def print_joint_configuration(self, joint_states):
        print("\nJoint Configuration:")
        for k in range(self.model.njoints-1):
            print(("  - {:<24} : {: .3f}".format(self.model.names[k+1], joint_states[k, 0])))


if __name__ == "__main__":
    pin_rob = PinRobot()
    home_new = np.array([np.pi/8, -0.4, 0.0, 3*np.pi/8, 0.0, 0.0, 0.0])
    
    SE3_w_tcp = pin_rob.FK(home_new)
    q_goal = pin_rob.ik_apollo(home_new, SE3_w_tcp.translation+0.1, SE3_w_tcp.rotation)
    
    
    print(q_goal)
    print()
    







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