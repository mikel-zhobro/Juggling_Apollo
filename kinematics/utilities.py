import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from apollo_interface.Apollo_It import JOINTS_LIMITS, R_joints, L_joints, JOINTS_V_LIMITS
import numpy as np
from scipy.linalg import logm
import math
try:
    import pinocchio as pin
except:
    pass

VERBOSE = False

def SO3_2_so3(R):
    theta = math.acos(clip_c((np.trace(R)-1.0)/2.0))
    if np.abs(theta) < 1e-6: return np.array([0.0, 0.0, 0.0], dtype='float').reshape(3,1), theta

    w = 0.5 * np.array([R[2,1] - R[1,2],
                        R[0,2] - R[2,0],
                        R[1,0] - R[0,1]], dtype='float').reshape(3,1)
    return w, theta

def orientation_error2(R_i, R_goal):
    R_e = R_i.T.dot(R_goal)  # error rotation
    n_e, theta_e = SO3_2_so3(R_e)
    return n_e*theta_e

def orientation_error(R_i, R_goal):
    R_e = R_i.T.dot(R_goal)  # error rotation
    w_e = logm(R_e)
    wx = w_e[-1, 1]
    wy = w_e[0, -1]
    wz = w_e[1, 0]
    n_e = np.array([wx, wy, wz]).reshape(3,1)
    return n_e

def errorForJacobianInverse(T_i, T_goal):
    p_j = T_i[:3, 3:]
    R_j = T_i[:3, :3]

    p_goal = T_goal[:3, 3:]
    R_goal = T_goal[:3, :3]

    # delta_x, delta_y, delta_z between start and desired position of the end effector
    delta_p = p_goal - p_j
    # delta_nx, delta_ny, delta_nz between start and desired orientation of the  end effector
    # delta_n1 = orientation_error(R_j, R_goal) # weird, returns complex delta for certain scenarious
    # print(delta_n1)
    delta_n = orientation_error2(R_j, R_goal)
    # print(delta_n)

    return np.vstack([delta_p, delta_n])

def reduce_model(FILENAME, jointsToUse):
    model = pin.buildModelFromUrdf(FILENAME)

    all_joints = set([n for n in model.names])
    jointsToKeep = set(["universe"] + jointsToUse)
    jointsToLock = list(all_joints - jointsToKeep)
    jointsToLockIDs = [model.getJointId(jn) for jn in jointsToLock]

    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, pin.neutral(model))

    # Info
    joint_index_dict = {jn: model_reduced.getJointId(jn) for jn in jointsToUse}
    if VERBOSE:
        sorted_jn_index = sorted(joint_index_dict.items(), key=lambda kv: kv[0])
        print("Free Joints and their corresponding indexes in Pinocchio:")
        print("    " + str(sorted_jn_index))
    return model_reduced

def modrad(q):
    """Limit q between where q is an numpy array"""
    res = np.array(q).copy()
    while np.any(res>np.pi):
        res = np.where(res>np.pi, res-2*np.pi, res)
    while np.any(res<-np.pi):
        res = np.where(res<-np.pi, res+2*np.pi, res)
    return res

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

def skew(v):
    return np.array([[ 0.0,   -v[2],  v[1]],
                     [ v[2],   0.0,  -v[0]],
                     [-v[1],   v[0],  0.0]], dtype='float')

def vec(elems):
    return np.array(elems, dtype='float').reshape(-1, 1)

def clip_c(v, min=-1.0, max=1.0):
    return np.clip(v, min, max)

def pR2T(p,R):
    """ From positions and rotations to homogenous transformations
        It handles both trajectories and single ones.

    Args:
        p ([type]): position or position trajectory
        R ([type]): rotation or rotation trajectory

    Returns:
        [type]: [description]
    """
    p_shape = p.shape
    R_shape = R.shape
    T_shape = (4,4) if len(p_shape)==2 else (-1, 4, 4)
    p = p.reshape(-1,3,1)
    R = R.reshape(-1,3,3)

    N = R.shape[0]
    T = np.zeros((N,4,4))
    for i, (p_, R_) in enumerate(zip(p, R)):
        T[i, :3,:3] = R_
        T[i, :3, 3:4] = p_
        T[i, 3, 3] = 1.0

    p = p.reshape(p_shape)
    R = R.reshape(R_shape)
    return T.reshape(T_shape)

def invT(T):
    T_ = T.copy()
    T_[:3,:3] = T[:3,:3].T
    T_[:3,3:4] = -T_[:3,:3].dot(T[:3,3:4])
    return T_



# T_i = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
#                 [0.0,  0.0, 1.0,  0.81],
#                 [-1.0, 0.0, 0.0, -0.49],
#                 [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

# T_goal = np.array([[1.0, 0.0, 0.0, 0.32],  # uppword orientation(cup is up)
#                    [0.0, -1.0, 0.0, 0.81],
#                    [0.0, 0.0, 1.0, 0.49],
#                    [0.0, 0.0, 0.0, 0.0 ]], dtype='float')
# errorForJacobianInverse(T_i, T_goal)