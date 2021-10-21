import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from apollo_interface.Apollo_It import JOINTS_LIMITS, R_joints, L_joints
import numpy as np
import math
try:
    import pinocchio as pin
except:
    pass

def reduce_model(FILENAME, jointsToUse):
    model = pin.buildModelFromUrdf(FILENAME)

    all_joints = set([n for n in model.names])
    jointsToKeep = set(["universe"] + jointsToUse)
    jointsToLock = list(all_joints - jointsToKeep)
    jointsToLockIDs = [model.getJointId(jn) for jn in jointsToLock]

    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, pin.neutral(model))

    # Info
    joint_index_dict = {jn: model_reduced.getJointId(jn) for jn in jointsToUse}
    sorted_jn_index = sorted(joint_index_dict.items(), key=lambda kv: kv[0])
    print("Free Joints and their corresponding indexes in Pinocchio:")
    print("    " + str(sorted_jn_index))
    return model_reduced

def modrad(q):
    """Limit q between where q is an numpy array"""
    res = np.array(q).copy()
    while np.any(res>np.pi):
        res = np.where(q>np.pi, q-2*np.pi, res)
    while np.any(res<-np.pi):
        res = np.where(q<-np.pi, q+2*np.pi, res)
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

def clip_c(v):
    return np.clip(v, -1.0, 1.0)


