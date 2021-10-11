import numpy as np
import math
import pinocchio as pin


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
    res = q.copy()
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


class ContinuousSet():
    def __init__(self, start, end, start_include=True, end_include=True):
        self.a = start
        self.b = end
        self.a_incl = start_include
        self.b_incl = end_include

    def __add__(self, other):
        if self.a < other.a:
            start = self.a
            start_include = self.a_incl
        else:
            start = other.a
            if abs(self.a - other.a) < 1e-7:
                end_include = self.a_incl or other.a_incl
            else:
                start_include = other.a_incl

        if self.b < other.b:
            end = other.b
            end_include = other.b_incl
        else:
            end = self.b
            if abs(self.b - other.b) < 1e-7:
                end_include = self.a_incl or other.a_incl
            else:
                end_include = self.b_incl

        return ContinuousSet(start, end, start_include, end_include)

    def __sub__(self, other):
        if self.a < other.a:
            start = other.a
            start_include = other.a_incl
        else:
            start = self.a
            if abs(self.a - other.a) < 1e-7:
                start_include = self.a_incl and other.a_incl
            else:
                start_include = self.a_incl

        if self.b < other.b:
            end = self.b
            end_include = self.b_incl
        else:
            end = other.b
            if abs(self.b - other.b) < 1e-7:
                end_include = self.a_incl and other.a_incl
            else:
                end_include = other.b_incl

        return ContinuousSet(start, end, start_include, end_include)

    def __str__(self):
        return '{}{}, {}{}'.format('[' if self.a_incl else '(', self.a, self.b, ']' if self.b_incl else ')')
def skew(v):
    return np.array([[ 0.0,   -v[2],  v[1]],
                     [ v[2],   0.0,  -v[0]],
                     [-v[1],   v[0],  0.0]], dtype='float')


def vec(elems):
    return np.array(elems, dtype='float').reshape(-1, 1)


def clip_c(v):
    return np.clip(v, -1.0, 1.0)
