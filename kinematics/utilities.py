import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from sortedcontainers import SortedList, SortedSet
import numpy as np
import math
import pinocchio as pin
from apollo_interface.Apollo_It import JOINTS_LIMITS, R_joints, L_joints


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

class ContinuousRange():
    def __init__(self, start=None, end=None, start_include=True, end_include=True):
        correct = start < end  # sort the range if given out of order
        if correct:
            self.a = start
            self.b = end if correct else start
            self.a_incl = start_include
            self.b_incl = end_include
        else:
            self.a = end
            self.b = start
            self.a_incl = end_include
            self.b_incl = start_include

    def __repr__(self):
        return '{}{}, {}{}'.format('[' if self.a_incl else '(', self.a, self.b, ']' if self.b_incl else ')')

    def __lt__(self, other):
        lower = False
        if self.a == other.a:
            lower = self.b < other.b
        else:
            lower = self.a < other.a
        return lower

class ContinuousSet():
    def __init__(self, c_ranges=SortedList()):
        c_ranges = SortedList(c_ranges)
        self.c_ranges = SortedList()  # set of ContinuousRanges
        for c_r in c_ranges:
            self.add(c_r)

    def add(self, c_range):
        self.c_ranges = self._add(c_range)
        pass

    def _add(self, c_range):
        c_ranges_new = self.c_ranges.copy()
        startt = c_range
        endd = c_range
        # Find the start and end overlaping intervals
        to_remove = set()
        for cr in self:
            if c_range.a < cr.a and c_range.b > cr.b:
                to_remove.add(cr)
            if cr.a <= c_range.a <= cr.b:
                startt = cr
                to_remove.add(cr)
            if cr.a <= c_range.b <= cr.b:
                endd = cr
                to_remove.add(cr)
                break

        # Remove continuous ranges in between
        for cr in to_remove:
            c_ranges_new.discard(cr)
        a_incl = startt.a_incl or c_range.a_incl if abs(startt.a-c_range.a)<1e-8 else startt.a_incl
        b_incl = endd.b_incl or c_range.b_incl if abs(endd.b-c_range.b)<1e-8 else endd.b_incl

        # Add the new range
        c_ranges_new.add(ContinuousRange(startt.a, endd.b, a_incl, b_incl))
        return c_ranges_new

    def __add__(self, other):
        ret_set = ContinuousSet()
        ret_set.c_ranges = self.c_ranges.copy()
        for cr in other:
            ret_set.c_ranges = ret_set._add(cr)
        return ret_set

    def __iter__(self):
        for c_r in self.c_ranges:
            yield c_r

    def _sub(self, new_range):
        c_ranges_new = SortedList()
        # Find the start and end overlaping intervals
        for cr in self:
            if new_range.a < cr.a and new_range.b > cr.b:
                # 1. new range includes range
                c_ranges_new.add(cr)

            if cr.a <= new_range.a < cr.b:
                # 2. start of new range is in range (and end of new range can be either in(2.1) or out(2.2))
                a_incl = cr.a_incl and new_range.a_incl if abs(cr.a - new_range.a) < 1e-6 else new_range.a_incl
                if new_range.b > cr.b:
                    # 2.1. only start of new range included in range
                    c_ranges_new.add(ContinuousRange(new_range.a, cr.b, a_incl, cr.b_incl))

                else:
                    # 2.2. new range included in range
                    b_incl = cr.b_incl and new_range.b_incl if abs(cr.b - new_range.b) < 1e-7 else new_range.b_incl
                    c_ranges_new.add(ContinuousRange(new_range.a, new_range.b, a_incl, b_incl))

            elif cr.a < new_range.b <= cr.b and new_range.a < cr.a:
                # 3. only end of new range included in range
                b_incl = cr.b_incl and new_range.b_incl if abs(cr.b - new_range.b) < 1e-7 else new_range.b_incl
                c_ranges_new.add(ContinuousRange(cr.a, new_range.b, a_incl, b_incl))

        return c_ranges_new

    def __sub__(self, other):
        ret_set = ContinuousSet()
        for o_r in other.c_ranges:
            ret_set += self._sub(o_r)
        return ret_set

    def __repr__(self):
        return str([c_r for c_r in self.c_ranges])

def skew(v):
    return np.array([[ 0.0,   -v[2],  v[1]],
                     [ v[2],   0.0,  -v[0]],
                     [-v[1],   v[0],  0.0]], dtype='float')


def vec(elems):
    return np.array(elems, dtype='float').reshape(-1, 1)


def clip_c(v):
    return np.clip(v, -1.0, 1.0)



# a = ContinuousSet()
# a.add(ContinuousRange(2,3))
# print(a)

ss = ContinuousSet()

ss.add(ContinuousRange(2,3,))
ss.add(ContinuousRange(2.1,3.1))
ss.add(ContinuousRange(4,15, False, False))
ss.add(ContinuousRange(12,5))
ss.add(ContinuousRange(4,5))
ss.add(ContinuousRange(5,6))

ss2 = ContinuousSet()

ss2.add(ContinuousRange(2, 3.4, False))
ss2.add(ContinuousRange(122, 13.9, False))
ss2.add(ContinuousRange(2.1,3.5))
ss2.add(ContinuousRange(4.2,13.8))
ss2.add(ContinuousRange(5, 11, False, False))


print('ss', ss)
print('ss2', ss2)
print('+', ss + ss2)
print('-', ss - ss2)
# print(ss)
