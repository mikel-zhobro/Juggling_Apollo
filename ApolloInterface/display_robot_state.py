import math
import numpy as np
import O8O_apollo as apollo
from pinocchio.rpy import rpyToMatrix
import pinocchio
from scipy.spatial.transform import Rotation as R
pinocchio.switchToNumpyMatrix()
from Apollo_It import R_joints, L_joints, jointsToIndexDict
np.set_printoptions(precision=4, suppress=True)
NB_DOFS=29

observation = apollo.read()

iteration = observation.get_iteration()
observed_states = observation.get_observed_states()
desired_states = observation.get_desired_states()
cartesian_states = observation.get_cartesian()


r_arm_indexes = [jointsToIndexDict[j] for j in R_joints]
l_arm_indexes = [jointsToIndexDict[j] for j in L_joints]


def str_array(a):
    format_ = " , ".join(["{:.2f}" for value in a])
    return format_.format(*[value for value in a])


def print_joint(index, observed_states, desired_states):
    j, jd, jdd = observed_states.get(index).get()
    desired_j, desired_jd, desired_jdd = desired_states.get(index).get()
    str_ = str("{} | (observed) {:.3f} , {:.3f} , {:.3f} | "
               "(desired) , {:.3f} , {:.3f} , {:.3f} ").format(index,
                                                               j, jd, jdd,
                                                               desired_j, desired_jd,
                                                               desired_jdd)
    print(str_)

hands = ["RIGHT", "LEFT"]
def print_hand(index, hand):
    print "\nHand", hands[index]
    print "X , Y , Z"
    print np.array(hand.position)
    # print "\t\tvelocity:", str_array(hand.velocity)
    # print "\t\tacceleration:", str_array(hand.acceleration)
    # print "\tAngular velocity:", str_array(hand.orientation_angular_velocity)
    # print "\tAngular acceleration:", str_array(hand.orientation_angular_acceleration)
    print "Orientation (quaternion):"
    print R.from_quat(hand.orientation).as_dcm()
    # print quaternionToMatrix(*hand.orientation)
    # print "\t\tvelocity", str_array(hand.orientation_velocity)
    # print "\t\tacceleration", str_array(hand.orientation_acceleration)

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
     
        return np.array([roll_x, pitch_y, yaw_z]).reshape(-1, 1) # in radians

def quaternionToMatrix(x, y, z, w):
    return rpyToMatrix(euler_from_quaternion(x, y, z, w))


print ""

print("Right arm: Joint Angles\n------------")
for index in r_arm_indexes:
    print_joint(index, observed_states, desired_states)

print ""

print("Left arm: Joint Angles\n------------")
for index in l_arm_indexes:
    print_joint(index, observed_states, desired_states)

print ""

print("Cartesian\n---------")
for index, hand in enumerate(cartesian_states.hands):
    print_hand(index, hand)

print ""
