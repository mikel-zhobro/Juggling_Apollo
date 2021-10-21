import numpy as np
from utilities import ContinuousSet
from utilities import JOINTS_LIMITS
from math import sin, cos


# (-1, 'BASE', 'R_BASE')
#   R =
#    -0.5          0.866025    0.0
#     0.866025     0.5         0.0
#     0.0          0.0        -1.0
#   R =
#    -sin(pi/6)     cos(pi/6)    0.0
#     cos(pi/6)     sin(pi/6)    0.0
#     0.0           0.0         -1.0
#   p = 0 0 0

spi6 = 0.5
cpi6 = cos(np.pi/6)

T_rbase_prim_rbase = np.array([  # Sets the Arm to the right base frame
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.0],
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0]], dtype = 'float')

class DH_revolut():
    n_joints = 0
    class Joint():
        def __init__(self, a, alpha, d, theta, name="", offset=0.0):
            self.a = a
            self.alpha = alpha
            self.d = d
            self.theta = theta
            self.name = name
            self.offset = offset
            self.index = DH_revolut.n_joints
            self.limit_range = ContinuousSet(JOINTS_LIMITS[self.name][0]-offset, JOINTS_LIMITS[self.name][1]-offset, False, False)
            DH_revolut.n_joints += 1

        def __repr__(self):
            return '{}. {}: a({}), b({}), d({}), theta({})'.format(self.index, self.name, self.a, self.alpha, self.d, self.theta)

    def __init__(self):
        self.joints = []

    def joint(self, i):
        return self.joints[i]

    def add_joint(self, a, alpha, d, theta, name):
        self.joints.append(self.Joint(a, alpha, d, theta, name))

    def getT(self, j, theta):
        c_th = cos(theta + j.theta + j.offset); s_th = sin(theta + j.theta + j.offset)
        if j.index==6:
            return np.array(
                [[s_th,   c_th,    0.0,  0.0],
                [-c_th,   s_th,    0.0,  0.0],
                [0.0,     0.0,     1.0,  j.d],
                [0.0,     0.0,     0.0,  1.0]], dtype='float')
        else:
            sig = float(np.sign(j.alpha))
            return np.array(
                [[c_th,   0.0,      s_th*sig,  0.0],
                [s_th,   0.0,      -c_th*sig,  0.0],
                [0.0,    sig,       0.0,       j.d],
                [0.0,    0.0,       0.0,       1.0]], dtype='float')

    def FK(self, Q):
        Q = Q.copy().reshape(-1, 1)
        T0_7 = np.eye(4)
        for j, theta_j in zip (self.joints, Q):
            T0_7 = T0_7.dot(self.getT(j, theta_j))
        return T_rbase_prim_rbase.dot(T0_7)

    def get_i_R_j(self, i, j, Qi_j):
        # Qi_j: angles of joints from i to j
        assert i != j, 'i:{} and j:{} cannot be equal'.format(i,j)
        transpose_at_end = False
        if i>j:
            transpose_at_end = True
            a  = j
            j = i
            i = a

        i_R_j = np.eye(3)
        for joint, th_j in zip(self.joints[i:j], Qi_j):
            i_R_j = i_R_j.dot(self.getT(joint, th_j)[:3, :3])

        if transpose_at_end:
            i_R_j = i_R_j.T
        return i_R_j

    def get_i_T_j(self, i, j, Qi_j, s_11=0, rot=False):
        assert i < j, 'i:{} and j:{} cannot be equal'.format(i,j)

        a = 3 if rot else 4
        i_T_j = np.eye(a)
        for joint, th_j in zip(self.joints[i:j], Qi_j):
            i_T_j = i_T_j.dot(self.getT(joint, th_j)[:a, :a])
        return i_T_j

    def get_goal_in_base_frame(self, p, R):
        R_ret = T_rbase_prim_rbase[:3,:3].T.dot(R)
        p_ret = T_rbase_prim_rbase[:3,:3].T.dot(p) - T_rbase_prim_rbase[:3,3:4]
        return p_ret, R_ret

    def plot(self):
        T0wshoulder = self.get_i_T_j(0, 2)
        T0wselbo    = self.get_i_T_j(0, 4)
        T0wswrist   = self.get_i_T_j(0, 6)
        pass
