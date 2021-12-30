import numpy as np
from Sets import ContinuousSet
from utilities import JOINTS_LIMITS, pR2T, invT
from math import sin, cos
np.set_printoptions(precision=4, suppress=True)


spi6 = 0.5
cpi6 = cos(np.pi/6)

T_base_rbase = np.array([  # Sets the Arm to the correct(main) base frame
    [-spi6,  cpi6,  0.0, 0.0],
    [ cpi6,  spi6,  0.0, 0.0],
    [ 0.0,   0.0,  -1.0, 0.0],
    [ 0.0,   0.0,   0.0, 1.0]], dtype = 'float')

T_rbase_dhbase = np.array([  # Sets the Arm to the right base frame
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.0],
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0]], dtype = 'float')


class DH_revolut():
    class Joint():
        def __init__(self, a, alpha, d, theta, index, limit=(-np.pi, np.pi), vlimit=(-10., 10.), name="", offset=0.0):
            self.a = a
            self.alpha = alpha
            self.d = d
            self.theta = theta
            self.name = name
            self.offset = offset
            self.index = index
            self.limit_range = ContinuousSet(limit[0], limit[1], False, False)
            self.vlimit_range = ContinuousSet(vlimit[0], vlimit[1], False, False)

        def __repr__(self):
            return '{}. {}: a({}), b({}), d({}), theta({})'.format(self.index, self.name, self.a, self.alpha, self.d, self.theta)

    def __init__(self):
        self.n_joints = 0
        self.joints = list()

    def joint(self, i):
        return self.joints[i]

    def add_joint(self, a, alpha, d, theta, limit=(-np.pi, np.pi), vlimit=(-10., 10.), name="", offset=0.0):
        self.joints.append(self.Joint(a, alpha, d, theta, self.n_joints, limit, vlimit, name, offset))
        self.n_joints += 1

    def getT(self, j, theta):
        """Returns the transformation matrix related to joint j and q_j=theta, according to the DH table.

        Args:
            j (Joint): joint
            theta (float): joint angle

        Returns:
            [np.array(4,4)]: j_T_j+1
        """
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

    def FK(self, Q, rbase_frame=False):
        Q = Q.copy().reshape(-1, 1)
        T0_7 = np.eye(4)
        for j, theta_j in zip(self.joints, Q):
            T0_7 = T0_7.dot(self.getT(j, theta_j))
        # dh_base -> r_base
        T0_7 = T_rbase_dhbase.dot(T0_7)
        if not rbase_frame:  # dont enter if we want fk in rbase frame
            # r_base -> base
            T0_7 = T_base_rbase.dot(T0_7)
        return T0_7

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

    def get_i_T_j(self, i, j, Qi_j):
        assert i < j, 'i:{} and j:{} cannot be equal'.format(i,j)

        i_T_j = np.eye(4)
        for joint, th_j in zip(self.joints[i:j], Qi_j):
            i_T_j = i_T_j.dot(self.getT(joint, th_j))
        return i_T_j

    def get_goal_in_dh_base_frame(self, p_base, R_base):
        # base-> rbase
        T_rbase_base = invT(T_base_rbase)
        R_ret = T_rbase_base[:3,:3].dot(R_base)
        p_ret = T_rbase_base[:3,:3].dot(p_base) + T_rbase_base[:3,3:4]

        # rbase -> dh_base
        T_dhbase_rbase = invT(T_rbase_dhbase)
        R_ret = T_dhbase_rbase[:3,:3].dot(R_ret)
        p_ret = T_dhbase_rbase[:3,:3].dot(p_ret) + T_dhbase_rbase[:3,3:4]

        return p_ret, R_ret

    def i_J_j(self, i, j, Qi_j):
        # J = [zi(x)delta(pi), .., zk(x)delta(pk), .., zj(x)delta(pj)
        #      zi            , .., zk            , .., zj            ] where delta(pk) = pj -pk
        assert i < j, 'i:{} and j:{} cannot be equal'.format(i,j)
        N = j-i
        i_T_k = np.eye(4)
        z_s = [None]*(N+1); z_s[0] = i_T_k[:3, 2:3]
        p_s = np.zeros((3, N+1)); p_s[:,0] = i_T_k[:3, -1]
        for n, joint_k, th_k in zip(range(N), self.joints[i:j], Qi_j):
            i_T_k = i_T_k.dot(self.getT(joint_k, th_k))
            z_s[n+1]    = i_T_k[:3, 2:3]
            p_s[:, n+1] = i_T_k[:3, 3]

        J_s = []
        for n in range(N):
            J_s.append(np.cross(z_s[n], p_s[:, N:N+1]-p_s[:, n:n+1], axis=0))

        JP = np.hstack(J_s)
        JO = np.hstack(z_s[:-1])

        J = np.vstack([JP, JO])
        return J

    def J(self, q, rbase_frame=False):
        # In order to tranfsorm the jacobian in the right base frame we have to premultiply with
        #  [R_base_base_dh    0
        #   0                 R_base_base_dh]
        # jacobian expresses the cartesian velocities caused byz joint velocities. We want the cartesian velocities in the right base.

        # dh_base -> rbase
        T_base_dhbase = T_rbase_dhbase.copy()
        # rbase -> base
        if not rbase_frame:  # dont enter if we want to work with reference to rbase
            T_base_dhbase = T_base_rbase.dot(T_base_dhbase)

        TMP = np.eye(6)
        TMP[:3,:3] = T_base_dhbase[:3,:3]
        TMP[3:,3:] = T_base_dhbase[:3,:3]
        return TMP.dot(self.i_J_j(0, 7, q))

    def H(self, q, rbase_frame=False):
        n = self.n_joints

        H = np.zeros((6, n, n))
        J0 = self.J(q, rbase_frame)

        for j in range(n):
            for i in range(j, n):
                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]
        return H

    def plot(self):
        T0wshoulder = self.get_i_T_j(0, 2)
        T0wselbo    = self.get_i_T_j(0, 4)
        T0wswrist   = self.get_i_T_j(0, 6)
        pass


if __name__ == "__main__":
    from kinematics.utilities import R_joints
    pi2 = np.pi/2
    th3_offset = np.pi/6
    d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186
    a_s            = [0.0] * 7
    alpha_s        = [pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
    d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
    theta_s = [0.0, -pi2, pi2, 0.0, -pi2, 0.0, 0.0]
    offsets = [0.0, 0.0, th3_offset, 0.0, 0.0, 0.0, 0.0]

    # Create Robot
    my_fk_dh = DH_revolut()
    for a, alpha, d, theta, name, offset in zip(a_s, alpha_s, d_s, theta_s, R_joints, offsets):
        my_fk_dh.add_joint(a, alpha, d, theta, JOINTS_LIMITS[name], name, offset)



    # Joint configuration
    home_pose = np.array([0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    base_T_tcp = my_fk_dh.FK(home_pose, False)
    print(base_T_tcp)  # base_T_tcp
    print(my_fk_dh.get_i_T_j(0,7, home_pose))  # basedh_T_tcp
