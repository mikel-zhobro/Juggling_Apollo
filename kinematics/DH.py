import numpy as np
from utilities import ContinuousSet, skew, vec, clip_c
from math import sin, cos, atan, acos, asin, sqrt, atan2
# from numpy import sin, cos, sqrt, arctan2, arccos
# from numpy import arctan2 as atan2
# from numpy import arccos as acos
# from kinematics.fk import FK
np.set_printoptions(precision=4, suppress=True)


pi2 = np.pi/2
d_bs = 0.1; d_se = 0.4; d_ew = 0.3; d_wt = 0.1
a_s            = [0.0] * 7
alpha_s        = [-pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
theta_offset_s = [0.0] * 7


class DH_revolut():
    n_joints = 0
    class Joint():
        def __init__(self, a, alpha, d, theta):
            self.a = a
            self.alpha = alpha
            self.d = d
            self.theta = theta
            self.index = DH_revolut.n_joints
            DH_revolut.n_joints += 1

    def __init__(self):
        self.joints = []

    def add_joint(self, a, alpha, d, theta):
        self.joints.append(self.Joint(a, alpha, d, theta))

    def getT(self, j, theta):
        c_th = cos(theta - j.theta); s_th = sin(theta - j.theta)
        if j.index==6:
            return np.array(
                [[c_th,  -s_th,    0.0,  0.0],
                [s_th,    c_th,    0.0,  0.0],
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

    def get_i_T_j(self, i, j, Qi_j, s_11=0, rot=False):
        assert i < j, 'i:{} and j:{} cannot be equal'.format(i,j)

        a = 3 if rot else 4
        i_T_j = np.eye(a)
        for joint, th_j in zip(self.joints[i:j], Qi_j):
            i_T_j = i_T_j.dot(self.getT(joint, th_j)[:a, :a])
        return i_T_j

    def get_i_p_j(self, Q):
        pass


def IK_anallytical(p07_d, R07_d, DH_model, verbose=False):
    """
        Implementation from paper: "Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators...
        With Joint Limits and Its Application to Redundancy Resolution",
        IEEE TRANSACTIONS ON ROBOTICS, VOL. 24, NO. 5, OCTOBER 2008

    Args:
        o_p_goal ([R^3]): the goal position in base frame [3x1]
        o_R_goal ([SO3]): the goal orientation in origin frame [3x3]
    """
    l0bs = vec([0,   0,     d_bs])
    l3se = vec([0,  -d_se,  0])
    l4ew = vec([0,   0,     d_ew])
    l7wt = vec([0,   0,     d_wt])

    # shoulder to wrist axis
    x0sw = p07_d - l0bs - R07_d.dot(l7wt)

    # Elbow joint
    c_th4 = clip_c((np.linalg.norm(x0sw)**2 - d_se**2 - d_ew**2) / (2*d_se*d_ew))
    th4 = acos(c_th4)
    if verbose:
        print('Theta4:', th4)
    assert (d_se**2 + d_ew**2 + (2*d_se*d_ew)*c_th4 - np.linalg.norm(x0sw)**2) <= 1e-6, 'Should have used -sqrt(aa) maybe'

    # Shoulder joints (reference plane)
    R23_ref = DH_model.get_i_R_j(2,3, [0.0])
    R34 = DH_model.get_i_R_j(3,4, [th4])

    d = R23_ref.dot(l3se + R34.dot(l4ew))

    d13_2 = d[0,0]**2 + d[2,0]**2
    aa = d13_2 - x0sw[2,0]**2

    s2 = clip_c((sqrt(aa) * d[2,0] - x0sw[2,0]*d[0,0]) / d13_2)
    c2 = clip_c((s2*d[0,0] + x0sw[2,0])/ d[2,0])
    assert (s2*d[0,0] - c2*d[2,0] + x0sw[2,0]) <= 1e-6, 'Should have used -sqrt(aa) maybe'
    if verbose:
        print('Theta2', atan2(s2, c2))

    if( abs(p07_d[0,0])<1e-6 and abs(p07_d[1,0])<1e-6 ):
        s1 = 0.0
        c1 = 1.0
    else:
        x12_2 = x0sw[0,0]**2 + x0sw[1,0]**2
        s1 = clip_c((x0sw[1,0]*(c2*d[0,0] + s2*d[2,0]) - d[1,0]*x0sw[0,0]) / x12_2)
        c1 = clip_c((x0sw[0,0]*(c2*d[0,0] + s2*d[2,0]) - d[1,0]*x0sw[1,0]) / x12_2)
        assert (-s1*x0sw[0,0] + c1*x0sw[1,0] - d[1,0]) <= 1e-6, "s1,c1 wrongly calculated"
    if verbose:
        print('Theta1', atan2(s1, c1), s1, c1)


    R03_ref = np.array([[ c1*c2,  -c1*s2,  -s1],
                        [ s1*c2,  -s1*s2,   c1],
                        [-s2,     -c2,      0.0]], dtype='float')
    # R03_ref = DH_model.get_i_R_j(0,3, [atan2(c1, s1), atan2(c2, s2), 0.0])



    u0sw = x0sw/np.linalg.norm(x0sw)
    u0sw = u0sw/np.linalg.norm(u0sw)
    u0sw_skew = skew(u0sw)

    # Shoulder
    As = u0sw_skew.dot(R03_ref)
    Bs = -np.matmul(u0sw_skew.dot(u0sw_skew), R03_ref)
    Cs = np.matmul(u0sw_skew.dot(u0sw_skew.T), R03_ref)
    Cs = R03_ref - Bs
    # Wrist
    Aw = np.matmul(R34.T, As.T.dot(R07_d))
    Bw = np.matmul(R34.T, Bs.T.dot(R07_d))
    Cw = np.matmul(R34.T, Cs.T.dot(R07_d))

    # 1. shoulder solutions
    t11 = lambda psi: ( As[1,1]*sin(psi)  + Bs[1,1]*cos(psi) + Cs[1,1] )
    t12 = lambda psi: ( As[0,1]*sin(psi)  + Bs[0,1]*cos(psi) + Cs[0,1] )
    c22 = lambda psi:   clip_c(-As[2,1]*sin(psi)  - Bs[2,1]*cos(psi) - Cs[2,1])
    t31 = lambda psi: (  As[2,2]*sin(psi)  + Bs[2,2]*cos(psi) + Cs[2,2] )
    t32 = lambda psi: ( As[2,0]*sin(psi)  + Bs[2,0]*cos(psi) + Cs[2,0] )

    # 2. wrist solutions
    t51 = lambda psi: ( Aw[1,2]*sin(psi)  + Bw[1,2]*cos(psi) + Cw[1,2] )
    t52 = lambda psi: ( Aw[0,2]*sin(psi)  + Bw[0,2]*cos(psi) + Cw[0,2] )
    c6 =  lambda psi:   clip_c(Aw[2,2]*sin(psi)  + Bw[2,2]*cos(psi) + Cw[2,2])
    t71 = lambda psi: (  Aw[2,1]*sin(psi)  + Bw[2,1]*cos(psi) + Cw[2,1] )
    t72 = lambda psi: ( Aw[2,0]*sin(psi)  + Bw[2,0]*cos(psi) + Cw[2,0] )

    return lambda psi: np.array([atan2(-t11(psi), -t12(psi) ), acos(c22(psi)),
                                 atan2(t31(psi), -t32(psi) ), th4,
                                 atan2(t51(psi), t52(psi) ), acos(c6(psi)),
                                 atan2(t71(psi), -t72(psi) )]).reshape(-1,1)


def tangent_type(an, bn, cn, ad, bd, cd):
    at = bd*cn - bn*cd; at_2 = at**2
    bt = an*cd - ad*cn; bt_2 = bt**2
    ct = an*bd - ad*bn; ct_2 = ct**2

    if at_2 + bt_2 - ct_2 > 1e-6:  # cyclic profile
        ss = at_2 + bt_2 - ct_2
        psi_min = 2 * atan2( at - sqrt(ss), bt-ct )
        psi_max = 2 * atan2( at + sqrt(ss), bt-ct )
    elif at_2 + bt_2 - ct_2 < 1e-6:  # monotonic profile
        pass
    else:  # discontinuous profile (2 possibilities)
        psi_stationary = 2 * atan2(at, bt-ct)
        psi_s_neg = atan2(-1/ct*(at*bn - bt*an), -1/ct*(at*bd - bt*ad))
        psi_s_neg = atan2(1/ct*(at*bn - bt*an), 1/ct*(at*bd - bt*ad))



def cosine_type(a, b, c):
    a_2 = a**2
    b_2 = b**2
    c_2 = c**2

    psi_stat_neg = 2 * atan2(-b - sqrt(a_2+b_2), a)
    psi_stat_neg = 2 * atan2(-b + sqrt(a_2+b_2), a)

    if (a_2 + b_2 - (c-1)**2) < 1e-6:  # cyclic jumping gradient1
        psi0 = 2 * atan2(a, b - (c-1))
        grad_neg = -sqrt(1-c)
        grad_pos = sqrt(1-c)
        pass
    elif (a_2 + b_2 - (c+1)**2) < 1e-6:  # cyclic jumping gradient2
        psi0 = 2 * atan2(a, b - (c+1))
        grad_neg = sqrt(1+c)
        grad_pos = -sqrt(1+c)
        pass
    else:  # cyclic diffable
        pass

# Create Robot
my_fk_dh = DH_revolut()
for a, alpha, d, theta in zip(a_s, alpha_s, d_s, theta_offset_s):
    my_fk_dh.add_joint(a, alpha, d, theta)



# Test with random goal poses
if False:
    for i in range(100):
        home_new = np.random.rand(7,1)*np.pi
        T07_home = my_fk_dh.FK(home_new)
        R07 = T07_home[:3, :3]
        p07 = T07_home[:3, 3:4]
        for f in np.arange(-1.0, 1.0, 0.02):
            solu = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh)
            s = solu(f*np.pi)
            nrr = np.linalg.norm(T07_home-my_fk_dh.get_i_T_j(0,7, s))
            if nrr >1e-6:
                solu = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, verbose=True)
                s = solu(f*np.pi)
                print('PSI: {} pi'.format(f))
                print('------------')
                print('ERR', nrr)
                print('pgoal', p07.T)
                print(home_new.T)
                print(s.T)


# solu[-1] = -solu[-3]
# print(my_fk_dh.FK(solu))