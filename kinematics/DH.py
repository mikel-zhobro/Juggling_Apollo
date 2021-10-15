import numpy as np
import matplotlib.pyplot as plt
from utilities import skew, vec, clip_c, modrad
from utilities import ContinuousSet
from utilities import JOINTS_LIMITS, R_joints, L_joints
from math import sin, cos, atan, acos, asin, sqrt, atan2
from tangent_type import tangent_type
from cosine_type import cosine_type

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
        def __init__(self, a, alpha, d, theta, name):
            self.a = a
            self.alpha = alpha
            self.d = d
            self.theta = theta
            self.name = name
            self.index = DH_revolut.n_joints
            DH_revolut.n_joints += 1

        @property
        def limit_range(self):
            return ContinuousSet(JOINTS_LIMITS[self.name][0], JOINTS_LIMITS[self.name][1], False, False)

        def __repr__(self):
            return '{}. {}: a({}), b({}), d({}), theta({})'.format(self.index, self.name, self.a, self.alpha, self.d, self.theta)

    def __init__(self):
        self.joints = []

    def joint(self, i):
        return self.joints[i]

    def add_joint(self, a, alpha, d, theta, name):
        self.joints.append(self.Joint(a, alpha, d, theta, name))

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

    def plot(self):
        T0wshoulder = self.get_i_T_j(0, 2)
        T0wselbo    = self.get_i_T_j(0, 4)
        T0wswrist   = self.get_i_T_j(0, 6)
        pass

def IK_anallytical(p07_d, R07_d, DH_model, GC2=1.0, GC4=1.0, GC6=1.0, verbose=False):
    """
        Implementation from paper: "Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators...
        With Joint Limits and Its Application to Redundancy Resolution",
        IEEE TRANSACTIONS ON ROBOTICS, VOL. 24, NO. 5, OCTOBER 2008

    Args:
        o_p_goal ([R^3]): the goal position in base frame [3x1]
        o_R_goal ([SO3]): the goal orientation in origin frame [3x3]
    """
    GC2  = np.sign((DH_model.joint(1).limit_range.a + DH_model.joint(1).limit_range.b)/2)
    GC4  = np.sign((DH_model.joint(3).limit_range.a + DH_model.joint(3).limit_range.b)/2)
    GC6  = np.sign((DH_model.joint(5).limit_range.a + DH_model.joint(5).limit_range.b)/2)

    GC2 = -1.0 if GC2==0.0 else GC2
    GC4 = -1.0 if GC4==0.0 else GC4
    GC6 = -1.0 if GC6==0.0 else GC6
    l0bs = vec([0,   0,     d_bs])
    l3se = vec([0,  -d_se,  0])
    l4ew = vec([0,   0,     d_ew])
    l7wt = vec([0,   0,     d_wt])

    # shoulder to wrist axis
    x0sw = p07_d - l0bs - R07_d.dot(l7wt)

    # Elbow joint
    c_th4 = clip_c((np.linalg.norm(x0sw)**2 - d_se**2 - d_ew**2) / (2*d_se*d_ew))
    th4 = GC4*acos(c_th4)
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
        print('Theta2', atan2(s2, c2))
        print('Theta4:', th4)


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

    feasible_sets = [None]*7
    feasible_sets[0] = tangent_type(-As[1,1], -Bs[1,1], -Cs[1,1], -As[0,1], -Bs[0,1], -Cs[0,1], DH_model.joint(0).limit_range, verbose)
    feasible_sets[1] = cosine_type(-As[2,1], -Bs[2,1], -Cs[2,1], DH_model.joint(1).limit_range, verbose=verbose)
    feasible_sets[2] = tangent_type(As[2,2], Bs[2,2], Cs[2,2], -As[2,0], -Bs[2,0], -Cs[2,0], DH_model.joint(2).limit_range, verbose)
    feasible_sets[3] = ContinuousSet(-np.pi, np.pi, False, True) if th4 in DH_model.joint(3).limit_range else ContinuousSet()
    feasible_sets[4] = tangent_type(Aw[1,2], Bw[1,2], Cw[1,2], Aw[0,2], Bw[0,2], Cw[0,2], DH_model.joint(4).limit_range, verbose)
    feasible_sets[5] = cosine_type(Aw[2,2], Bw[2,2], Cw[2,2], DH_model.joint(5).limit_range, verbose=verbose)
    feasible_sets[6] = tangent_type(Aw[2,1], Bw[2,1], Cw[2,1], -Aw[2,0], -Bw[2,0], -Cw[2,0], DH_model.joint(6).limit_range, verbose)
    psi_feasible_set = ContinuousSet(-np.pi, np.pi)
    for fs in feasible_sets:
        psi_feasible_set -= fs

    # 1. shoulder solutions
    t11 = lambda psi: -GC2*( As[1,1]*sin(psi)  + Bs[1,1]*cos(psi) + Cs[1,1] )
    t12 = lambda psi: -GC2*( As[0,1]*sin(psi)  + Bs[0,1]*cos(psi) + Cs[0,1] )
    c22 = lambda psi:  clip_c( -As[2,1]*sin(psi) - Bs[2,1]*cos(psi) - Cs[2,1] )
    t31 = lambda psi:  GC2*( As[2,2]*sin(psi)  + Bs[2,2]*cos(psi) + Cs[2,2] )
    t32 = lambda psi: -GC2*( As[2,0]*sin(psi)  + Bs[2,0]*cos(psi) + Cs[2,0] )

    # 2. wrist solutions
    t51 = lambda psi: GC6*( Aw[1,2]*sin(psi)  + Bw[1,2]*cos(psi) + Cw[1,2] )
    t52 = lambda psi: GC6*( Aw[0,2]*sin(psi)  + Bw[0,2]*cos(psi) + Cw[0,2] )
    c6 =  lambda psi: clip_c( Aw[2,2]*sin(psi)  + Bw[2,2]*cos(psi) + Cw[2,2] )
    t71 = lambda psi: GC6*(  Aw[2,1]*sin(psi)  + Bw[2,1]*cos(psi) + Cw[2,1] )
    t72 = lambda psi: -GC6*( Aw[2,0]*sin(psi)  + Bw[2,0]*cos(psi) + Cw[2,0] )

    th1 = lambda psi: atan2(t11(psi), t12(psi))
    th2 = lambda psi: GC2*acos(c22(psi))
    th3 = lambda psi: atan2(t31(psi),t32(psi))
    th5 = lambda psi: atan2(t51(psi), t52(psi))
    th6 = lambda psi: GC6*acos(c6(psi))
    th7 = lambda psi: atan2(t71(psi), t72(psi))

    print('FEASIBLE SET', psi_feasible_set)
    if not psi_feasible_set.empty:
        plt.figure()
        plt.subplot(111, polar=True)
        for psi in psi_feasible_set.c_ranges:
            plt.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label='feasible')
        plt.title('Final Feasible Set [GC2({}), GC4({}), GC6({}])'.format(GC2, GC4, GC6))
        plt.show()

    return (lambda psi: np.array([ th1(psi), th2(psi), th3(psi), th4, th5(psi), th6(psi), th7(psi)]).reshape(-1,1), psi_feasible_set)


def IK(p07_d, R07_d, DH_model, vis=False):
    feasible_armangles = list()
    solu, psi_feasible_set = IK_anallytical(p07_d, R07_d, DH_model)

    if vis:
        p = [plt.plot(vals, qs[:, i], label=r'$\theta_{}$'.format(i)) for i, joint in enumerate(DH_model.joints)]
        plt.axhspan(DH_model.joints[1].limit_range.a, DH_model.joints[1].limit_range.b, color=p[1][0].get_color(), alpha=0.3)
        plt.legend()
        plt.xlabel(r'$\psi$')
        plt.ylabel(r'$\theta_i$')
        plt.show()
    for v, q in zip(vals, qs):
        feasible = [q[i] in joint.limit_range for i, joint in enumerate(DH_model.joints)]
        # print(feasible)
        if all(feasible):
            feasible_armangles.append(v)

    return feasible_armangles, solu


def bisection_method(f, value, interval, eps=1e-12):
    """ Finds the root of f(psi) - value

    Args:
        f ([function]):  f(psi)
        value ([float]): value = f(psi0)
        interval ([ContinuousSet]): interval of psi to look into
    Returns:
        [float]: root of f(psi) - value
    """
    min_r = interval.a
    max_r = interval.b

    increasing = f(min_r) <= value <= f(max_r)
    decreasing = f(min_r) >= value >= f(max_r)
    assert increasing or decreasing, 'Make sure that value exist in the given interval'

    start = min_r
    endd = max_r
    err = 1.0
    while abs(err) > eps:
        psi = (start + endd) / 2.0
        err = f(psi) - value

        if increasing:
            if err > 0.0:
                endd = psi
            else:
                start = psi
        else:
            if err > 0.0:
                start = psi
            else:
                endd = psi
    return psi


def feasible_set_for_monotonic_function(f, FeasibleOutRange, InputRange):
    """Find feasible input range that satisfies the feasible_out_range

    Args:
        f ([function]): f(in) = out
        FeasibleOutRange ([ContinuousSet]): out in FeasibleOutRange
        InputRange ([ContinuousSet]): in in InputRange

    Returns:
        [ContinuousSet]: Feasible Input Range
    """
    PossibleOutRange = ContinuousSet(f(InputRange.a), f(InputRange.b), InputRange.a_incl, InputRange.b_incl)
    FeasiblePossibleOutRange = FeasibleOutRange - PossibleOutRange
    assert not FeasiblePossibleOutRange.empty, 'No feasible region exists for output range {} in the input range {}'.format(FeasibleOutRange, InputRange)

    psi0 = bisection_method(f, value=FeasiblePossibleOutRange.a, interval=InputRange)
    psi1 = bisection_method(f, value=FeasiblePossibleOutRange.b, interval=InputRange)

    FeasibleInRange = InputRange - ContinuousSet(psi0, psi1)
    return FeasibleInRange


# Create Robot
my_fk_dh = DH_revolut()
for a, alpha, d, theta, name in zip(a_s, alpha_s, d_s, theta_offset_s, R_joints):
    my_fk_dh.add_joint(a, alpha, d, theta, name)


# Test with random goal poses
if False:
    import matplotlib.pyplot as plt
    for i in range(1):
        home_new = np.random.rand(7,1)*np.pi/2
        home_new[1] = -1
        T07_home = my_fk_dh.FK(home_new)
        R07 = T07_home[:3, :3]
        p07 = T07_home[:3, 3:4]
        print(home_new.T)

        feasible_set, solu = IK(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, vis=True)
        feasible_solutions = np.array([solu(v) for v in feasible_set]).squeeze()
        for sv in feasible_solutions:
            nrr = np.linalg.norm(T07_home-my_fk_dh.FK(sv))
            # print(nrr)
            if nrr > 1e-6:
                print('ERROR', nrr)
        if feasible_solutions.size ==0:
            print('NO SOLUTIONS')
        else:
            for i in range(7):
                plt.plot(feasible_set, feasible_solutions[:,i], label ='joint_{}'.format(i+1))
            plt.legend()
            plt.show()

if True:
    GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
    for i in range(10000):
        home_new = np.random.rand(7,1)*np.pi

        home_new = np.array([ j.limit_range.sample() for j in my_fk_dh.joints ]).reshape(7,1)
        print(home_new.T)
        T07_home = my_fk_dh.FK(home_new)
        R07 = T07_home[:3, :3]
        p07 = T07_home[:3, 3:4]

        # for GC2, GC4, GC6 in GCs:
        solu, feasible_set = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, verbose=True)
        for f in np.arange(-1.0, 1.0, 0.02):
            s = solu(f*np.pi)
            nrr = np.linalg.norm(T07_home-my_fk_dh.FK(s))
            if nrr >1e-6:
                print('PSI: {} pi'.format(f))
                print('------------')
                print('ERR', nrr)
                print('pgoal', p07.T)
                print(home_new.T)
                print(s.T)


if False:
    import matplotlib.pyplot as plt
    for i in range(1000):
        home_new = np.random.rand(7,1)*np.pi
        T07_home = my_fk_dh.FK(home_new)
        R07 = T07_home[:3, :3]
        p07 = T07_home[:3, 3:4]

        solu, coefs = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh)
        vals = np.linspace(-np.pi, np.pi, np.pi/0.01, endpoint=True)
        solu_v = np.array([solu(v) for v in vals]).squeeze()
        for sv in solu_v:
            nrr = np.linalg.norm(T07_home-my_fk_dh.FK(sv))
            if nrr > 1e-6:
                print('ERROR')

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        for i in range(6):
            if i in [0, 2, 4, 6]:
                coef = coefs[i]
                ss, ret_vl, ret_fl = tangent_type(*coef)
                typ = None
                if ss<-1e-3:
                    typ = 'cont'
                elif ss>1e-3:
                    typ = 'cyclic'
                else:
                    typ = 'jump'
                axs[0].plot(vals, solu_v[:, i], label='{}. [{}] ss={}'.format(i,  typ, ss))
                axs[0].scatter(ret_vl, ret_fl)
            elif i !=3:
                axs[1].plot(vals, solu_v[:, i], label='costype, start-end: {}'.format(solu_v[:, i][0] -solu_v[:, i][-1]))

            # plt.plot(vals, solu_v[:, [0,2,6]])
        axs[0].legend()
        axs[1].legend()
        plt.show()





# f = lambda x: x**2
# # x0 = bisection_method(f, f(2.11), (0, 6), eps=1e-18)
# print(ContinuousSet(2.0,8.0, False))
# print(feasible_set_for_monotonic_function(f, ContinuousSet(16.0, 128.0), ContinuousSet(2.0,8.0, False, False)))
