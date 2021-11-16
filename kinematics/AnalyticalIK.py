import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, acos, sqrt, atan2, asin, atan ,tan

from DH import DH_revolut
from utilities import R_joints, L_joints, JOINTS_LIMITS
from utilities import skew, vec, clip_c
from random import random
from Sets import ContinuousSet

np.set_printoptions(precision=4, suppress=True)

# DH Params
pi2 = np.pi/2
th3_offset = np.pi/6
d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186
a_s            = [0.0] * 7
alpha_s        = [pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
theta_s = [0.0, -pi2, pi2, 0.0, -pi2, 0.0, 0.0]
offsets = [0.0, 0.0, th3_offset, 0.0, 0.0, 0.0, 0.0]

eps_psi = 1e-4
delta_psi = 5.0/180*np.pi


def IK_anallytical(p07_d, R07_d, DH_model, GC2=1.0, GC4=1.0, GC6=1.0, verbose=False, p06=None, p07=None):
    """
        Implementation from paper: "Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators...
        With Joint Limits and Its Application to Redundancy Resolution",
        IEEE TRANSACTIONS ON ROBOTICS, VOL. 24, NO. 5, OCTOBER 2008

    Args:
        o_p_goal ([R^3]): the goal position in base frame [3x1]
        o_R_goal ([SO3]): the goal orientation in origin frame [3x3]

    Considering the limits for each joints is considered in the analytical solution.
    - One exception is the 3rd joint of apollo which has an offset of pi/6 (has to be rotated with pi/6 in order to match the real one).
        * Offset the joint range accordingly
        * Offset the solution accordingly
    """
    plot_params = [None]*7
    d_bs = DH_model.joint(0).d; l0bs = vec([0,   0,     d_bs])
    d_se = DH_model.joint(2).d; l3se = vec([0,  -d_se,  0])
    d_ew = DH_model.joint(4).d; l4ew = vec([0,   0,     d_ew])
    d_wt = DH_model.joint(6).d; l7wt = vec([0,   0,     d_wt])

    p07_d, R07_d = DH_model.get_goal_in_dh_base_frame(p07_d, R07_d)
    # Shoulder to Wrist axis
    x0sw = p07_d - l0bs - R07_d.dot(l7wt)  # p26

    # Elbow joint
    c_th4 = clip_c((np.linalg.norm(x0sw)**2 - d_se**2 - d_ew**2) / (2*d_se*d_ew))
    th4 = GC4*acos(c_th4)

    # Shoulder joints (reference plane)
    R34 = DH_model.get_i_R_j(3,4, [th4])

    # Theta1 and Theta2 reference
    th1_ref = 0.0 if( abs(p07_d[0,0])<1e-6 and abs(p07_d[1,0])<1e-6 ) else atan2(-x0sw[1, 0], -x0sw[0, 0])
    small_psi  = acos(clip_c((np.linalg.norm(x0sw)**2 + d_se**2 - d_ew**2) / (2*d_se*np.linalg.norm(x0sw)))) # angle betweren SW and SE(see page. 6 2Paper)
    th2_ref = atan2(-x0sw[2, 0], sqrt(x0sw[0, 0]**2 + x0sw[1, 0]**2)) - GC4*small_psi

    R03_ref = DH_model.get_i_R_j(0, 3, [th1_ref, th2_ref, -DH_model.joint(2).theta-DH_model.joint(2).offset])

    if verbose:
        print('Theta1', th1_ref)
        print('Theta2', th2_ref)
        print('Theta4:', th4)


    u0sw = x0sw/np.linalg.norm(x0sw)
    u0sw = u0sw/np.linalg.norm(u0sw)
    u0sw_skew = skew(u0sw)

    # Shoulder
    As = u0sw_skew.dot(R03_ref)
    Bs = -np.matmul(u0sw_skew.dot(u0sw_skew), R03_ref)
    Cs = R03_ref - Bs
    # Wrist
    Aw = np.matmul(R34.T, As.T.dot(R07_d))
    Bw = np.matmul(R34.T, Bs.T.dot(R07_d))
    Cw = np.matmul(R34.T, Cs.T.dot(R07_d))

    # Params
    S1 = GC2 * np.array([ As[1,1],  Bs[1,1],  Cs[1,1],  As[0,1],   Bs[0,1],  Cs[0,1]]).reshape(6,1)
    S2 =       np.array([ As[2,1],  Bs[2,1],  Cs[2,1]]).reshape(3,1)
    S3 = GC2 * np.array([ As[2,0],  Bs[2,0],  Cs[2,0],  As[2,2],   Bs[2,2],  Cs[2,2]]).reshape(6,1)
    W5 = GC6 * np.array([ Aw[0,2],  Bw[0,2],  Cw[0,2], -Aw[1,2],  -Bw[1,2], -Cw[1,2]]).reshape(6,1)
    W6 =       np.array([ Aw[2,2],  Bw[2,2],  Cw[2,2]]).reshape(3,1)
    W7 = GC6 * np.array([-Aw[2,0], -Bw[2,0], -Cw[2,0], -Aw[2,1],  -Bw[2,1], -Cw[2,1]]).reshape(6,1)

    # Feasible set satisfying the limits
    feasible_sets = [None]*7
    feasible_sets[0], plot_params[0] = tangent_type(*S1, joint=DH_model.joint(0), verbose=verbose)
    feasible_sets[1], plot_params[1] =    sine_type(*S2, joint=DH_model.joint(1), GC=GC2, verbose=verbose)
    feasible_sets[2], plot_params[2] = tangent_type(*S3, joint=DH_model.joint(2), verbose=verbose)
    # feasible_sets[3], plot_params[3] = ContinuousSet(-np.pi, np.pi, False, True), ContinuousSet(-np.pi, np.pi, False, True) if th4 in DH_model.joint(3).limit_range else ContinuousSet(), ContinuousSet()
    feasible_sets[3]  = ContinuousSet(-np.pi, np.pi, False, True) if th4 in DH_model.joint(3).limit_range else ContinuousSet()
    plot_params[3] = feasible_sets[3] if verbose else None
    feasible_sets[4], plot_params[4] = tangent_type(*W5, joint=DH_model.joint(4), verbose=verbose)
    feasible_sets[5], plot_params[5] =  cosine_type(*W6, joint=DH_model.joint(5), GC=GC6, verbose=verbose)
    feasible_sets[6], plot_params[6] = tangent_type(*W7, joint=DH_model.joint(6), verbose=verbose)
    psi_feasible_set = ContinuousSet(-np.pi, np.pi)
    for fs in feasible_sets:
        psi_feasible_set -= fs
        if verbose:
            print('fs', fs)
            print("feas" ,psi_feasible_set)

    # Lambda functions that deliver solutions for each joint given arm-angle psi
    v = lambda psi: np.array([sin(psi), cos(psi), 1.0]).reshape(1,3)
    th1 = lambda psi: atan2(v(psi).dot(S1[0:3]), v(psi).dot(S1[3:]))
    if GC2>0.0:
        th2 = ( lambda psi: GC2* asin(clip_c( v(psi).dot(S2))) )
    else:
        th2 = lambda psi: ( np.pi - asin(clip_c( v(psi).dot(S2))) ) if asin(clip_c( v(psi).dot(S2))) > 0.0 else ( -np.pi - asin(clip_c( v(psi).dot(S2))) )
    th3 = lambda psi: atan2(v(psi).dot(S3[0:3]), v(psi).dot(S3[3:])) - th3_offset
    th5 = lambda psi: atan2(v(psi).dot(W5[0:3]), v(psi).dot(W5[3:]))
    th6 = lambda psi: GC6*acos(clip_c( v(psi).dot(W6) ))
    th7 = lambda psi: atan2(v(psi).dot(W7[0:3]), v(psi).dot(W7[3:]))

    if p06 is not None:
        p06_ref = DH_model.get_i_T_j(0, 6, [th1(0.0), th2(0.0), th3(0.0), th4, th5(0.0), th6(0.0), th7(0.0)])[:3, 3]
        assert np.linalg.norm(p06-p06_ref) < 1e-6
    if p07 is not None:
        p07_ref = DH_model.get_i_T_j(0, 7, [th1(0.0), th2(0.0), th3(0.0), th4, th5(0.0), th6(0.0), th7(0.0)])[:3, 3]
        assert np.linalg.norm(p07-p07_ref) < 1e-6

    if verbose:
        plt_analIK(plot_params, psi_feasible_set)

    # print(psi_feasible_set)

    return (lambda psi: np.array([ th1(psi), th2(psi), th3(psi), th4, th5(psi), th6(psi), th7(psi)]).reshape(-1,1), psi_feasible_set)


def cosine_type(a, b, c, joint, GC=1.0, verbose=False, sine_type=False):
    feasible_set = ContinuousSet(-np.pi, np.pi, False, True)  # here we capture the feasible set of psi
    stat_psi = list()
    psi_singular = False
    if sine_type:
        if GC>0.0:
            theta_f = lambda psi: asin( clip_c(a*sin(psi) + b*cos(psi) + c))
            grad_theta_f = lambda psi: GC * (a*cos(psi) - b*sin(psi))/ cos(theta_f(psi))  # if theta_f(psi) != np.pi/2 else 1.0
        else:
            theta_f = lambda psi: ( np.pi - asin( clip_c(a*sin(psi) + b*cos(psi) + c)) ) if asin( clip_c(a*sin(psi) + b*cos(psi) + c)) > 0.0 else  ( -np.pi - asin( clip_c(a*sin(psi) + b*cos(psi) + c)) )
            grad_theta_f = lambda psi: -GC * (a*cos(psi) - b*sin(psi))/ cos(theta_f(psi))  # if theta_f(psi) != np.pi/2 else 1.0
    else:
        theta_f = lambda psi: GC * acos( clip_c(a*sin(psi) + b*cos(psi) + c))
        grad_theta_f = lambda psi: -GC * (a*cos(psi) - b*sin(psi))/ sin(theta_f(psi))  # if theta_f(psi) != 0.0 else 1.0

    at_2 = a**2
    bt_2 = b**2
    ct_2 = c**2

    ss = at_2 + bt_2

    if ss > eps_psi:  # cyclic profile
        # The idea is to split the cyclic profile in sub-intervals of monotonic profile
        if verbose:
            print('2 Stationary points(cyclic): ss>0')
        psi_min = 2*atan( (-b - sqrt(ss)) / (a) )
        psi_max = 2*atan( (-b + sqrt(ss)) / (a) )
        stat_psi = [psi_min, psi_max]

    else:  # discontinuous profile (2 possibilities)
        if verbose:
            print('Singularity: ss={}'.format(ss))
        psi_singular = 2 * atan(-b/a) # should not be avoided


    def find_root(theta, lower_lim):
        class Root():
            def __init__(self, root, lower_lim):
                self.root = root
                self.lower_lim = lower_lim
                self.grad = grad_theta_f(root)

            @property
            def enters(self):
                return (self.grad>0.0 and self.lower_lim) or (self.grad<0.0 and (not self.lower_lim))

            def __repr__(self):
                return '{}: root({}), grad({}) {}'.format('lower_lim' if self.lower_lim else 'upper_lim', self.root, self.grad, 'entering' if self.enters else 'leaving')
        if sine_type:
            fun = sin
        else:
            fun = cos
        tt = ss - (c - fun(theta))**2
        roots = list()
        if tt < 0.0:
            if verbose:
                print('NO ROOTS FOR THETA_LIMIT = {}'.format(theta))
        else:
            psi0 = 2*atan( (a + sqrt(tt)) / (fun(theta) + b - c) )
            psi1 = 2*atan( (a - sqrt(tt)) / (fun(theta) + b - c) )
            roots = [Root(psi, lower_lim) for psi in [psi0, psi1] if abs(theta_f(psi)-theta) < 1e-6]
        return roots

    # Find jumpings roots
    jumpings = sorted(find_root(-np.pi+1e-8, True) + find_root(np.pi-1e-8, False), key=lambda r: r.root)

    # Joint Limits roots
    limits = joint.limit_range - ContinuousSet(-np.pi+eps_psi, np.pi-eps_psi, False, True)
    # theta_min = joint.limit_range.a
    # theta_max = joint.limit_range.b
    theta_min = limits.a
    theta_max = limits.b
    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)

    # Joint Limits feasible sets
    # The idea is to go through all the roots and capture the feasible regions
    # of roots entering the limits. If the last root entered, the rest is feasible set.
    prev_entering = -np.pi
    limits_feasible_set = ContinuousSet()
    if len(roots)>0:
        for i, r in enumerate(roots):
            if r.enters:
                prev_entering = r.root
            else:
                limits_feasible_set.add_c_range(prev_entering, r.root)
        if roots[-1].enters:
            limits_feasible_set.add_c_range(prev_entering, np.pi)
    else:
        if theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max and theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max:
            limits_feasible_set.add_c_range(-np.pi, np.pi)
    feasible_set -= limits_feasible_set

    # Ploting
    if verbose:
        title = r'$\theta_{}$ -- '.format(joint.index+1) + ("cosine_type" if not sine_type else "sine_type") + r'$\ GC_{}$={}'.format(joint.index+1, GC)
        return feasible_set, (theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, ContinuousSet())
    else:
        return feasible_set, None


def tangent_type(an, bn, cn, ad, bd, cd, joint, verbose=False):
    feasible_set = ContinuousSet(-np.pi, np.pi, False, True)  # here we capture the feasible set of psi
    singular_feasible_set = ContinuousSet()
    stat_psi = list()

    theta_f = lambda psi: atan2(an*sin(psi) + bn*cos(psi) + cn,
                              ad*sin(psi) + bd*cos(psi) + cd)

    fn = lambda psi: an*sin(psi) + bn*cos(psi) + cn
    fd = lambda psi: ad*sin(psi) + bd*cos(psi) + cd
    ft = lambda psi: at*sin(psi) + bt*cos(psi) + ct
    grad_theta_f = lambda psi: (ft(psi)) / (fn(psi)**2 + fd(psi)**2)

    at = bd*cn - bn*cd; at_2 = at**2
    bt = an*cd - ad*cn; bt_2 = bt**2
    ct = an*bd - ad*bn; ct_2 = ct**2

    ss = at_2 + bt_2 - ct_2

    if ss > eps_psi:  # cyclic profile
        # The idea is to split the cyclic profile in sub-intervals of monotonic profile
        psi_min = 2*atan( (at - sqrt(ss)) / (bt-ct) )
        psi_max = 2*atan( (at + sqrt(ss)) / (bt-ct) )
        stat_psi = [psi_min, psi_max]
        if verbose:
            print('Stationary points(cyclic): ss({})>0'.format(ss))

    elif ss < -eps_psi:  # monotonic profile
        if verbose:
            print('No Stationary points(monotonic): ss({})<0'.format(ss))
    else:  # discontinuous profile (2 possibilities)
        theta_s_neg = atan((at*bn - bt*an) / (at*bd - bt*ad))
        psi_singular = 2 * atan(at/ (bt-ct)) # should be avoided
        if psi_singular + delta_psi > np.pi:
            singular_feasible_set.add_c_range(psi_singular+delta_psi-2*np.pi - delta_psi, psi_singular - delta_psi)
        elif psi_singular - delta_psi < -np.pi:
            singular_feasible_set.add_c_range(psi_singular + delta_psi, 2*np.pi+(psi_singular-delta_psi))
        else:
            singular_feasible_set.add_c_range(-np.pi, psi_singular-delta_psi)
            singular_feasible_set.add_c_range(psi_singular+delta_psi, np.pi)

        feasible_set -= singular_feasible_set

        if verbose:
            print('Singularity: ss={}'.format(ss))
            print(singular_feasible_set)
            print(feasible_set)
            print('theta_neg:'+ str(theta_s_neg))
            print('theta_pos:'+ str(theta_s_neg + (np.pi if theta_s_neg < 0.0 else  -np.pi)))


    def find_root(theta, lower_lim):
        class Root():
            def __init__(self, root, lower_lim):
                self.root = root
                self.lower_lim = lower_lim
                self.grad = grad_theta_f(root)

            @property
            def enters(self):
                return (self.grad>0.0 and self.lower_lim) or (self.grad<0.0 and (not self.lower_lim))

            def __repr__(self):
                return str(self.root) + '~' + ('l' if self.lower_lim else 'u')
                # return '{}: root({}), grad({})'.format('lower_lim' if self.lower_lim else 'upper_lim', self.root, self.grad)

        ap = (cd-bd)*tan(theta) + (bn-cn)
        bp = 2*(ad*tan(theta) - an)
        cp = (bd+cd)*tan(theta) - (bn+cn)

        tt = bp**2 - 4*ap*cp
        roots = list()
        if tt < 0.0:
            if verbose:
                print('NO ROOTS FOR THETA_LIMIT = {}'.format(theta))
        else:
            psi0 = 2*atan( (-bp + sqrt(tt)) / (2*ap) )
            psi1 = 2*atan( (-bp - sqrt(tt)) / (2*ap) )
            roots = [Root(psi, lower_lim) for psi in [psi0, psi1] if abs(theta_f(psi)-theta) < 1e-6]
        return roots

    # Jumpings roots
    jumpings = sorted(find_root(-np.pi+1e-8, True) + find_root(np.pi-1e-8, False), key=lambda r: r.root)

    # Joint Limits roots
    limits = joint.limit_range - ContinuousSet(-np.pi+eps_psi, np.pi-eps_psi, False, True)
    # theta_min = joint.limit_range.a
    # theta_max = joint.limit_range.b
    theta_min = limits.a + joint.offset
    theta_max = limits.b + joint.offset
    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)

    # Joint Limits feasible sets
    # The idea is to go through all the roots and capture the feasible regions
    # of roots entering the limits. If the last root entered, the rest is feasible set.
    prev_entering = -np.pi
    limits_feasible_set = ContinuousSet()
    if len(roots)>0:
        for i, r in enumerate(roots):
            if r.enters:
                prev_entering = r.root
            else:
                limits_feasible_set.add_c_range(prev_entering, r.root)
        if roots[-1].enters:
            limits_feasible_set.add_c_range(prev_entering, np.pi)
    else:
        if  theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max and theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max:
            limits_feasible_set.add_c_range(-np.pi, np.pi)
    feasible_set -= limits_feasible_set

    # Ploting
    if verbose:
        title = r'$\theta_{}$ -- tangent_type'.format(joint.index+1)
        return feasible_set, (theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, singular_feasible_set)
    else:
        return feasible_set, None


def sine_type(a, b, c, joint, GC=1.0, verbose=False):
    # GC = 1: sin(x) = cos(x-pi/2)
        # x_min = theta_lim_range.a + pi/2
        # x_max = theta_lim_range.b + pi/2
    # GC = -1: a = sin(pi-x) = cos(pi/2-x)
        # x_min = pi/2 - theta_lim_range.b
        # x_max = pi/2 - theta_lim_range.a
    return cosine_type(a, b, c, joint, GC=GC, verbose=verbose, sine_type=True)


def plot_type(axis, theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, singular_feasible_set=ContinuousSet()):
    psi_s =  np.linspace(-np.pi, np.pi, np.pi/0.01)
    thetas =   [theta_f(psi) for psi in psi_s]
    grad_thetas =   [grad_theta_f(psi) for psi in psi_s]


    ax = axis[0]
    ax2 = axis[1]

    # Limits
    ax.axhline(theta_min, color='k')
    ax.axhline(theta_max, color='k')

    # Thetas and Theta_grads
    ax.plot(psi_s, thetas, label=r'$\theta_i = f(\psi)$')
    ax.plot(psi_s, grad_thetas/np.max(np.abs(grad_thetas)), label=r'$\frac{d\theta_i}{d\psi} = g(\psi)$')

    # Singularities
    if not singular_feasible_set.empty:
        for psi in singular_feasible_set.inverse(-np.pi, np.pi):
            ax.axvspan(psi.a, psi.b, color='red', alpha=0.3, label='singularity')

    # Stationary points
    thetas = [theta_f(v) for v in stat_psi]
    stat_points = ax.scatter(stat_psi, thetas, c='b')

    # Roots
    root_psis = [r.root for r in roots]
    thetas = [theta_f(v) for v in root_psis]
    root_points = ax.scatter(root_psis, thetas, c='r')

    # Jumpings
    jumping_psis = [r.root for r in jumpings]
    thetas = [theta_f(v) for v in jumping_psis]
    jumping_points = ax.scatter(jumping_psis, thetas, c='g')

    # Feasible Set
    for psi in feasible_set.c_ranges:
        ax.axvspan(psi.a, psi.b, color='green', alpha=0.3)
    ax.set_xlabel(r'$\psi$')
    ax.set_ylabel(r'$\theta_i$')

    # Feasible Set -- Green
    for i, psi in enumerate(feasible_set.c_ranges):
        ax2.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label="_"*i + 'feasible')

    # Singularities -- Red
    if not singular_feasible_set.empty:
        for i, psi in enumerate(singular_feasible_set.inverse(-np.pi, np.pi)):
            ax2.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='red', alpha=0.3, label="_"*i + 'singularity')


def IK_heuristic1(p07_d, R07_d, DH_model, verbose=False):
    GC2_plus = ContinuousSet(-np.pi/2 , np.pi)
    GC2_minus = ContinuousSet(-np.pi , -np.pi/2) + ContinuousSet(np.pi/2 , np.pi)

    GC46_plus = ContinuousSet(0.0, np.pi)
    GC46_minus = ContinuousSet(0.0, -np.pi)

    GC2 = 1.0 if (GC2_plus - DH_model.joint(1).limit_range).size >= (GC2_minus - DH_model.joint(1).limit_range).size else -1.0
    GC4 = 1.0 if (GC46_plus - DH_model.joint(3).limit_range).size >= (GC46_minus - DH_model.joint(3).limit_range).size else 1.0
    GC6 = 1.0 if (GC46_plus - DH_model.joint(5).limit_range).size >= (GC46_minus - DH_model.joint(5).limit_range).size else -1.0

    solu_function, psi_feasible_set = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=verbose, p06=None, p07=None)
    return GC2, GC4, GC6, solu_function

def IK_heuristic2(p07_d, R07_d, DH_model):
    # Finds best branch of solutions
    biggest_feasible_set = ContinuousSet()
    solu_function = None
    GC2_final = 1.0
    GC4_final = 1.0
    GC6_final = 1.0
    GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
    for GC2, GC4, GC6 in GCs:
        sf, psi_feasible_set = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False, p06=None, p07=None)
        if psi_feasible_set.max_range().size > biggest_feasible_set.size:
            biggest_feasible_set = psi_feasible_set.max_range()
            solu_function = sf
            GC2_final = GC2
            GC4_final = GC4
            GC6_final = GC6
    return GC2_final, GC4_final, GC6_final, biggest_feasible_set, solu_function

def IK_heuristic3(p07_d, R07_d, DH_model):
    # Finds best branch of solutions (with elbo human like)
    biggest_feasible_set = ContinuousSet()
    solu_function = None
    GC2_final = 1.0
    GC4_final = 1.0
    GC6_final = 1.0
    GCs = [(i, ii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0]]
    for GC2, GC6 in GCs:
        sf, psi_feasible_set = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2_final, GC4=GC4_final, GC6=GC6, verbose=False, p06=None, p07=None)
        if psi_feasible_set.max_range().size > biggest_feasible_set.size:
            biggest_feasible_set = psi_feasible_set.max_range()
            solu_function = sf
            GC2_final = GC2_final
            GC6_final = GC6
    return GC2_final, GC4_final, GC6_final, biggest_feasible_set, solu_function

def plt_analIK(plot_params, psi_feasible_set):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(15, 10))
    outer = gridspec.GridSpec(6, 2, wspace=0.2, hspace=0.5)


    for i in range(7):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        if i == 3:
            ax = plt.Subplot(fig, outer[i])
            ax.set_title(r'$\theta_{}$ -- '.format(4) + "cosine_type")
            ax.axis('off')
            fig.add_subplot(ax)
            axis = (fig.add_subplot(inner[0,0]), fig.add_subplot(inner[0,1], projection='polar'))
            ax = axis[0]
            ax2 = axis[1]
            if not plot_params[i].empty:
                ax.axvspan(-np.pi, np.pi, color='green', alpha=0.3)
                ax2.bar(0, height=1, width=2*np.pi, bottom=0, align='edge', color='green', alpha=0.3, label="_"*i + 'singularity')
        else:
            ax = plt.Subplot(fig, outer[i])
            ax.set_title(plot_params[i][-2])
            ax.axis('off')
            fig.add_subplot(ax)
            axis = (fig.add_subplot(inner[0,0]), fig.add_subplot(inner[0,1], projection='polar'))
            plot_type(axis, *plot_params[i])


    # Total feasible set
    ax = fig.add_subplot(outer[4:,:], projection='polar')
    ax.set_title("Total feasible set")
    for psi in psi_feasible_set.c_ranges:
        ax.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label='feasible')


    # Legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    blue_patch = Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='b', markersize=12),
    red_patch = Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='r', markersize=12),
    green_patch = Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='g', markersize=12),

    lines2 = [blue_patch, red_patch, green_patch]
    labels2 = ['Statpoitns', 'Limit Roots', 'Jumpings']

    liness = []
    labelss = []
    for i in range(3):
        liness.append(lines[i])
        liness.append(lines2[i])
        labelss.append(labels[i])
        labelss.append(labels2[i])

    fig.legend(liness, labelss, loc ="lower right",
            mode=None, borderaxespad=1, ncol=3, fontsize=12)
    fig.suptitle("Feasibility ranges for each joint")
    plt.show()


if __name__ == "__main__":
    from tqdm import tqdm
    # Create Robot
    my_fk_dh = DH_revolut()
    for a, alpha, d, theta, name, offset in zip(a_s, alpha_s, d_s, theta_s, R_joints, offsets):
        my_fk_dh.add_joint(a, alpha, d, theta, JOINTS_LIMITS[name], name, offset)


    home_pose = np.array([0.0, -0.0, -np.pi/6, np.pi/2, 0.0, -0.0, 0.0]).reshape(-1,1)
    home_pose = np.array([1.0, 1.0, np.pi/6, 1.0, np.pi/4, 1.0, 2.0]).reshape(-1,1)

    print(my_fk_dh.FK(home_pose))

    # Test with random goal poses
    if True:
        GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
        for i in tqdm(range(12)):
            home_new = np.array([ j.limit_range.sample() for j in my_fk_dh.joints ]).reshape(7,1)

            # print(home_new.T)
            T07_home = my_fk_dh.FK(home_new)
            R07 = T07_home[:3, :3]
            p07 = T07_home[:3, 3:4]
            # for GC2, GC4, GC6 in GCs:
            #     IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, GC2=1, GC4=GC4, GC6=GC6, verbose=False, p06=my_fk_dh.get_i_T_j(0,6,home_new.flatten())[:3, 3], p07=my_fk_dh.get_i_T_j(0,7,home_new.flatten())[:3, 3])
            # continue
                # solu, feasible_set = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)  # , p06=my_fk_dh.get_i_T_j(0,6,home_new.flatten())[:3, 3]
                # solu, feasible_set = IK_heuristic1(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, verbose=True)
            GC2, GC4, GC6, _, _ =  IK_heuristic2(p07_d=p07, R07_d=R07, DH_model=my_fk_dh)
            solu, feasible_set = IK_anallytical(p07, R07, my_fk_dh, GC2, GC4, GC6, verbose=True)

            for f in np.arange(-1.0, 1.0, 0.02):
                s = solu(f*np.pi)
                nrr = np.linalg.norm(T07_home-my_fk_dh.FK(s))
                if nrr >1e-6:
                    print('ERR', nrr)
                    print('PSI: {} pi'.format(f))
                    print('------------')
                    print('pgoal', p07.T)
                    print(home_new.T)
                    print(s.T)
                    assert False
                    break
