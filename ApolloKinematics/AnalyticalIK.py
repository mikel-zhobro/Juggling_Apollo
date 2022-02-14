import numpy as np
from math import sin, cos, acos, sqrt, atan2, asin, atan ,tan
from random import random

from DHFK import DH_revolut
from utilities import R_joints, L_joints, JOINTS_LIMITS
from utilities import skew, vec, mod2Pi#, clip_c
from Sets import ContinuousSet

clip_c = mod2Pi
np.set_printoptions(precision=4, suppress=True)

# DH Params
th3_offset = np.pi/6
eps_psi = 1e-4
delta_psi = 5.0/180*np.pi


def IK_anallytical(p07_d, R07_d, DH_model, GC2=1.0, GC4=1.0, GC6=1.0, verbose=False, p06=None, p07=None, considered_joints=list(range(7))):
    """
        Implementation from paper: "Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators...
        With Joint Limits and Its Application to Redundancy Resolution",
        IEEE TRANSACTIONS ON ROBOTICS, VOL. 24, NO. 5, OCTOBER 2008

    Args:
        p07_d ([R^3]): the goal position in world frame [3x1]
        R07_d ([SO3]): the goal orientation in world frame [3x3]

    Considering the limits for each joints is considered in the analytical solution.
    - One exception is the 3rd joint of apollo which has an offset of pi/6 (has to be rotated with pi/6 in order to match the real one).
        * Offset the joint range accordingly
        * Offset the solution accordingly
    """
    # Transform goal in dh base frame
    p07_d, R07_d = DH_model.get_goal_in_dh_base_frame(p07_d, R07_d)
    p07_d, R07_d = DH_model.get_goal_in_dhtcp_frame(p07_d, R07_d)

    # Prepare desired wrist position
    d_wt = DH_model.joint(6).d; l7wt = vec([0,   0,  d_wt])
    p0w_d = p07_d - R07_d.dot(l7wt)

    # 1. Shoulder and elbo joints (set the wrist position)
    th1, th2, th3, th4, As, Bs, Cs, fset_1_4, plot_params_1_4, root_funcs_1_4 = IK_elbow_shoulder(DH_model, p0w_d, GC2, GC4, verbose)

    # 2. Wrist joint (finds th5, th6, th7 as function of psi to set the orientation of the wrist(== orientation of the tcp))
    R34 = DH_model.get_i_R_j(3,4, [th4])  # reference plane
    th5, th6, th7, fset_5_7, plot_params_5_7, root_funcs_5_7 = IK_wrist(DH_model, R07_d, R34, As, Bs, Cs, GC6, verbose)

    # Feasible set satisfying the limits
    feasible_sets = fset_1_4 + fset_5_7
    root_funcs = root_funcs_1_4 + root_funcs_5_7

    psi_feasible_set = ContinuousSet(-np.pi, np.pi)
    for kkk in considered_joints:
        fs = feasible_sets[kkk]
        psi_feasible_set -= fs
        if verbose:
            print('fs', fs)
            print("feas" ,psi_feasible_set)

    if p06 is not None:
        p06_ref = DH_model.get_i_T_j(0, 6, [th1(0.0), th2(0.0), th3(0.0), th4, th5(0.0), th6(0.0), th7(0.0)])[:3, 3]
        assert np.linalg.norm(p06-p06_ref) < 1e-6
    if p07 is not None:
        p07_ref = DH_model.get_i_T_j(0, 7, [th1(0.0), th2(0.0), th3(0.0), th4, th5(0.0), th6(0.0), th7(0.0)])[:3, 3]
        assert np.linalg.norm(p07-p07_ref) < 1e-6

    if verbose:
        plot_params = plot_params_1_4 + plot_params_5_7
        plt_analIK(plot_params, psi_feasible_set)

    return (lambda psi: np.array([ th1(psi), th2(psi), th3(psi), th4, th5(psi), th6(psi), th7(psi)]).reshape(-1,1), psi_feasible_set, root_funcs)


def IK_elbow_shoulder(DH_model, p0w_d, GC2, GC4, verbose):
    """ Computes shoulder-elbow joint angles to realize a certain wrist position.
    Args:
        p0w_d ([np.array]): desired wrist world position
        GC2, GC4 ([+-1]): branch of solution

    Returns:
        th1, th2, th3, th4, As, Bs, Cs, fset_1_4, plot_params_1_4
    """
    d_bs = DH_model.joint(0).d; l0bs = vec([0,   0,     d_bs])
    d_se = DH_model.joint(2).d; l3se = vec([0,  -d_se,  0])
    d_ew = DH_model.joint(4).d; l4ew = vec([0,   0,     d_ew])

    # Make sure position is within the reachable space
    dst1 = d_se + d_ew
    dst2 = np.linalg.norm(p0w_d-l0bs)
    assert dst1  >= dst2, 'Unreachable wrist position {} >= {}'.format(dst1, dst2)

    # Shoulder to Wrist axis
    x0sw = p0w_d - l0bs

    # 1. Elbow joint (finds th4 to set the wrist position)
    th4, fset_4, plot_params_4, root_func_4 = IK_elbow(DH_model, x0sw, d_se, d_ew, GC4, verbose)

    # 2. Shoulder joint (finds th1, th2, th3 as function of psi to set the orientation of the shoulder)
    As, Bs, Cs, th1, th2, th3, fset_1_3, plot_params_1_3, root_funcs_1_3 = IK_shoulder(DH_model, x0sw, d_se, d_ew, GC2, GC4, verbose)

    return th1, th2, th3, th4, As, Bs, Cs, fset_1_3+fset_4, plot_params_1_3+plot_params_4, root_funcs_1_3+root_func_4


def IK_elbow(DH_model, x0sw, d_se, d_ew, GC4, verbose):
    """ Computes the elbo angle to realize a certain shoulder-wrist length.

    Args:
        x0sw ([np.array]): shoulder to wrist axis (3,1)
        GC4 (+-1): whether elbo is inside or outside

    Returns:
        R34, th4, fset_4, plot_params_4
    """
    c_th4 = clip_c((np.linalg.norm(x0sw)**2 - d_se**2 - d_ew**2) / (2*d_se*d_ew))
    th4 = GC4*acos(c_th4)
    root_func_4 = [None]
    fset_4 = [ContinuousSet(-np.pi, np.pi, False, True) if th4 in DH_model.joint(3).limit_range else ContinuousSet()]
    plot_params_4 = [(DH_model.joint(3).limit_range, th4) if verbose else None]
    if verbose:
        print('Theta4:', th4)
    return th4, fset_4, plot_params_4, root_func_4


def IK_shoulder(DH_model, x0sw, d_se, d_ew, GC2, GC4, verbose):
    """Solves 2DOF IK for th1_ref, th2_ref given the th3_ref=np.pi/2 + np.pi/6 and th4_ref(from above) in order to set the wrist position.
       Then returns th1 th2 th3 as functions of psi, which describe the rotation of elbow around sholder-wrist axis with psi.

    Args:
        x0sw ([np.array]): shoulder-wrist vector in world frame (3,1)
        d_se, d_ew ([float]): shoulder-elbow and elbow-wrist joint lengths
        GC2, GC4: branch of solutions

    Returns:
        As, Bs, Cs, th1, th2, th3, feasible_sets, plot_params
    """
    # Theta1 and Theta2 reference
    # th1_ref = 0.0 if( abs(p07_d[0,0])<1e-6 and abs(p07_d[1,0])<1e-6 ) else atan2(-x0sw[1, 0], -x0sw[0, 0])
    th1_ref = 0.0 if( abs(x0sw[0, 0])<1e-6 and abs(x0sw[1, 0])<1e-6 ) else atan2(-x0sw[1, 0], -x0sw[0, 0])
    small_psi  = acos(clip_c((np.linalg.norm(x0sw)**2 + d_se**2 - d_ew**2) / (2*d_se*np.linalg.norm(x0sw)))) # angle betweren SW and SE(see page. 6 2Paper)
    th2_ref = atan2(-x0sw[2, 0], sqrt(x0sw[0, 0]**2 + x0sw[1, 0]**2)) - GC4*small_psi

    if verbose:
        print('Theta1', th1_ref)
        print('Theta2', th2_ref)

    R03_ref = DH_model.get_i_R_j(0, 3, [th1_ref, th2_ref, -DH_model.joint(2).theta-DH_model.joint(2).offset])

    # Rotation axis to rotation matrix
    u0sw = x0sw/np.linalg.norm(x0sw)
    u0sw = u0sw/np.linalg.norm(u0sw)
    u0sw_skew = skew(u0sw)

    # Shoulder R03(psi) = As*sin(psi) + Bs*cos(psi) + C_s
    As = u0sw_skew.dot(R03_ref)
    Bs = -np.matmul(u0sw_skew.dot(u0sw_skew), R03_ref)
    Cs = R03_ref - Bs

    # Params
    S1 = GC2 * np.array([ As[1,1],  Bs[1,1],  Cs[1,1],  As[0,1],   Bs[0,1],  Cs[0,1]]).reshape(6,1)
    S2 =       np.array([ As[2,1],  Bs[2,1],  Cs[2,1]]).reshape(3,1)
    S3 = GC2 * np.array([ As[2,0],  Bs[2,0],  Cs[2,0],  As[2,2],   Bs[2,2],  Cs[2,2]]).reshape(6,1)

    # Feasible set satisfying the limits
    feasible_sets = [None]*3; plot_params = [None]*3; root_funcs = [None]*3
    feasible_sets[0], plot_params[0], root_funcs[0] = tangent_type(*S1, joint=DH_model.joint(0), verbose=verbose)
    feasible_sets[1], plot_params[1], root_funcs[1] =    sine_type(*S2, joint=DH_model.joint(1), GC=GC2, verbose=verbose)
    feasible_sets[2], plot_params[2], root_funcs[2] = tangent_type(*S3, joint=DH_model.joint(2), verbose=verbose)

    # Lambda functions that deliver solutions for each joint given arm-angle psi
    v = lambda psi: np.array([sin(psi), cos(psi), 1.0]).reshape(1,3)
    th1 = lambda psi: atan2(v(psi).dot(S1[0:3]), v(psi).dot(S1[3:]))
    if GC2>0.0:
        th2 = ( lambda psi: asin(clip_c( v(psi).dot(S2))) )
    else:
        th2 = lambda psi: ( np.pi - asin(clip_c( v(psi).dot(S2))) ) if asin(clip_c( v(psi).dot(S2))) > 0.0 else ( -np.pi - asin(clip_c( v(psi).dot(S2))) )
    th3 = lambda psi: atan2(v(psi).dot(S3[0:3]), v(psi).dot(S3[3:])) - th3_offset

    return As, Bs, Cs, th1, th2, th3, feasible_sets, plot_params, root_funcs


def IK_wrist(DH_model, R07_d, R34, As, Bs, Cs, GC6, verbose):
    """ Computes th5 th6 th7 as functions of psi, which describe the rotation of elbow around sholder-wrist axis with psi.

    Args:
        R07_d ([np.array(3,3)]): desired end-effector rotation matrix
        R34 ([np.array(3,3)]): elbow rotation matrix
        As, Bs, Cs ([type]): reference cos and sin matrix of the rotation around shoulder wrist axis
        GC6 ([type]): branch of solutions

    Returns:
        th5 th6 th7 as functions of psi
    """

    # Wrist R47(psi) = Aw*sin(psi) + Bw*cos(psi) + C_w
    Aw = np.matmul(R34.T, As.T.dot(R07_d))
    Bw = np.matmul(R34.T, Bs.T.dot(R07_d))
    Cw = np.matmul(R34.T, Cs.T.dot(R07_d))

    # Params
    W5 = GC6 * np.array([ Aw[0,2],  Bw[0,2],  Cw[0,2], -Aw[1,2],  -Bw[1,2], -Cw[1,2]]).reshape(6,1)
    W6 =       np.array([ Aw[2,2],  Bw[2,2],  Cw[2,2]]).reshape(3,1)
    W7 = GC6 * np.array([-Aw[2,0], -Bw[2,0], -Cw[2,0], -Aw[2,1],  -Bw[2,1], -Cw[2,1]]).reshape(6,1)

    # Feasible set satisfying the limits
    f_set = [None]*3; plot_params = [None]*3; root_funcs = [None]*3
    f_set[0], plot_params[0], root_funcs[0] = tangent_type(*W5, joint=DH_model.joint(4), verbose=verbose)
    f_set[1], plot_params[1], root_funcs[1] =  cosine_type(*W6, joint=DH_model.joint(5), GC=GC6, verbose=verbose)
    f_set[2], plot_params[2], root_funcs[2] = tangent_type(*W7, joint=DH_model.joint(6), verbose=verbose)

    # Lambda functions that deliver solutions for each joint given arm-angle psi
    v = lambda psi: np.array([sin(psi), cos(psi), 1.0]).reshape(1,3)
    th5 = lambda psi: atan2(v(psi).dot(W5[0:3]), v(psi).dot(W5[3:]))
    th6 = lambda psi: GC6*acos(clip_c( v(psi).dot(W6) ))
    th7 = lambda psi: atan2(v(psi).dot(W7[0:3]), v(psi).dot(W7[3:]))

    return th5, th6, th7, f_set, plot_params, root_funcs


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
        psi_min = 2*atan( (-b - sqrt(ss)) / (a + 1e-10))
        psi_max = 2*atan( (-b + sqrt(ss)) / (a + 1e-10))
        stat_psi = [psi_min, psi_max]

    else:  # discontinuous profile (2 possibilities)
        if verbose:
            print('Singularity: ss={}'.format(ss))
        # psi_singular = 2 * atan(-b/(a + 1e-10)) # should not be avoided


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

    # Root function for inverse calculation theta -> psi
    root_f = lambda theta: [round(root.root, 7) for root in find_root(theta, False)]
    # Ploting
    if verbose:
        title = r'$\theta_{}$ -- '.format(joint.index+1) + ("cosine_type" if not sine_type else "sine_type") + r'$\ GC_{}$={}'.format(joint.index+1, GC)
        return feasible_set, (theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, ContinuousSet()), root_f
    else:
        return feasible_set, None, root_f


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
    # Jumps
    # add jumps in the root_jumps list
    tmp = list()
    for i in range(0, len(jumpings), 2):
        jump1 = jumpings[i]; th1_tmp = theta_f(jump1.root)
        jump2 = jumpings[i+1]; th2_tmp = theta_f(jump2.root)
        if (th1_tmp < theta_min) and (th2_tmp < theta_max):
            jump2.enters=True
            tmp.append(jump2)
        elif (th1_tmp > theta_max) and (th2_tmp > theta_min):
            jump2.enters=True
            tmp.append(jump2)
        elif (th1_tmp > theta_min) and (th2_tmp > theta_max):
            jump1.enters=False
            tmp.append(jump1)
        elif (th1_tmp < theta_max) and (th2_tmp < theta_min):
            jump1.enters=False
            tmp.append(jump1)

    # Roots & jumps
    # The idea is to go through all the roots&jumps and capture the feasible regions
    # of roots entering the limits. If the last root entered, the rest is feasible set.
    root_jumps  = sorted(tmp + roots, key=lambda r: r.root)
    prev_entering = -np.pi
    limits_feasible_set = ContinuousSet()
    if len(root_jumps)>0:
        for i, r in enumerate(root_jumps):
            if r.enters:
                prev_entering = r.root
            else:
                if prev_entering is not None:
                    limits_feasible_set.add_c_range(prev_entering, r.root)
                prev_entering = None
        if root_jumps[-1].enters:
            if prev_entering is not None:
                limits_feasible_set.add_c_range(prev_entering, np.pi)
    else:
        if  theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max and theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max:
            limits_feasible_set.add_c_range(-np.pi, np.pi)
    feasible_set -= limits_feasible_set

    # Root function for inverse calculation theta -> psi
    root_f = lambda theta: [round(root.root, 7) for root in find_root(theta, False)]
    # Ploting
    if verbose:
        title = r'$\theta_{}$ -- tangent_type'.format(joint.index+1)
        return feasible_set, (theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, singular_feasible_set), root_f
    else:
        return feasible_set, None, root_f


def sine_type(a, b, c, joint, GC=1.0, verbose=False):
    # GC = 1: sin(x) = cos(x-pi/2)
        # x_min = theta_lim_range.a + pi/2
        # x_max = theta_lim_range.b + pi/2
    # GC = -1: a = sin(pi-x) = cos(pi/2-x)
        # x_min = pi/2 - theta_lim_range.b
        # x_max = pi/2 - theta_lim_range.a
    return cosine_type(a, b, c, joint, GC=GC, verbose=verbose, sine_type=True)


def plot_type(axis, index, theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, singular_feasible_set=ContinuousSet()):
    psi_s =  np.linspace(-np.pi, np.pi, np.pi/0.01)
    thetas =   [theta_f(psi) for psi in psi_s]
    grad_thetas =   [grad_theta_f(psi) for psi in psi_s]

    ax = axis[0]; ax.set_ylim([-np.pi, np.pi])
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
    ax.set_xlabel(r'$\psi$',y=0.85)
    ax.xaxis.set_label_coords(0.02,-0.1)
    ax.set_ylabel(r'$\theta_%d$'%(index+1),rotation=0)

    # Feasible Set -- Green
    for i, psi in enumerate(feasible_set.c_ranges):
        ax2.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label="_"*i + 'feasible')

    # Singularities -- Red
    if not singular_feasible_set.empty:
        for i, psi in enumerate(singular_feasible_set.inverse(-np.pi, np.pi)):
            ax2.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='red', alpha=0.3, label="_"*i + 'singularity')
    ax2.set_xticklabels([r'$0^\circ$', '', '', '', r'$180^\circ$', '', '', ''])
    ax2.set_yticklabels([])
    ax2.set_title(r"$\psi$",y=-0.26)


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

def IK_heuristic2(p07_d, R07_d, DH_model, considered_joints=list(range(7))):
    # Finds best branch of solutions
    biggest_feasible_set = ContinuousSet()
    siz = -15555.
    solu_function = None
    GC2_final = 1.0
    GC4_final = 1.0
    GC6_final = 1.0
    GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
    for GC2, GC4, GC6 in GCs:
        sf, psi_feasible_set, root_funcs = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False, p06=None, p07=None, considered_joints=considered_joints)
        if psi_feasible_set.max_range().size > siz:
            biggest_feasible_set = psi_feasible_set.max_range()
            siz = biggest_feasible_set.size
            solu_function = sf
            GC2_final = GC2
            GC4_final = GC4
            GC6_final = GC6
    return GC2_final, GC4_final, GC6_final, biggest_feasible_set, solu_function

def IK_heuristic3(p07_d, R07_d, DH_model, considered_joints=list(range(7))):
    # Finds best branch of solutions (with elbo human like)
    biggest_feasible_set = ContinuousSet()
    siz = -15555.
    solu_function = None
    GC2_final = 1.0
    GC4_final = 1.0
    GC6_final = 1.0
    GCs = [(i, ii) for i in [1.0] for ii in [-1.0, 1.0]]
    for GC2, GC6 in GCs:
        sf, psi_feasible_set, root_funcs = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4_final, GC6=GC6, verbose=False, p06=None, p07=None, considered_joints=considered_joints)
        if psi_feasible_set.max_range().size > siz:
            biggest_feasible_set = psi_feasible_set.max_range()
            siz = biggest_feasible_set.size
            solu_function = sf
            GC2_final = GC2
            GC6_final = GC6
    return GC2_final, GC4_final, GC6_final, biggest_feasible_set, solu_function

def IK_find_psi_and_GCs(p07_d, R07_d, q_init, DH_model):
    # Finds best branch of solutions and psi for given q_init
    th2 = q_init[1]
    GC2 = -1. if np.abs(th2) >= np.pi/2. else 1.
    GC4 = np.sign(q_init.squeeze()[3])
    GC6 = np.sign(q_init.squeeze()[5])

    sf, psi_feasible_set, root_funcs = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)

    root_list = [set(rf(mod2Pi(thi+j.offset))) for thi, rf, j in zip(q_init, root_funcs, DH_model.joints) if rf is not None]
    u = set.intersection(*root_list[:])
    if len(u) == 0:
        u = root_list[-3]

    assert len(u)==1, "This cannot happen! {}".format(root_list)
    psi = u.pop()
    # if np.linalg.norm(q_init-sf(psi)) > 1e-5:
    #     sf, psi_feasible_set, root_funcs = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=True)
    #     print(q_init[2], sf(psi)[2])
    #     pass

    return psi, GC2, GC4, GC6, psi_feasible_set.max_range(), sf


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
            ax = axis[0]; ax.set_ylim([-np.pi, np.pi])
            ax2 = axis[1]
            if not plot_params[i][0].empty:
                ax.axhline(plot_params[3][0].a, color='k')
                ax.axhline(plot_params[3][0].b, color='k')
                ax.plot(np.linspace(-np.pi, np.pi, np.pi/0.01), [plot_params[3][1]]*int(np.pi/0.01))
                ax.axvspan(-np.pi, np.pi, color='green', alpha=0.3)
                ax.set_xlabel(r'$\psi$')
                ax.xaxis.set_label_coords(0.02,-0.1)
                ax.set_ylabel(r'$\theta_%d$' %4,rotation=0)
                ax2.bar(0, height=1, width=2*np.pi, bottom=0, align='edge', color='green', alpha=0.3, label="_"*i + 'singularity')
                ax2.set_xticklabels([r'$0^\circ$', '', '', '', r'$180^\circ$', '', '', ''])
                ax2.set_title(r"$\psi$",y=-0.26,)
        else:
            ax = plt.Subplot(fig, outer[i])
            ax.set_title(plot_params[i][-2])
            ax.axis('off')
            fig.add_subplot(ax)
            axis = (fig.add_subplot(inner[0,0]), fig.add_subplot(inner[0,1], projection='polar'))
            plot_type(axis, i, *plot_params[i])


    # Total feasible set
    ax = fig.add_subplot(outer[4:,:], projection='polar')
    ax.set_title("Total feasible set")
    for psi in psi_feasible_set.c_ranges:
        ax.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label='feasible')
    ax.set_title(r"$\psi$",y=-0.11)
    ax.set_xticklabels([r'$0^\circ$', '', '', '', r'$180^\circ$', '', '', ''])
    ax.set_yticklabels([])

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
    plt.show(block=False)


if __name__ == "__main__":
    from tqdm import tqdm

    pi2 = np.pi/2
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
                # solu, feasible_set, root_funcs = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)  # , p06=my_fk_dh.get_i_T_j(0,6,home_new.flatten())[:3, 3]
                # solu, feasible_set = IK_heuristic1(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, verbose=True)
            GC2, GC4, GC6, _, _ =  IK_heuristic2(p07_d=p07, R07_d=R07, DH_model=my_fk_dh)
            solu, feasible_set, root_funcs = IK_anallytical(p07, R07, my_fk_dh, GC2, GC4, GC6, verbose=True)

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
