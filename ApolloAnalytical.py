import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sin, cos, acos, sqrt, atan2, asin

from kinematics.IK_analytical import IK_anallytical
from kinematics.DH import DH_revolut
from kinematics.utilities import R_joints, L_joints
from kinematics.fk import FK_DH
from kinematics.utilities import skew, vec, clip_c
from kinematics.Feasibility import sine_type, cosine_type, tangent_type
from kinematics.Sets import ContinuousSet


th3_offset = np.pi/6
np.set_printoptions(precision=4, suppress=True)


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
    p07_d, R07_d = DH_model.get_goal_in_base_frame(p07_d, R07_d)
    l0bs = vec([0,   0,     d_bs])
    l3se = vec([0,  -d_se,  0])
    l4ew = vec([0,   0,     d_ew])
    l7wt = vec([0,   0,     d_wt])

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
    feasible_sets[0] = tangent_type(*S1, joint=DH_model.joint(0), verbose=verbose)
    feasible_sets[1] =    sine_type(*S2, joint=DH_model.joint(1), GC=GC2, verbose=verbose)
    feasible_sets[2] = tangent_type(*S3, joint=DH_model.joint(2), verbose=verbose)
    feasible_sets[3] = ContinuousSet(-np.pi, np.pi, False, True) if th4 in DH_model.joint(3).limit_range else ContinuousSet()
    feasible_sets[4] = tangent_type(*W5, joint=DH_model.joint(4), verbose=verbose)
    feasible_sets[5] =  cosine_type(*W6, joint=DH_model.joint(5), GC=GC6, verbose=verbose)
    feasible_sets[6] = tangent_type(*W7, joint=DH_model.joint(6), verbose=verbose)
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
        th2 = lambda psi: ( np.pi - asin(clip_c( v(psi).dot(S2))) ) if asin(clip_c( v(psi).dot(S2))) > 0.0 else  ( -np.pi - asin(clip_c( v(psi).dot(S2))) )

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
        print(psi_feasible_set)
        plt.figure()
        plt.subplot(111, polar=True)
        for psi in psi_feasible_set.c_ranges:
            plt.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label='feasible')
        plt.title('Final Feasible Set [GC2({}), GC4({}), GC6({}])'.format(GC2, GC4, GC6))
        plt.show()
    # print(psi_feasible_set)
    solution_function = lambda psi: np.array([ th1(psi), th2(psi), th3(psi), th4, th5(psi), th6(psi), th7(psi)]).reshape(-1,1)

    return (solution_function, psi_feasible_set)


def IK_heuristic1(p07_d, R07_d, DH_model, verbose=False):
    GC2_plus = ContinuousSet(-np.pi/2 , np.pi)
    GC2_minus = ContinuousSet(-np.pi , -np.pi/2) + ContinuousSet(np.pi/2 , np.pi)

    GC46_plus = ContinuousSet(0.0, np.pi)
    GC46_minus = ContinuousSet(0.0, -np.pi)

    GC2 = 1.0 if (GC2_plus - DH_model.joint(1).limit_range).size >= (GC2_minus - DH_model.joint(1).limit_range).size else -1.0
    GC4 = 1.0 if (GC46_plus - DH_model.joint(3).limit_range).size >= (GC46_minus - DH_model.joint(3).limit_range).size else 1.0
    GC6 = 1.0 if (GC46_plus - DH_model.joint(5).limit_range).size >= (GC46_minus - DH_model.joint(5).limit_range).size else -1.0

    return IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=verbose, p06=None, p07=None)


def IK_heuristic2(p07_d, R07_d, DH_model, verbose=False):
    biggest_feasible_set = ContinuousSet()
    GC2_final = 1.0
    GC4_final = 1.0
    GC6_final = 1.0
    GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
    for GC2, GC4, GC6 in GCs:
        _, psi_feasible_set = IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False, p06=None, p07=None)
        if psi_feasible_set.size > biggest_feasible_set:
            biggest_feasible_set = psi_feasible_set
            GC2_final = GC2
            GC4_final = GC4
            GC6_final = GC6
    return IK_anallytical(p07_d, R07_d, DH_model, GC2=GC2_final, GC4=GC4_final, GC6=GC6_final, verbose=verbose, p06=None, p07=None)


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
    my_fk_dh.add_joint(a, alpha, d, theta, name, offset)

home_pose = np.array([0.0, -0.0, -np.pi/6, np.pi/2, 0.0, -0.0, 0.0]).reshape(-1,1)
home_pose = np.array([1.0, 1.0, np.pi/6, 1.0, np.pi/4, 1.0, 2.0]).reshape(-1,1)

print(my_fk_dh.FK(home_pose))

# Test with random goal poses
if False:
    GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
    for i in tqdm(range(10000)):
        # home_new = np.random.rand(7,1)*np.pi
        # home_new = 2*(np.random.rand(7,1)-0.5)*np.pi

        home_new = np.array([ j.limit_range.sample() for j in my_fk_dh.joints ]).reshape(7,1)

        # print(home_new.T)
        T07_home = my_fk_dh.FK(home_new)
        R07 = T07_home[:3, :3]
        p07 = T07_home[:3, 3:4]
        # print(p07.T)
        for GC2, GC4, GC6 in GCs:
        #     print(GC2, GC4, GC6)
        #     IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, GC2=1, GC4=GC4, GC6=GC6, verbose=False, p06=my_fk_dh.get_i_T_j(0,6,home_new.flatten())[:3, 3], p07=my_fk_dh.get_i_T_j(0,7,home_new.flatten())[:3, 3])
        # continue
            solu, feasible_set = IK_anallytical(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False, p06=my_fk_dh.get_i_T_j(0,6,home_new.flatten())[:3, 3])
        # solu, feasible_set = IK_heuristic1(p07_d=p07, R07_d=R07, DH_model=my_fk_dh, verbose=False)

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



if False:
    for i in range(122):
        home_new = np.random.rand(7,1)*np.pi/2
        T_1 = FK_DH(home_new.copy())
        T_2 = my_fk_dh.FK(home_new.copy())
        nrr = np.linalg.norm(T_1-T_2)
        # print(T_1)
        # print(T_2)
        if nrr >1e-6:
            print("ERROR")
