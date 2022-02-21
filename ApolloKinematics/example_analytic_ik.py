import numpy as np

from AnalyticalIK import IK_anallytical, IK_heuristic3, th3_offset
from DHFK import DH_revolut
import utilities


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
    for a, alpha, d, theta, name, offset in zip(a_s, alpha_s, d_s, theta_s, utilities.R_joints, offsets):
        my_fk_dh.add_joint(a, alpha, d, theta, utilities.JOINTS_LIMITS[name], utilities.JOINTS_V_LIMITS[name], name, offset)


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
            GC2, GC4, GC6, _, _ =  IK_heuristic3(p07_d=p07, R07_d=R07, DH_model=my_fk_dh)
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
