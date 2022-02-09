import path
import numpy as np
from fk_pin_local import PinRobot
from DH import DH_revolut
from AnalyticalIK import IK_anallytical, IK_find_psi_and_GCs
from utilities import R_joints, L_joints, JOINTS_LIMITS, pR2T, JOINTS_V_LIMITS
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)


r_arm = True # TODO: left arm FK not working

joints2Use = R_joints if r_arm else L_joints


pi2 = np.pi/2
th3_offset = np.pi/6
d_bs = 0.378724; d_se = 0.4; d_ew = 0.39; d_wt = 0.186
a_s            = [0.0] * 7
alpha_s        = [pi2, pi2, -pi2, pi2, -pi2, pi2, 0.0]
d_s            = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
theta_s = [0.0, -pi2, pi2, 0.0, -pi2, 0.0, 0.0]
offsets = [0.0, 0.0, th3_offset, 0.0, 0.0, 0.0, 0.0]

# Create Robots
dh_rob = DH_revolut()
for a, alpha, d, theta, name, offset in zip(a_s, alpha_s, d_s, theta_s, joints2Use, offsets):
    dh_rob.add_joint(a, alpha, d, theta, JOINTS_LIMITS[name], JOINTS_V_LIMITS[name], name, offset)

try:
    pin_rob = dh_rob # PinRobot(r_arm=r_arm)
except:
    pin_rob = dh_rob



def check_FK():
    # Compare FK functions of pinocchio model and DH model
    for _ in tqdm(range(10000)):
        home_pose = np.array([ j.limit_range.sample() for j in dh_rob.joints ]).reshape(7,1)
        T_DH = dh_rob.FK(home_pose)
        T_pin = pin_rob.FK(home_pose)
        nrr = np.linalg.norm(T_DH - T_pin)
        if nrr >4e-4:
            print('ERR', nrr)
            print(home_pose.T)
            print(T_DH)
            print(T_pin)
            assert False

def check_IK():
    # Create poses with pinocchio FK and calculate the input with dh_IK
    GCs = [(i, ii, iii) for i in [-1.0, 1.0] for ii in [-1.0, 1.0] for iii in [-1.0, 1.0]]
    for _ in tqdm(range(1000)):
        home_pose = np.array([ j.limit_range.sample() for j in dh_rob.joints ]).reshape(7,1)
        T_pin = pin_rob.FK(home_pose)
        # T_pin = np.array([[0.0, -1.0, 0.0,  0.32],  # out of reachable set position
        #                 [0.0,  0.0, 1.0,  0.71],
        #                 [-1.0, 0.0, 0.0, -0.89],
        #                 [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
        for GC2, GC4, GC6 in GCs:
            solu, _, _ = IK_anallytical(p07_d=T_pin[:3,3:4], R07_d=T_pin[:3,:3], DH_model=dh_rob, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)
            for f in np.arange(-1.0, 1.0, 0.2):
                s = solu(f*np.pi)
                nrr = np.linalg.norm(dh_rob.FK(s) - T_pin)
                if nrr >4e-4:
                    print(dh_rob.FK(s) - T_pin)
                    print('ERR', nrr)
                    print("GCs", GC2, GC4, GC6)
                    print(home_pose.T)
                    assert False

def check_IK_feasible_set():
    pass


def check_get2Base():
    # Test that we properly transform any world_T_tcp to basedh_T_tcp.
    # This is performed every time before starting to calculate IK
    for _ in tqdm(range(10000)):
        home_pose = np.array([ j.limit_range.sample() for j in dh_rob.joints ]).reshape(7,1)
        base_T_tcp = dh_rob.FK(home_pose)
        basedh_T_tcp1 = pR2T(*dh_rob.get_goal_in_dh_base_frame(base_T_tcp[:3, 3:4], base_T_tcp[:3,:3]))
        basedh_T_tcp2 = dh_rob.get_i_T_j(0, 7, home_pose)
        nrr = np.linalg.norm(basedh_T_tcp1 - basedh_T_tcp2)
        if nrr >1e-8:
            print('ERR', nrr)
            print(home_pose.T)
            assert False

def check_IKWithQinit():
    # Create poses with pinocchio FK and calculate the input with dh_IK
    for i in tqdm(range(100000)):
        home_pose = np.array([ j.limit_range.sample() for j in dh_rob.joints ]).reshape(7,1)
        T_pin = pin_rob.FK(home_pose)
        psi, GC2, GC4, GC6, feasible_set, sf = IK_find_psi_and_GCs(p07_d=T_pin[:3,3:4], R07_d=T_pin[:3,:3], q_init=home_pose, DH_model=dh_rob)

        # solu, _, _ = IK_anallytical(p07_d=T_pin[:3,3:4], R07_d=T_pin[:3,:3], DH_model=dh_rob, GC2=GC2, GC4=GC4, GC6=GC6, verbose=False)

        q = sf(psi)
        T = pin_rob.FK(q)

        nrr = np.linalg.norm(home_pose - q)
        if nrr >4e-3:
            print(T_pin-T)
            print('i', i)
            print(home_pose.T)
            print(q.T)
            print('ERR', nrr)
            print("GCs", GC2, GC4, GC6)
            assert False

if __name__ == "__main__":
    # check_FK()
    # check_get2Base()
    # check_IK()
    check_IKWithQinit()