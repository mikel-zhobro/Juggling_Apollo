import numpy as np


import __add_path__
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics
from ApolloKinematics import utilities

np.set_printoptions(precision=4, suppress=True)

# PARAMS
print("juggling_apollo")

# Home Configuration
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.41],
                   [-1.0, 0.0, 0.0, -0.69],
                   [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
T_home = np.array([[0.0, -1.0, 0.0,  0.47],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.52],
                   [-1.0, 0.0, 0.0, -0.7],
                   [0.0,  0.0, 0.0,  1.0 ]], dtype='float')

# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=NOISE)  ## kinematics with noise (used for its (wrong)IK calculations)
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)               ## kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)

# C) PLANNINGs
if False:
    for z in np.arange(-0.4, -0.2, 0.05):
        for y in np.arange(0.8, 1., 0.05):
            for x in np.arange(0.3, 0.5, 0.05):

                T_home = np.array([[0.0, -1.0, 0.0,  x],  # uppword orientation(cup is up)
                                [0.0,  0.0, 1.0,     y],
                                [-1.0, 0.0, 0.0,     z],
                                [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
                raw_input("Press Enter to continue...")
                try:
                    qh = rArmKinematics.IK(T_home)
                    rArmInterface.go_to_home_position(qh, 100)
                    print(x,y,z)
                except:
                    print("Not vaild home")
                    print(x,y,z)


# C) Rotate around shoulder-wrist axis
if True:
    T_home = np.array([[0.0, -1.0, 0.0,  0.3],  # uppword orientation(cup is up)
                    [0.0,  0.0, 1.0,  0.7],
                    [-1.0, 0.0, 0.0, -0.5],
                    [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
    _, _, _, _, _, solu, feasible_set = rArmKinematics.IK(T_home, for_seqik=True)
    startt = feasible_set.b; endd = feasible_set.a
    rArmInterface.go_to_home_position(solu(feasible_set.b), 2000, eps=3., )
    # rArmInterface.go_to_home_position(solu(feasible_set.a), 2000)
    # rArmInterface.go_to_home_position(solu(feasible_set.b), 2000)

    raw_input("Press Enter to continue...")
    for i in range(8):
        for psi in np.linspace(startt, endd, 100):
            qh = solu(psi)
            rArmInterface.go_to_home_position(qh, it_time=30, eps=3., wait=1, zero_speed= (psi==endd), verbose=False, reset_PID=False)

        tmp = endd
        endd =startt
        startt = tmp



if False:
    from scipy.optimize import minimize
    def findBestThrowPosition(FK, J, q_init, qdot_init, R_des, jac=None):
        con = lambda i: lambda qqd: J(qqd[:7])[i, :].dot(qqd[7:])
        con_R = lambda i: lambda qqd: FK(qqd[:7])[i,i] - R_des[i,i]

        cons = tuple({'type':'eq', 'fun': con(i)} for i in range(2)) + tuple({'type':'eq', 'fun': con_R(i)} for i in range(3))


        bounds =  [utilities.JOINTS_LIMITS[j] for j in utilities.R_joints] + [utilities.JOINTS_V_LIMITS[j] for j in utilities.R_joints]

        def fun(qqd):
            return - J(qqd[:7])[2, :].dot(qqd[7:])

        result = minimize(fun, (q_init, qdot_init) , method="SLSQP", bounds=bounds, constraints=cons)
        qqd = result.x
        if not result.success:
            print("optim was unseccussfull")
            return q_init, qqd[7:]
        else:
            print(result)
        return qqd[:7], qqd[7:]




    for z in np.arange(-0.4, -0.2, 0.05):
        for y in np.arange(0.8, 1., 0.05):
            for x in np.arange(0.3, 0.5, 0.05):

                T_home = np.array([[0.0, -1.0, 0.0,  x],  # uppword orientation(cup is up)
                                [0.0,  0.0, 1.0,     y],
                                [-1.0, 0.0, 0.0,     z],
                                [0.0,  0.0, 0.0,  1.0 ]], dtype='float')
                raw_input("Press Enter to continue...")
                try:
                    q_init = rArmKinematics.IK(T_home)
                    q, qdot = findBestThrowPosition(rArmKinematics.FK, rArmKinematics.J, q_init, np.zeros((7,1)), R_des=T_home[:3,:3])
                    rArmInterface.go_to_home_position(q, 2000)
                    print(x,y,z)
                except:
                    print("Not vaild home")
                    print(x,y,z)


########################################################################################################

    """
    (0.3, 0.8999999999999999, -0.5000000000000001)
    (0.4, 0.8999999999999999, -0.5000000000000001)
    (0.3, 0.8999999999999999, -0.4000000000000001)
    (0.4, 0.8999999999999999, -0.4000000000000001)
    (0.3, 0.9, -0.30000000000000004)
    (0.4, 0.9, -0.30000000000000004)
    (0.3, 1.0, -0.30000000000000004)
    """



    """Throw
    1.91  ,  1.91  , -2.23  , -2.23  ,  3.56  ,  3.21  ,  0.
    """