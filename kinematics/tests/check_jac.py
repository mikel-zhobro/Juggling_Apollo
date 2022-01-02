
from pickle import FALSE
import path
import numpy as np
from fk import J, FK, FK_DH
from scipy.optimize import approx_fprime
from DH import DH_revolut

np.set_printoptions(precision=3, suppress=True)

eps = np.sqrt(np.finfo(float).eps)

J_  = lambda q: J(*q)

from kinematics.utilities import R_joints, JOINTS_LIMITS, JOINTS_V_LIMITS
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
    my_fk_dh.add_joint(a, alpha, d, theta, JOINTS_LIMITS[name], JOINTS_V_LIMITS[name], name, offset)


# rbase_frame = False
fk_dh = lambda rbase_frame: lambda *q: my_fk_dh.FK(np.array(q), rbase_frame)
J_dh = lambda rbase_frame: lambda q: my_fk_dh.J(q, rbase_frame)
H_dh = lambda rbase_frame: lambda q: my_fk_dh.H(q, rbase_frame)


def numerical_J_pos(q, fk):
    def pos(q, i):
        T = fk(*q)
        p = T[:-1, -1]
        return p[i]

    jac = np.array([approx_fprime(q, pos, eps, i) for i in range(3)]).reshape(3,7)
    return jac

def numerical_H_pos(q, J):
    def pos(q, i, j):
        Jac = J(q)
        return Jac[i,j]

    jac = np.array([ [approx_fprime(q, pos, eps, i, j) for j in range(7)] for i in range(6)])
    return jac


def numerical_J_orientation(q, fk): #TODO
    def orientation(q, i):
        T = fk(*q)
        R = T[:-1, :-1]
        # return p[i]

    jac = np.array([approx_fprime(q, orientation, eps, i) for i in range(3)]).reshape(3,7)
    return jac


def check_J_position(fk, jac):
    for _ in range(8):
        x = np.random.uniform(0, np.pi, 7) # draw joint values between 0 and pi/2

        J_num = numerical_J_pos(x, fk)
        J_an = jac(x)[:3,:]

        print(np.linalg.norm(J_an-J_num))


def check_FK_FKDH():
    for _ in range(8):
        x = np.random.uniform(0, np.pi, 7) # draw joint values between 0 and pi/2

        T = FK(*x.copy())
        T_DH = FK_DH(x)

        print(np.linalg.norm(T-T_DH))


def check_Hessian(hess, jacc):
    for _ in range(8):
        x = np.random.uniform(0, np.pi, 7) # draw joint values between 0 and pi/2
        H = hess(x)
        H_num = numerical_H_pos(x, jacc)
        print(np.linalg.norm(H-H_num))



print("1. DH Jacobian Test")
print("\n-- a) Not rBase --\n")  # REFERENCE IS SIM-APOLLO'S BASE
check_J_position(fk_dh(False), J_dh(False))
print("\n-- b) rBase --\n")  # REFERENCE IS RIGHT-HAND BASE

check_J_position(fk_dh(True), J_dh(True))
print("\n2. Analytical Jacobian Test\n")
check_J_position(FK, J_)
print("\n-- Position fk-fk_dh --\n")
check_FK_FKDH()

print("\n3. Hessian Test\n")
print("\n-- a) Not rBase --\n")  # REFERENCE IS SIM-APOLLO'S BASE
check_Hessian(H_dh(True), J_dh(True))
print("\n-- b) rBase --\n")  # REFERENCE IS RIGHT-HAND BASE
check_Hessian(H_dh(False), J_dh(False))