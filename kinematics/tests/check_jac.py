
import path
import numpy as np
from fk import J, FK, FK_DH
from scipy.optimize import approx_fprime
np.set_printoptions(precision=3, suppress=True)

eps = np.sqrt(np.finfo(float).eps)

def numerical_J_pos(q):
    def pos(q, i):
        T = FK(*q)
        p = T[:-1, -1]
        return p[i]

    jac = np.array([approx_fprime(q, pos, eps, i) for i in range(3)]).reshape(3,7)
    return jac


def numerical_J_orientation(q): #TODO
    def orientation(q, i):
        T = FK(*q)
        R = T[:-1, :-1]
        # return p[i]

    jac = np.array([approx_fprime(q, orientation, eps, i) for i in range(3)]).reshape(3,7)
    return jac


def check_J_position():
    for _ in range(8):
        x = np.random.uniform(0, np.pi, 7) # draw joint values between 0 and pi/2

        J_num = numerical_J_pos(x)
        J_an = J(*x)[:3,:]

        print(np.linalg.norm(J_an-J_num))


def check_FK_FKDH():
    for _ in range(8):
        x = np.random.uniform(0, np.pi, 7) # draw joint values between 0 and pi/2

        T = FK(*x.copy())
        T_DH = FK_DH(x)

        print(np.linalg.norm(T-T_DH))

check_J_position()
check_FK_FKDH()