import numpy as np
from fk import J, FK
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
    
x = np.ones(7)


def check_J_position():
    for _ in range(8):
        x = np.random.uniform(0,np.pi/2, 7)
        
        J_num = numerical_J_pos(x)
        J_an = J(*x)[:3,:]
        
        print(np.linalg.norm(J_an-J_num))






check_J_position()