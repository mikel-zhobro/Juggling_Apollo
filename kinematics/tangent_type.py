from math import sin, cos, tan, atan2, atan, sqrt
import matplotlib.pyplot as plt
import numpy as np
from utilities import ContinuousSet
from random import random



eps_psi = 1e-4
delta_psi = 15.0/180*np.pi
print('Delta psi:' + str(eps_psi))
def tangent_type(an, bn, cn, ad, bd, cd):
    feasible_set = ContinuousSet(-np.pi, np.pi, False, True)

    theta_f = lambda psi: atan2(an*sin(psi) + bn*cos(psi) + cn,
                              ad*sin(psi) + bd*cos(psi) + cd)

    tan_theta = lambda psi: (an*sin(psi) + bn*cos(psi) + cn) / (ad*sin(psi) + bd*cos(psi) + cd)

    fn = lambda psi: an*sin(psi) + bn*cos(psi) + cn
    fd = lambda psi: ad*sin(psi) + bd*cos(psi) + cd
    ft = lambda psi: at*sin(psi) + bt*cos(psi) + ct
    grad_theta_f = lambda psi: (ft(psi)) / (fn(psi)**2 + fd(psi)**2)

    at = bd*cn - bn*cd; at_2 = at**2
    bt = an*cd - ad*cn; bt_2 = bt**2
    ct = an*bd - ad*bn; ct_2 = ct**2

    ss = at_2 + bt_2 - ct_2
    # print('ss' + str(ss))
    stat_psi = list()

    # if abs(ss) > eps_psi+1e-7:
    #     return
    print('an, bn, cn, ad, bd, cd')
    print(an, bn, cn, ad, bd, cd)
    print('at, bt, ct')
    print(at, bt, ct)
    if ss > eps_psi:  # cyclic profile
        # The idea is to split the cyclic profile in sub-intervals of monotonic profile
        print('Stationary points(cyclic): ss>0')
        psi_min = 2*atan( (at - sqrt(ss)) / (bt-ct) )
        psi_max = 2*atan( (at + sqrt(ss)) / (bt-ct) )
        stat_psi = [psi_min, psi_max]

    elif ss < -eps_psi:  # monotonic profile
        print('No Stationary points(monotonic): ss<0')

        # feas_psi = feasible_set_for_monotonic_function(tan_f, j.limit_range, ContinuousSet(-np.pi, np.pi, False))

    else:  # discontinuous profile (2 possibilities)
        print('Singularity: ss={}'.format(ss))
        singular_feasible_set = ContinuousSet()
        theta_s_neg = atan((at*bn - bt*an) / (at*bd - bt*ad))
        psi_singular = 2 * atan(at/ (bt-ct)) # should be avoided
        if psi_singular + delta_psi > np.pi:
            singular_feasible_set.add_c_range(psi_singular+delta_psi-2*np.pi - delta_psi, psi_singular - delta_psi)
        elif psi_singular - delta_psi < -np.pi:
            singular_feasible_set.add_c_range(psi_singular + delta_psi, 2*np.pi+(psi_singular-delta_psi))
        else:
            singular_feasible_set.add_c_range(-np.pi, psi_singular-delta_psi)
            singular_feasible_set.add_c_range(psi_singular+delta_psi, np.pi)


        print(singular_feasible_set)
        feasible_set -= singular_feasible_set
        print(feasible_set)

        # singular_theta = [theta_f(psi) for psi in singular_psi]
        print('theta_neg:'+ str(theta_s_neg))
        print('theta_pos:'+ str(theta_s_neg + (np.pi if theta_s_neg < 0.0 else  -np.pi)))
        # singular_theta.append(theta_s_neg)
        # singular_theta.append(theta_s_neg + (np.pi if theta_s_neg < 0.0 else  -np.pi))
        # singular_theta.append(0.0)



    # Joint Limits
    # theta_max = -2.2

    # Lower Lim
    theta_min = -1.2
    ap = (cd-bd)*tan(theta_min) + (bn-cn)
    bp = 2*(ad*tan(theta_min) - an)
    cp = (bd+cd)*tan(theta_min) - (bn+cn)

    tt = bp**2 - 4*ap*cp
    psi0s = list()
    lower_limit_feasible_set = ContinuousSet()
    if tt < 0.0:
        print('NO ROOTS FOR THETA_MIN = {}'.format(theta_min))
    else:
        psi0 = 2*atan( (-bp + sqrt(tt)) / (2*ap) )
        grad_psi0 = grad_theta_f(psi0)
        psi1 = 2*atan( (-bp - sqrt(tt)) / (2*ap) )
        grad_psi1 = grad_theta_f(psi1)
        psi0s =  sorted([psi for psi in [psi0, psi1] if abs(theta_f(psi)-theta_min) < 1e-6])

        if len(psi0s) == 2:
            psi0 = psi0s[0]
            psi1 = psi0s[1]
            if grad_theta_f(psi0) > 0:
                assert grad_theta_f(psi1) <= 0, 'The other gradient should be neg'
                lower_limit_feasible_set.add_c_range(psi0, psi1)  # entering in psi0, leaving in psi1
            else:
                assert grad_theta_f(psi1) >= 0, 'The other gradient should be pos'
                lower_limit_feasible_set.add_c_range(-np.pi, psi0)  # leaving in psi0
                lower_limit_feasible_set.add_c_range(psi1, np.pi)  # re-entering in psi1
        elif len(psi0s)==1:
            psi0 = psi0s[0]
            if grad_theta_f(psi0) > 0:
                lower_limit_feasible_set.add_c_range(psi0, np.pi)
            else:
                lower_limit_feasible_set.add_c_range(-np.pi, psi0)
        else:
            # if nullstellen, either every psi is in feasible set or none of them
            # sample to psi and check if feasible
            if theta_f(2*random()*np.pi - np.pi) >= theta_min and theta_f(2*random()*np.pi - np.pi) >= theta_min:
                lower_limit_feasible_set.add_c_range(-np.pi, np.pi)
            else:
                lower_limit_feasible_set.add_c_range(-66, -66)

        feasible_set -= lower_limit_feasible_set



    psi_s =  np.linspace(-np.pi, np.pi, np.pi/0.01)
    thetas =   [theta_f(psi) for psi in psi_s]
    grad_thetas =   [grad_theta_f(psi) for psi in psi_s]
    plt.plot(psi_s, thetas)
    plt.plot(psi_s, grad_thetas/np.max(np.abs(grad_thetas)), label='grad')
    # Stationary points
    for v in stat_psi:
        plt.scatter(v, theta_f(v), c='b')
    # Roots for theta_min
    for psi0 in psi0s:
        plt.scatter(psi0, theta_f(psi0), c='r')
    # Singular poinrs
    for psi in feasible_set.c_ranges:
        plt.axvspan(psi.a, psi.b, color='green', alpha=0.3)
    plt.axhline(theta_min, color='k')
    plt.xlabel(r'$\psi$')
    plt.ylabel(r'$\theta_i$')
    plt.legend()
    plt.show()
    return ss



ans = []
bns = []
cns = []

ads = []
bds = []
cds = []

while True:
    params = np.random.rand(6)*2.0 - 1.0
    # params = (0.7842348, 0.66533477, 0.48922079, 0.04910175, 0.04834799, 0.02821791)
    tangent_type(*params)
