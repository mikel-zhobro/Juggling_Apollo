from math import sin, cos, tan, atan2, atan, sqrt, acos
import matplotlib.pyplot as plt
import numpy as np
from Sets import ContinuousSet
from utilities import clip_c
from random import random



eps_psi = 1e-4
delta_psi = 5.0/180*np.pi
# print('Delta psi:' + str(eps_psi))

def cosine_type(a, b, c, joint, GC=1.0, verbose=False):
    feasible_set = ContinuousSet(-np.pi, np.pi, False, True)  # here we capture the feasible set of psi
    stat_psi = list()
    psi_singular = False

    theta_f = lambda psi: GC * acos( clip_c(a*sin(psi) + b*cos(psi) + c))
    grad_theta_f = lambda psi: -GC * (a*cos(psi) - b*sin(psi))/ sin(theta_f(psi)) if theta_f(psi) != 0.0 else 1.0

    at_2 = a**2
    bt_2 = b**2
    ct_2 = c**2

    ss = at_2 + bt_2

    if verbose:
        print('a, b, c')
        print(a, b, c)
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


    # Joint Limits
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
                # return str(self.root) + '~' + ('l' if self.lower_lim else 'u')
                return '{}: root({}), grad({}) {}'.format('lower_lim' if self.lower_lim else 'upper_lim', self.root, self.grad, 'entering' if self.enters else 'leaving')

        tt = ss - (c - cos(theta))**2
        roots = list()
        if tt < 0.0:
            if verbose:
                print('NO ROOTS FOR THETA_LIMIT = {}'.format(theta))
        else:
            psi0 = 2*atan( (a + sqrt(tt)) / (cos(theta) + b - c) )
            psi1 = 2*atan( (a - sqrt(tt)) / (cos(theta) + b - c) )
            roots = [Root(psi, lower_lim) for psi in [psi0, psi1] if abs(theta_f(psi)-theta) < 1e-6]
        return roots

    # Consider Limitss
    theta_min = joint.limit_range.a
    theta_max = joint.limit_range.b
    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)
    if verbose:
        print('roots', roots)

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

    if verbose:
        psi_s =  np.linspace(-np.pi, np.pi, np.pi/0.01)
        thetas =   [theta_f(psi) for psi in psi_s]
        grad_thetas =   [grad_theta_f(psi) for psi in psi_s]
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.axhline(theta_min, color='k')
        plt.axhline(theta_max, color='k')
        plt.plot(psi_s, thetas, label=r'$\theta_i = f(\psi)$')
        plt.plot(psi_s, grad_thetas/np.max(np.abs(grad_thetas)), label=r'$\frac{d\theta_i}{d\psi} = g(\psi)$')
        # Stationary points
        for v in stat_psi:
            plt.scatter(v, theta_f(v), c='b')
        # Roots for theta_min
        for r in roots:
            plt.scatter(r.root, theta_f(r.root), c='r')
        # Singular poinrs
        for psi in feasible_set.c_ranges:
            plt.axvspan(psi.a, psi.b, color='green', alpha=0.3)
        if psi_singular:
            plt.scatter(psi_singular, theta_f(psi_singular), c='k')

        plt.xlabel(r'$\psi$')
        plt.ylabel(r'$\theta_i$')
        plt.legend(loc=1)

        # Add a bar in the polar coordinates
        plt.subplot(122, polar=True)
        for psi in feasible_set.c_ranges:
            plt.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label='feasible')
        plt.legend(loc=1)
        plt.suptitle(r'$\theta_i$'.format(joint.index))
        plt.show()
    return feasible_set


if __name__ == "__main__":
    while True:
        params = np.random.rand(3)
        cosine_type(*params, theta_lim_range=ContinuousSet(-2.0, 2.0), verbose=True)
