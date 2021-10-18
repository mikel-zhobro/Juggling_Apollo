from math import sin, cos, tan, atan2, atan, sqrt, acos
import matplotlib.pyplot as plt
import numpy as np
from utilities import ContinuousSet, clip_c
from random import random


eps_psi = 1e-4
delta_psi = 5.0/180*np.pi
# print('Delta psi:' + str(eps_psi))

def cosine_type(a, b, c, joint, GC=1.0, verbose=False, sine_type=False):
    feasible_set = ContinuousSet(-np.pi, np.pi, False, True)  # here we capture the feasible set of psi
    stat_psi = list()
    psi_singular = False

    theta_f = lambda psi: GC * acos( clip_c(a*sin(psi) + b*cos(psi) + c))
    grad_theta_f = lambda psi: -GC * (a*cos(psi) - b*sin(psi))/ sin(theta_f(psi)) if theta_f(psi) != 0.0 else 1.0

    at_2 = a**2
    bt_2 = b**2
    ct_2 = c**2

    ss = at_2 + bt_2

    if ss > eps_psi:  # cyclic profile
        # The idea is to split the cyclic profile in sub-intervals of monotonic profile
        # if verbose:
        #     print('2 Stationary points(cyclic): ss>0')
        psi_min = 2*atan( (-b - sqrt(ss)) / (a) )
        psi_max = 2*atan( (-b + sqrt(ss)) / (a) )
        stat_psi = [psi_min, psi_max]

    else:  # discontinuous profile (2 possibilities)
        # if verbose:
        #     print('Singularity: ss={}'.format(ss))
        psi_singular = 2 * atan(-b/a) # should not be avoided


    # Joint Limits
    if sine_type:
        theta_min = joint.limit_range.a + np.pi/2 if GC > 0.0 else np.pi/2-joint.limit_range.b
        theta_max = joint.limit_range.b + np.pi/2 if GC > 0.0 else np.pi/2-joint.limit_range.a
    else:
        theta_min = joint.limit_range.a
        theta_max = joint.limit_range.b

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
            # if verbose:
            #     print('NO ROOTS FOR THETA_LIMIT = {}'.format(theta))
            pass
        else:
            psi0 = 2*atan( (a + sqrt(tt)) / (cos(theta) + b - c) )
            psi1 = 2*atan( (a - sqrt(tt)) / (cos(theta) + b - c) )
            roots = [Root(psi, lower_lim) for psi in [psi0, psi1] if abs(theta_f(psi)-theta) < 1e-6]
        return roots

    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)
    # if verbose:
    #     print('roots', roots)

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
        plt.suptitle(r'$\theta_{}$ -- '.format(joint.index) + ("cosine_type" if not sine_type else "sine_type") + r'$\ GC_{}$={}'.format(joint.index, GC))
        plt.show()
    return feasible_set



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

    # if verbose:
    #     print('an, bn, cn, ad, bd, cd')
    #     print(an, bn, cn, ad, bd, cd)
    #     print('at, bt, ct')
    #     print(at, bt, ct)

    ss = at_2 + bt_2 - ct_2

    if ss > eps_psi:  # cyclic profile
        # The idea is to split the cyclic profile in sub-intervals of monotonic profile
        psi_min = 2*atan( (at - sqrt(ss)) / (bt-ct) )
        psi_max = 2*atan( (at + sqrt(ss)) / (bt-ct) )
        stat_psi = [psi_min, psi_max]
        # if verbose:
        #     print('Stationary points(cyclic): ss>0')

    elif ss < -eps_psi:  # monotonic profile
        # if verbose:
        #     print('No Stationary points(monotonic): ss<0')
        pass
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

        # if verbose:
        #     print('Singularity: ss={}'.format(ss))
        #     print(singular_feasible_set)
        #     print(feasible_set)
        #     print('theta_neg:'+ str(theta_s_neg))
        #     print('theta_pos:'+ str(theta_s_neg + (np.pi if theta_s_neg < 0.0 else  -np.pi)))

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

    # Consider Limitss
    theta_min = joint.limit_range.a
    theta_max = joint.limit_range.b
    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)
    # if verbose:
    #     print('roots', roots)

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
        if  theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max and theta_min <= theta_f(2*random()*np.pi - np.pi) <= theta_max:
            limits_feasible_set.add_c_range(-np.pi, np.pi)

    feasible_set -= limits_feasible_set

    psi_s =  np.linspace(-np.pi, np.pi, np.pi/0.01)
    thetas =   [theta_f(psi) for psi in psi_s]
    grad_thetas =   [grad_theta_f(psi) for psi in psi_s]

    # Ploting
    if verbose:
        # Theta(psi)
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
        plt.xlabel(r'$\psi$')
        plt.ylabel(r'$\theta_i$')
        plt.legend(loc=1)

        # Psi in polar coordinates
        plt.subplot(122, polar=True)
        for psi in feasible_set.c_ranges:
            plt.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label='feasible')
        if not singular_feasible_set.empty:
            for psi in singular_feasible_set.inverse(-np.pi, np.pi):
                plt.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='red', alpha=0.3, label='singularity')
        plt.legend(loc=1)
        plt.suptitle(r'$\theta_{}$ -- tangent_type'.format(joint.index))
        plt.show()

    return feasible_set


def sine_type(a, b, c, joint, GC=1.0, verbose=False):
    # a = sin(x) = cos(x-pi/2)
    # GC = 1: sin(x) = cos(x-pi/2)
        # x_min = theta_lim_range.a + pi/2
        # x_max = theta_lim_range.b + pi/2
    # GC = -1: a = sin(pi-x) = cos(pi/2-x)
        # x_min = pi/2 - theta_lim_range.b
        # x_max = pi/2 - theta_lim_range.a
    return cosine_type(a, b, c, joint, GC=GC, verbose=verbose, sine_type=True)


cosine = False
if __name__ == "__main__":
    # while cosine:
    #     params = np.random.rand(3)
    #     cosine_type(*params, theta_lim_range=ContinuousSet(-2.0, 2.0), verbose=True)
    # while  not cosine:
    #     params = np.random.rand(6)*2.0 - 1.0
    #     tangent_type(*params, theta_lim_range=ContinuousSet(-1.2, 2.2), verbose=True)
    a = ContinuousSet(-2.0, 2.0)
    print(a)
    a.a=a.a+1
    print(a)