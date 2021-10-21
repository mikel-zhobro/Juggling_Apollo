from math import sin, cos, tan, atan2, atan, sqrt, acos, asin
import matplotlib.pyplot as plt
import numpy as np
from utilities import clip_c
from Sets import ContinuousSet
from random import random
from DH import DH_revolut

eps_psi = 1e-4
delta_psi = 5.0/180*np.pi

def plot_type(theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, singular_feasible_set=ContinuousSet()):
    psi_s =  np.linspace(-np.pi, np.pi, np.pi/0.01)
    thetas =   [theta_f(psi) for psi in psi_s]
    grad_thetas =   [grad_theta_f(psi) for psi in psi_s]
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121)

    # Limits
    ax.axhline(theta_min, color='k')
    ax.axhline(theta_max, color='k')

    # Thetas and Theta_grads
    ax.plot(psi_s, thetas, label=r'$\theta_i = f(\psi)$')
    ax.plot(psi_s, grad_thetas/np.max(np.abs(grad_thetas)), label=r'$\frac{d\theta_i}{d\psi} = g(\psi)$')

    # Singularities
    if not singular_feasible_set.empty:
        for psi in singular_feasible_set.inverse(-np.pi, np.pi):
            plt.axvspan(psi.a, psi.b, color='red', alpha=0.3, label='singularity')

    # Stationary points
    thetas = [theta_f(v) for v in stat_psi]
    stat_points = ax.scatter(stat_psi, thetas, c='b')

    # Roots
    root_psis = [r.root for r in roots]
    thetas = [theta_f(v) for v in root_psis]
    root_points = ax.scatter(root_psis, thetas, c='r')

    # Jumpings
    jumping_psis = [r.root for r in jumpings]
    thetas = [theta_f(v) for v in jumping_psis]
    jumping_points = ax.scatter(jumping_psis, thetas, c='g')

    # Feasible Set
    for psi in feasible_set.c_ranges:
        plt.axvspan(psi.a, psi.b, color='green', alpha=0.3)
    ax.set_xlabel(r'$\psi$')
    ax.set_ylabel(r'$\theta_i$')
    l2 = ax.legend((stat_points, root_points, jumping_points),
                ("Statpoitns", "Limit Roots", "Jumpings"),
                bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=8)
    ax.legend(loc=1)
    ax.add_artist(l2)

    # Psi in polar coordinates
    ax2 = plt.subplot(122, polar=True)

    # Feasible Set -- Green
    for i, psi in enumerate(feasible_set.c_ranges):
        ax2.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='green', alpha=0.3, label="_"*i + 'feasible')

    # Singularities -- Red
    if not singular_feasible_set.empty:
        for i, psi in enumerate(singular_feasible_set.inverse(-np.pi, np.pi)):
            plt.bar(psi.a, height=1, width=psi.b-psi.a, bottom=0, align='edge', color='red', alpha=0.3, label="_"*i + 'singularity')
    ax2.legend(loc=1)

    plt.suptitle(title)
    plt.show(block=False)


def cosine_type(a, b, c, joint, GC=1.0, verbose=False, sine_type=False):
    feasible_set = ContinuousSet(-np.pi, np.pi, False, True)  # here we capture the feasible set of psi
    stat_psi = list()
    psi_singular = False
    if sine_type:
        if GC>0.0:
            theta_f = lambda psi: asin( clip_c(a*sin(psi) + b*cos(psi) + c))
            grad_theta_f = lambda psi: GC * (a*cos(psi) - b*sin(psi))/ cos(theta_f(psi))  # if theta_f(psi) != np.pi/2 else 1.0
        else:
            theta_f = lambda psi: ( np.pi - asin( clip_c(a*sin(psi) + b*cos(psi) + c)) ) if asin( clip_c(a*sin(psi) + b*cos(psi) + c)) > 0.0 else  ( -np.pi - asin( clip_c(a*sin(psi) + b*cos(psi) + c)) )
            grad_theta_f = lambda psi: -GC * (a*cos(psi) - b*sin(psi))/ cos(theta_f(psi))  # if theta_f(psi) != np.pi/2 else 1.0
    else:
        theta_f = lambda psi: GC * acos( clip_c(a*sin(psi) + b*cos(psi) + c))
        grad_theta_f = lambda psi: -GC * (a*cos(psi) - b*sin(psi))/ sin(theta_f(psi))  # if theta_f(psi) != 0.0 else 1.0

    at_2 = a**2
    bt_2 = b**2
    ct_2 = c**2

    ss = at_2 + bt_2

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
                return '{}: root({}), grad({}) {}'.format('lower_lim' if self.lower_lim else 'upper_lim', self.root, self.grad, 'entering' if self.enters else 'leaving')
        if sine_type:
            fun = sin
        else:
            fun = cos
        tt = ss - (c - fun(theta))**2
        roots = list()
        if tt < 0.0:
            if verbose:
                print('NO ROOTS FOR THETA_LIMIT = {}'.format(theta))
        else:
            psi0 = 2*atan( (a + sqrt(tt)) / (fun(theta) + b - c) )
            psi1 = 2*atan( (a - sqrt(tt)) / (fun(theta) + b - c) )
            roots = [Root(psi, lower_lim) for psi in [psi0, psi1] if abs(theta_f(psi)-theta) < 1e-6]
        return roots

    # Find jumpings roots
    jumpings = sorted(find_root(-np.pi+1e-8, True) + find_root(np.pi-1e-8, False), key=lambda r: r.root)

    # Joint Limits roots
    limits = joint.limit_range - ContinuousSet(-np.pi+eps_psi, np.pi-eps_psi, False, True)
    # theta_min = joint.limit_range.a
    # theta_max = joint.limit_range.b
    theta_min = limits.a
    theta_max = limits.b
    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)

    # Joint Limits feasible sets
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

    # Ploting
    if verbose:
        title = r'$\theta_{}$ -- '.format(joint.index) + ("cosine_type" if not sine_type else "sine_type") + r'$\ GC_{}$={}'.format(joint.index, GC)
        plot_type(theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title)

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

    ss = at_2 + bt_2 - ct_2

    if ss > eps_psi:  # cyclic profile
        # The idea is to split the cyclic profile in sub-intervals of monotonic profile
        psi_min = 2*atan( (at - sqrt(ss)) / (bt-ct) )
        psi_max = 2*atan( (at + sqrt(ss)) / (bt-ct) )
        stat_psi = [psi_min, psi_max]
        if verbose:
            print('Stationary points(cyclic): ss({})>0'.format(ss))

    elif ss < -eps_psi:  # monotonic profile
        if verbose:
            print('No Stationary points(monotonic): ss({})<0'.format(ss))
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

        if verbose:
            print('Singularity: ss={}'.format(ss))
            print(singular_feasible_set)
            print(feasible_set)
            print('theta_neg:'+ str(theta_s_neg))
            print('theta_pos:'+ str(theta_s_neg + (np.pi if theta_s_neg < 0.0 else  -np.pi)))


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

    # Jumpings roots
    jumpings = sorted(find_root(-np.pi+1e-8, True) + find_root(np.pi-1e-8, False), key=lambda r: r.root)

    # Joint Limits roots
    limits = joint.limit_range - ContinuousSet(-np.pi+eps_psi, np.pi-eps_psi, False, True)
    # theta_min = joint.limit_range.a
    # theta_max = joint.limit_range.b
    theta_min = limits.a
    theta_max = limits.b
    roots = sorted(find_root(theta_min, True) + find_root(theta_max, False), key=lambda r: r.root)

    # Joint Limits feasible sets
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

    # Ploting
    if verbose:
        title = r'$\theta_{}$ -- tangent_type'.format(joint.index)
        plot_type(theta_f, grad_theta_f, theta_min, theta_max, stat_psi, roots, jumpings, feasible_set, title, singular_feasible_set)

    return feasible_set


def sine_type(a, b, c, joint, GC=1.0, verbose=False):
    # GC = 1: sin(x) = cos(x-pi/2)
        # x_min = theta_lim_range.a + pi/2
        # x_max = theta_lim_range.b + pi/2
    # GC = -1: a = sin(pi-x) = cos(pi/2-x)
        # x_min = pi/2 - theta_lim_range.b
        # x_max = pi/2 - theta_lim_range.a
    return cosine_type(a, b, c, joint, GC=GC, verbose=verbose, sine_type=True)


if __name__ == "__main__":
    class Joint():
        def __init__(self, a, b, theta=0.0):
            self.theta = theta
            self.index = DH_revolut.n_joints
            self.limit_range = ContinuousSet(a, b, False, False)
            DH_revolut.n_joints += 1

    def get_singular_tangent_type_params():
        while True:
            paramsn = np.random.rand(3,1)*2.0 - 1.0
            paramsd = np.random.rand(3,1)*2.0 - 1.0
            z = np.cross(paramsn, paramsd, axis=0)
            if np.linalg.norm(z[0,0]**2 + z[1,0]**2 - z[2,0]**2) <1e-5:
                break
        params = np.zeros(6, dtype=float)
        params[:3] = paramsn[:,0]
        params[3:] = paramsd[:,0]
        return params

    cosine = False
    while cosine:
        params = np.random.rand(3)
        cosine_type(*params, joint=Joint(-2.0, 2.0), verbose=True)
    while not cosine:
        # params = np.random.rand(6)*2.0 - 1.0
        params = get_singular_tangent_type_params()
        tangent_type(*params, joint=Joint(-1.2, 2.2), verbose=True)
