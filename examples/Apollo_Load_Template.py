import numpy as np
import matplotlib.pyplot as plt


import __add_path__
from utils import plot_info, load
from juggling_apollo.MinJerk import plotMJ

np.set_printoptions(precision=4, suppress=True)


FILENAME_err = "2021_12_01-11_32_51joint_[0, 1, 2, 3, 4, 5, 6]_alpha_[16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]_eps_0.0001_time_domain_cart_err_off"
FILENAME_no_err = "2021_12_01-11_37_14joint_[0, 1, 2, 3, 4, 5, 6]_alpha_[16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]_eps_0.0001_time_domain_cart_err_on"
ld = load('examples/data/AllJoints3/' + FILENAME_no_err)

if False:
    plotMJ(ld.dt, ld.tt, ld.yy, ld.uu)
    plotMJ(ld.dt, ld.tt, ld.zz, ld.uu)
    plt.show()


if False:
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(ld.xyz_traj_des[:,0], ld.xyz_traj_des[:,1], ld.xyz_traj_des[:,2], 'gray')
    plt.show()

plot_info(1, -1, ld.learnable_joints,
          joints_q_vec=ld.joints_q_vec, q_traj_des=ld.q_traj_des[1:], 
          u_ff_vec=ld.u_ff_vec, q_v_traj=ld.joints_vq_vec[-1,1:],
          joint_torque_vec=ld.joint_torque_vec,
          disturbanc_vec=ld.disturbanc_vec, d_xyz=ld.d_xyz_vec[-1], error_norms=ld.error_norms,
          v=False, p=True, dp=False, e_xyz=True, e=True, torque=False)
