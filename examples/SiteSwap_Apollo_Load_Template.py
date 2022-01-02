import numpy as np
import matplotlib.pyplot as plt


import __add_path__
from utils import plot_info, load

np.set_printoptions(precision=4, suppress=True)


FILENAME= "examples/data/AllJoints3/2021_12_09-16_59_04siteswap_joint_[0, 1, 2, 3, 5]_alpha_[18. 18. 18. 18. 18. 18. 18.]_eps_0.001_freq_domain_cart_err_off"
ld = load(FILENAME)


if False:
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(ld.T_traj[:,0,3], ld.T_traj[:,1,3], ld.T_traj[:,2,3], 'gray')
    plt.show()

plot_info(1, -1, ld.learnable_joints,
          joints_q_vec=ld.joints_q_vec, q_traj_des=ld.q_traj_des_vec[-1],
          u_ff_vec=ld.u_ff_vec, q_v_traj=ld.joints_vq_vec[-1,],
          joint_torque_vec=ld.joint_torque_vec, cartesian_error_norms =ld.cartesian_error_norms,
          disturbanc_vec=ld.disturbanc_vec, d_xyz_rpy_vec=ld.d_xyz_rpy_vec[-1], joint_error_norms=ld.joint_error_norms,
          v=False, p=True, dp=True, e_xyz=True, e=True, torque=True)
