import numpy as np
import matplotlib.pyplot as plt


import __add_path__
from utils import plot_info, load
from juggling_apollo.MinJerk import plotMJ

np.set_printoptions(precision=4, suppress=True)


with open('examplesReal/dataReal/MinJerkTest/list_files.txt') as topo_file:
    for filename in topo_file:
        filename = filename.strip()  # The comma to suppress the extra new line char

        ld = load(filename)
        print(filename)

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

        plot_info(ld.dt, -1, ld.learnable_joints,
                  ld.joints_q_vec, ld.q_traj_des, ld.u_ff_vec, ld.joints_vq_vec,
                  ld.joint_torque_vec,
                  ld.disturbanc_vec, ld.d_xyz, ld.error_norms,
                  v=True, p=True, dp=False, e_xyz=False, e=False, torque=False)