import numpy as np
import time

import __add_path__

from utils import plot_info, load


np.set_printoptions(precision=4, suppress=True)



with open('examplesReal/dataReal/TrapezoidTest/list_files.txt') as topo_file:
    for filename in topo_file:
        filename = filename.strip()  # The comma to suppress the extra new line char
        
        ld = load(filename)
    print(filename)

    plot_info(ld.dt, -1, ld.learnable_joints, 
              ld.joints_q_vec, ld.q_traj_des, ld.u_ff_vec, ld.joints_vq_vec[-1], 
              ld.joint_torque_vec,
              ld.disturbanc_vec, ld.d_xyz, ld.error_norms,
              v=True, p=True, dp=False, e_xyz=True, e=True, torque=True)