import numpy as np

import __add_path__
from utils import plot_info, load

np.set_printoptions(precision=4, suppress=True)


with open('examplesReal/dataReal/StepResponseTest/list_files.txt') as topo_file:
    for filename in topo_file:
        filename = filename.strip()  # The comma to suppress the extra new line char
        
        ld = load(filename)

        plot_info(ld.dt, learnable_joints=ld.learnable_joints,
                u_ff_vec=ld.u_ff[np.newaxis, :], q_v_traj=ld.q_v_traj[1:],
                v=True, p=False, dp=False, e_xyz=False, e=False, torque=False, N=1)