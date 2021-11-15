import numpy as np

import __add_path__
from utils import plot_info, load

np.set_printoptions(precision=4, suppress=True)


with open('examplesReal/dataReal/TorqueTest/list_files.txt') as topo_file:
    for filename in topo_file:
        filename = filename.strip()  # The comma to suppress the extra new line char
        
        ld = load(filename)

        plot_info(0.004, joint_torque_vec=ld.joint_torque_vec,
                v=False, p=False, dp=False, e_xyz=False, e=False, torque=True)