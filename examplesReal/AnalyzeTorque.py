import numpy as np
import time

import __add_path__
from juggling_apollo.utils import steps_from_time, plt
from juggling_apollo.JugglingPlanner import calc
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys2
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, save, colors, line_types, plot_info, load


np.set_printoptions(precision=4, suppress=True)



with open('examplesReal/dataReal/TorqueTest/list_files.txt') as topo_file:
    for filename in topo_file:
        filename = filename.strip()  # The comma to suppress the extra new line char
        
        ld = load(filename)

        plot_info(0.004, joint_torque_vec=ld.joint_torque_vec,
                v=False, p=False, dp=False, e_xyz=False, e=False, torque=True)