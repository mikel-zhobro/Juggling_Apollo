import subprocess
import sys

import globs
import O8O_apollo as apollo


# IF we want to call it from terminal
# python start_apollo.py 1 1 # corresponds to both Ik_dynamics and bursting set to True
argv = sys.argv
N = len(argv)
assert N < 4, "Too many input arguments"

if N == 1:
    IK_dynamics = True
    bursting = False
elif N == 2:
    IK_dynamics = argv[1] == '1'
    bursting = False
else:
    IK_dynamics = argv[1] == '1'
    bursting = argv[2] == '1'




# # Meant for quick manual changes:
# IK_dynamics = True
# bursting = True

# globs.IK_dynamics = IK_dynamics
# globs.bursting    = bursting

apollo.set_inverse_dynamics(globs.IK_dynamics)
apollo.set_bursting(globs.bursting)

# Some Warning Prints
if not IK_dynamics:
    print("O8O Apollo (thread) task will be started without inverse dynamics")
    
if bursting:
    print("\nnext O8O Apollo (thread) task will be started in bursting mode")
    print("-- !! do *not* start on real robot !! --\n")
 
    
subprocess.call(['/bin/bash', '-i', '-c', "sa"])