# Juggling Apollo


![Apollo Robot](https://am.is.mpg.de/uploads/ckeditor/pictures/80/content_Apollo2__1600x1067_.jpg)


This tutorial assumes that the Apollo software is installed successfully.
If that is not done, look into [Apollo : start here](https://atlas.is.localnet/confluence/display/AMDW/Apollo+%3A+start+here) and [MPI System Apollo - Getting Started](https://atlas.is.localnet/confluence/display/AMDW/MPI+System+Apollo+-+Getting+Started) or contact [Vincent Berenz](https://ei.is.mpg.de/person/vberenz).
Once you do that, you should be able to start the simulation and run simple scripts to read the state of Apollo's arms or send position/velocity commands.


We then clone this repo and install its dependencies:
```
git clone https://gitlab.com/learning-and-dynamical-systems/juggling_apollo.git
pip install -r /path/to/requirements.txt
```

## Directory Structure
The essential parts of the code are organized as follows:

```
main.py                             # main script (maybe parse arguments here too)
config.py                           # file with all the parameters used across all modules

Requirements.py                     # file with the dependencies

ApolloILC
├── ILC.py                          # defines class to put ilc components together and define the update steps
├── DynamicSystem.py                # defines the state space equations of the plant we want to use in ILC
├── LiftedStateSpace.py             # unrolls the state space equations and creates the LSS mappings for the ILC
├── KalmanFilter.py                 # implementation of a simple KF for disturbance estimation
├── OptimLss.py                     # the optimization problem to compute feedforward input for next iteration
├── settings.py                     # contains physical parameters used in ILC
└── utils.py                        # contains helper functions


ApolloInterface
├── Apollo_It.py                    # loader for the something-something v2 dataset
├── display_robot_state.py          # exampe 1: print out the state information of Apollo
├── position_control.py             # exampe 2: send position control commands
├── velocity_control.py             # exampe 3: send velocity control commands
├── globs.py                        # file containing parameters for start_apollo.py
└── start_apollo.py                 # script to start the simulation in different modes


ApolloKinematics
├── ApolloKinematics.py             # defines class with Apollo's kinematics(fk, ik, seq_ik, seq_fk, etc.)
├── DHFK.py                         # defines class for denavit-hartenberg forward kinematics
├── PinFK.py                        # defines class for pinocchio based forward and inverse kinematics
├── AnalyticalIK.py                 # defines class for the analytical inverse kinematics
├── Sets.py                         # defines helper classes(range and set) required in AnalyticalIK
├── utilities.py                    # contains helper functions
├── settings.py                     # contains Apollo related information(names of joints, limits etc.)
└── tests                           # contains tests for jacobian computations and inverse kinematics computations
    ├── check_jac.py
    └── check_ik_real_dh.py

ApolloPlanners
├── MinJerk.py                      # contains functions for minimum jerk interpolation
├── SiteSwapPlanner.py              # defines a juggling pattern parser for arbitrary patterns and hands
├── SiteSwapJointPlanner.py         # transforms plans for usage in Apollo
├── JugglingPlanner.py              # transforms plans for usage in Apollo
├── OneBallThrowPlanner.py          # trajectory genneration for simple catch-throw
└── utils.py                        # contains helper functions


examples
├── TemplateLearn.py                                            # the full dataset
├── SiteSwap_Apollo_Learn_Template.py                           # the full dataset
├── SiteSwap_Apollo_LoadNLearn_Template.py                      # the full dataset
├── Throw_Learn.py
├── Throw_LoadNLearn.py
└── utils.py                                                    #

tests
└── test_LiftedStateSpace.py        # test LSS
```
