import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC


print(plan_ball_trajectory(0.5))
print(plan_ball_trajectory(1))
print(plan_ball_trajectory(1.5))
print(plan_ball_trajectory(2))



