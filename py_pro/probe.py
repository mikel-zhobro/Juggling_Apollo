import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time, plotIterations, plt, flyTime2HeightAndVelocity
from juggling_apollo.Simulation import Simulation, plot_simulation
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC

import turtle
print(plan_ball_trajectory(0.65))
print(plan_ball_trajectory(1))
print(plan_ball_trajectory(1.5))
print(plan_ball_trajectory(9))



# %%

# Height

# E = Catch_point - Throw_point(vertical)

# F = Catch_point - Throw_point(horizontal) <- 3d case

# tau = d+e

# Dwell-rate r = d/(d+r)

# r_c_t = t_catch/(t_catch+t_throw)= t_catch/d, normally > 0.5 as we need more time to catch
# t_catch+t_throw = d. This is no parameter normally as deceleration for catching is of no importance
# when fingers are present. In our case we have no fingers.


def calc_minjerk(tau, dwell_ratio, catch_throw_ratio, E, F, n_b, n_a):
  """ Calc list of sub
      For up to 2 armes the juggling can always be performed in a 2D surface.
      Designing the trajectory of the hand corresponds to planing the x(t) and z(t) trajectories

      2 Ball, 1 Hand case (easy):
        The movement in x direction takes place only during the time the hand is free(e)
      nb Balls, 1 Hand case (general):


  Args:
      - d: time the ball spends on the hand (d=t_catch+t_throw)
       -- t_catch: time from ball impact to the ruhe-position(where we are ready again to throw)
       -- t_catch: time from ruhe position to throw position
      - e: time the hand is free

      tau ([double]): time from catch to catch (tau = d+e), for us this is the shortest time we require to learn everthing
      dwell_ratio ([double]): d/(d+r)
      catch_throw_ratio ([double]): t_catch_(t_catch+t_throw) = t_catch/d
      E ([type]): vertical distance between catch and throw position
      F ([type]): horizontal distance between catch and throw position
      n_b ([int]): number of balls
      n_a ([int]): number of arms
  """
  # -> calc H -> T_fly, v0
  assert n_a<3, "We can have up to 2 arms"
  pass


def traj_nb_2_na_1(tau, dwell_ration, catch_throw_ratio, E):
  # no x-movement
  nb = 2
  na = 1

  # times
  d = tau * dwell_ration  # time the ball spends on the hand
  e = tau - d             # time the hand is free
  Tfly = 2*e + d

  # throw and catch position
  t_catch = catch_throw_ratio * d
  t_throw = d - t_catch

  z_throw = 0.0
  z_catch = z_throw + E


  H, ub_0 = flyTime2HeightAndVelocity(Tfly)

  t_end = self.t_f + self.t_h
  t0 = 0;           t1 = t_throw;    t2 = t1 + self.t_f/2;  t3 = t_end - t_catch;  t4 = t_end
  x0 = self.x_0[0]; x1 = 0;          x2 = x0;               x3 = 0;                x4 = x0
  u0 = self.x_0[2]; u1 = ub_0;       u2 = -ub_0/2;          u3 = -ub_0/2;          u4 = u0


from juggling_apollo.MinJerk import get_minjerk_xyz
from juggling_apollo.JugglingPlanner import calc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


E = 0.15
# E = 0.0
tau = 0.5
dwell_ration = 0.6
catch_throw_ratio = 0.5
T_throw, T_hand, ub_throw, T_empty, H,  z_catch = calc(tau, dwell_ration, catch_throw_ratio, E)
T_fly = T_hand + 2*T_empty
T_FULL = T_throw + T_fly


# Init state
x_ruhe = -0.2                # starting position for the plate
x0 = [x_ruhe, x_ruhe, 0, 0]  # the plate and ball in ruhe
x000 =  [[x0[0], H], x0[1], [x0[2],0], x0[3]]

smooth_acc = False
ub_catch = -ub_throw*0.9
i_a_end = None
# a0 = None;        a1 = None;       a2 = None;          a3 = None
tt=[0,        T_throw,     T_throw+T_empty,    T_FULL-T_empty,   T_FULL  ]
xx=[[x0[0],    0.0,         z_catch,            0.0,              z_catch ],
    [0.0,    0.0,         -z_catch*3,           -z_catch*3,         0.0 ]]
uu=[[x0[2],    ub_throw,    ub_catch,           ub_throw,         ub_catch],
    [x0[2],    0.0,    0.0,           0.0,         0.0]]

xxx = get_minjerk_xyz(dt, tt, xx, uu, smooth_acc)

ax.plot(np.arange(len(xxx[0])), xxx[1], xxx[0])
ax.scatter([0,len(xxx[0])], xxx[1][[0,-1]], xxx[0][[0,-1]])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

plt.figure()
plt.plot(xxx[1])
plt.plot(xxx[0])
plt.show()
