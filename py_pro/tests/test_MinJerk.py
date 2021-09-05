import path_setter
import numpy as np
from juggling_apollo.utils import plan_ball_trajectory, steps_from_time
from juggling_apollo.MinJerk import get_min_jerk_trajectory, plotMinJerkTraj
from juggling_apollo.settings import dt, g


def test1():
  # Desired Trajectory planning example
  # Initialize disturbances
  d1 = 0         # disturbance1
  d2 = 0         # disturbance2

  # Initialize throw and catch point
  h_b_max = 1  # [m] maximal height the ball achievs
  x_p0 = 0
  x_pTb = x_p0
  ap_0 = 0
  ap_T = -g

  # A] Ball Height and Time
  [Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2)
  ub_T = 0

  # B] Plate Trajectory
  # 1) Free start and end acceleration
  x, v, a, j = get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, ub_0, ub_T)
  plotMinJerkTraj(x, v, a, j, dt, 'Free start and end acceleration')

  # 2) Free end acceleration
  x, v, a, j = get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0)
  plotMinJerkTraj(x, v, a, j, dt, 'Free start and end acceleration')

  # 3) Set start and end acceleration
  x, v, a, j = get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0, ap_T)
  plotMinJerkTraj(x, v, a, j, dt, 'Free start and end acceleration')


def test2():
  ta = 0; tb = 0.9032/2
  x_ta=-0.425; x_tb=0; u_ta=0; u_tb=4.4287; a_ta=None; a_tb=None

  x1, v1, a1, j1 = get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb)

  tb2=tb+0.9032
  x_ta=0; x_tb=0; u_ta=4.4287; u_tb=-u_ta/99
  a_ta = a1[-1]
  x2, v2, a2, j2 = get_min_jerk_trajectory(dt, tb, tb2, x_ta, x_tb, u_ta, u_tb, a_ta)

  tb3=tb2+0.9032/2
  x_ta=0; x_tb=-0.4
  u_ta=u_tb; u_tb=0
  a_ta = a2[-1]
  a_tb = 0
  x3, v3, a3, j3 = get_min_jerk_trajectory(dt, tb2, tb3, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)

  intervals = {(ta/dt, tb/dt), (tb/dt, tb2/dt), (tb2/dt, tb3/dt)}
  colors = ('gray', 'gray', 'red')
  plotMinJerkTraj(np.concatenate((x1, x2, x3)),
                  np.concatenate((v1, v2, v3)),
                  np.concatenate((a1, a2, a3)),
                  np.concatenate((j1, j2, j3)), dt, "Plate Free Motion", intervals, colors)


def test_all():
  test1()
  test2()


if __name__ == "__main__":
    test_all()
