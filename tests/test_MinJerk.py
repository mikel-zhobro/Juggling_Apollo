import numpy as np

import __add_path__
from ApolloILC.utils import steps_from_time, plt
from ApolloPlanners import utils, MinJerk
from ApolloILC.settings import dt, g


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
  [Tb, ub_0] = utils.plan_ball_trajectory(h_b_max, d1, d2)
  ub_T = 0

  # B] Plate Trajectory
  # 1) Free start and end acceleration
  x, v, a, j = MinJerk.get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, ub_0, ub_T)
  MinJerk.plotMinJerkTraj(x, v, a, j, dt, 'Free start and end acceleration')

  # 2) Free end acceleration
  x, v, a, j = MinJerk.get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0)
  MinJerk.plotMinJerkTraj(x, v, a, j, dt, 'Free start and end acceleration')

  # 3) Set start and end acceleration
  x, v, a, j = MinJerk.get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0, ap_T)
  MinJerk.plotMinJerkTraj(x, v, a, j, dt, 'Free start and end acceleration')


def test2():
  ta = 0; tb = 0.9032/2
  x_ta=-0.425; x_tb=0; u_ta=0; u_tb=4.4287; a_ta=None; a_tb=None

  x1, v1, a1, j1 = MinJerk.get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb)

  tb2=tb+0.9032
  x_ta=0; x_tb=0; u_ta=4.4287; u_tb=-u_ta/99
  # a_ta = a1[-1]
  x2, v2, a2, j2 = MinJerk.get_min_jerk_trajectory(dt, tb, tb2, x_ta, x_tb, u_ta, u_tb, a_ta)

  tb3=tb2+0.9032/2
  x_ta=0; x_tb=-0.4
  u_ta=u_tb; u_tb=0
  # a_ta = a2[-1]
  # a_tb = 0
  x3, v3, a3, j3 = MinJerk.get_min_jerk_trajectory(dt, tb2, tb3, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)

  intervals = {(ta/dt, tb/dt), (tb/dt, tb2/dt), (tb2/dt, tb3/dt)}
  colors = ('gray', 'gray', 'red')
  MinJerk.plotMinJerkTraj(np.concatenate((x1, x2, x3)),
                  np.concatenate((v1, v2, v3)),
                  np.concatenate((a1, a2, a3)),
                  np.concatenate((j1, j2, j3)), dt, "Plate Free Motion", intervals, colors)


def test3():
  dt = 0.007
  x_0 = [-0.425, -0.425, 0, 0]
  t_f = 0.904
  t_h = t_f/2
  ub_0 = 4.4287
  N_1 = steps_from_time(t_h+t_f, dt)
  # new MinJerk
  t0 = 0;      t1 = t_h/2; t2 = t1 + t_f
  x0 = x_0[0]; x1 = 0;     x2 = 0
  u0 = x_0[2]; u1 = ub_0;  u2 = -ub_0/6

  x, v, a, j = MinJerk.get_multi_interval_minjerk_1D(dt, smooth_acc=True,
                                      tt=(t0, t1, t2),
                                      xx=(x0, x1, x2),
                                      uu=(u0, u1, u2))
  assert x.size == N_1, "SIZE NO MATCH: " + str(N_1) + " != " + str(x.size)
  MinJerk.plotMinJerkTraj(x, v, a, j, dt, "Plate Free Motion(smooth acc)")


def test_all():
  test1()
  test2()
  # test3()


if __name__ == "__main__":
    test_all()
    plt.show()