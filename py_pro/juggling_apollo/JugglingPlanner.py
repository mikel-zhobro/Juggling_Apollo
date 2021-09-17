from utils import flyTime2HeightAndVelocity
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory

def traj_nb_2_na_1(tau, dwell_ration, catch_throw_ratio, E, x_0, dt, smooth_acc):
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


  H, ub_throw = flyTime2HeightAndVelocity(Tfly)
  ub_catch = -ub_throw

  # TODO: based on z_catch-x0[0] we need the min ub_catch for no disconnection
  #

  t_end = tau
  t0 = 0;      t1 = t_throw;    t2 = t_throw + e; t3 = t_end
  x0 = x_0[0]; x1 = 0;          x2 = z_catch;     x3 = x0
  u0 = x_0[1]; u1 = ub_throw;   u2 = ub_catch;    u3 = u0

  tt = [t0, t1, t2, t3]
  xx = [x0, x1, x2, x3]
  uu = [u0, u1, u2, u3]

  # x, v, a, j = get_minjerk_trajectory(dt,
  #                                     ta  =(t0, t1, t2), tb  =(t1, t2, t3),
  #                                     x_ta=(x0, x1, x2), x_tb=(x1, x2, x3),
  #                                     u_ta=(u0, u1, u2), u_tb=(u1, u2, u3))
  x, v, a, j = get_minjerk_trajectory(dt, tt=tt, xx=xx, uu=uu, smooth_acc=smooth_acc)
  print(
    "\n t_throw: " +str(t_throw)+
    "\n t_catch: " +str(t_catch)+
    "\n X: " +str(xx) +
    "\n U: " +str(uu)
        )
  # plotMinJerkTraj(x, v, a, j, dt, "TITLE")

  return x, ub_throw, ub_catch, H



if __name__ == "__main__":
  dt = 0.004
  x_0 = [-0.4, 0]
  E = 0.25
  tau = 0.53
  dwell_ration = 0.68
  catch_throw_ratio = 0.5
  smooth_accs = [False, True]
  for smooth in smooth_accs:
  # for cr in catch_throw_ratio:
    traj_nb_2_na_1(tau, dwell_ration, catch_throw_ratio, E, x_0, dt, smooth)
