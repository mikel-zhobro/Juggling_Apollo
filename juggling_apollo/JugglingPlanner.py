from utils import flyTime2HeightAndVelocity, plt
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory

def calc(tau, dwell_ration, catch_throw_ratio, E):
  # TODO: based on z_catch-x0[0] we need the min ub_catch for no disconnection
  # no x-movement
  nb = 2
  na = 1

  # times
  T_hand = tau * dwell_ration  # time the ball spends on the hand
  T_empty = tau - T_hand             # time the hand is free
  T_fly = 2*T_empty + T_hand

  # throw and catch position
  T_catch = catch_throw_ratio * T_hand
  T_throw = T_hand - T_catch

  z_throw = 0.0
  z_catch = z_throw + E

  H, ub_throw = flyTime2HeightAndVelocity(T_fly)

  return T_throw, T_hand, ub_throw, T_empty, H, z_catch

def traj_nb_2_na_1(T_throw, T_hand, ub_catch, ub_throw, T_empty, z_catch, x_0, dt, smooth_acc, plot=False):
  #
  ub_catch = -ub_throw
  t0 = 0;        t1 = T_throw;    t2 = t1+T_empty;  t4 = t2+T_hand;  t5 = t4+T_empty
  x0 = x_0[0];   x1 = 0;          x2 = z_catch;     x4 = 0;          x5 = x2
  u0 = x_0[1];   u1 = ub_throw;   u2 = ub_catch;    u4 = ub_throw;   u5 = u2

  tt = [t0, t1, t2, t4, t5]
  xx = [x0, x1, x2, x4, x5]
  uu = [u0, u1, u2, u4, u5]
  x, v, a, j = get_minjerk_trajectory(dt, tt=tt, xx=xx, uu=uu, smooth_acc=smooth_acc)

  if plot:
    print(
    "\n X: " +str(xx) +
    "\n T: " +str(tt) +
    "\n U: " +str(uu)
        )
    title = "Min-Jerk trajectory with " +  ("" if smooth_acc else "non") +"-smoothed acceleration."
    plotMinJerkTraj(x, v, a, j, dt, title, tt=tt[0:4], xx=xx[0:4], uu=uu[0:4])
  return x



if __name__ == "__main__":
  dt = 0.004
  x_0 = [-0.4, 0]
  E = 0.25
  tau = 0.53
  dwell_ration = 0.68
  catch_throw_ratio = 0.5
  smooth_accs = [False, True]
  for smooth in smooth_accs:
    t_throw, t_catch, ub_catch, ub_throw, t_empty, H, z_catch = calc(tau, dwell_ration, catch_throw_ratio, E)
    traj_nb_2_na_1(t_throw, t_catch, ub_catch, ub_throw, t_empty, z_catch, x_0, dt, smooth, True)

  plt.show()