from utils import flyTime2HeightAndVelocity, plt
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory
from settings import g
import math
import numpy as np


def calc(tau, dwell_ration, E, slower=1.0):
  # times
  T_hand = tau * dwell_ration  # time the ball spends on the hand
  T_empty = tau - T_hand       # time the hand is free
  T_fly = 2*tau - T_hand       # time the ball is in the air

  # positions
  z_throw = 0.0
  z_catch = z_throw + E

  # ball height and velocity
  H, ub_throw = flyTime2HeightAndVelocity(T_fly)

  return T_hand*slower, T_empty*slower, ub_throw/slower, H, z_catch

def traj_nb_2_na_1(T_throw, T_hand, ub_catch, ub_throw, T_empty, z_catch, x_0, dt, smooth_acc, plot=False):
  # 2balls 1hand
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

def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).
    """
    while b:
        a, b = b, a%b
    return a

def lcm(a,b):
    "Calculate the lowest common multiple of two integers a and b"
    return a*b//gcd(a,b)

class Interval():
  def __init__(self):
      self.xa = 0.0
      self.xb = 0.0
      self.va = 0.0
      self.vb = 0.0



class JugglingPeriod():
  def __init__(self, ):
    pass

class Jugglinghand():
  def __init__(self, N):
    pass

class JugglingPlanner():
  def __init__(self, nh=2, h=0.3, w=0.4, r_dwell=0.5):
    """[summary]

    Args:
        nh ([int]): number of hands
        h ([float]): height of the standard 3-throw
        w ([float]): weidth of the hand positions
        r_dwell ([float]): ratio of time the ball stays in hand and time between two catches of the same hand

        We assume here that hands throw the balls one after the other, as such they are not allowed to throw simultaniously.
        That means that a hand can throw at most every second step. We call this time, a hand period,
        i.e. the time between two catches of the same hand if we are are performing a cascade or shower with 2+ balls.
        tH = 2*tB

        h----> tf3 --r_dwell --> tB, t_handperiode
    """
    assert 0 < nh <= 2, 'For the moment we accept only 2 hands or less.'

    self.hands = None  # this variable keeps the juggling trajectories for each hand
    self.nh = nh
    self.h = h
    self.w = w
    self.r_dwell = r_dwell

    tf3 = math.sqrt(2.0*h/g)  # time of flight for a 3-throw

    # tf3 = (3*tB - 2*r_dwell*tH) = (3*tB - 4*r_dwell*tB) = tB(3 - 4*r_dwell)
    # where 2*r_dwell*tH is the dwell time for the two hands for one ball throw
    self.tB = tf3 / (3.0 - 4*self.r_dwell)
    self.tH = 2.0 * self.tB

  def plan(self, pattern=(3,)):
    """Plan the juggling trajectories for the hand

    Args:
        pattern (tuple, optional): The siteswap pattern to be juggled. Defaults to (3,).
    """
    pattern = tuple(pattern)

    # Check if pattern is jugglable
    assert self.isJugglable(pattern), 'The given pattern {} is not jugglable'.format(pattern)

    # Get catch-throw time diagram
    ct_time_diagram = self.getCatchThrowTimeDiagram(pattern)

    # Create time diagram (M x nr hands x(c_i, t_i))
    # column i gives the c-t period for hand i
    ct_period_per_hand = self.getCatchThrowTimeHandDiagram(ct_time_diagram)
    self.plot(ct_period_per_hand)

  def isJugglable(self, pattern):
    jugglable = True
    N = len(pattern)
    tmp = [False] * N

    # t1  t2  t3  t4
    #+ 0   1   2   3
    #  _   _   _   _  modn
    # In order to be jugglable all _, .., _ have to be different
    for i, n in enumerate(pattern):
      imodn = (i+n) % N
      if tmp[imodn]:
        jugglable = False
        break
      tmp[imodn] = True
    return jugglable

  def getCatchThrowTimeDiagram(self, pattern):
    # t1  t2  t3  t4
    #+ 0   1   2   3  modn
    # c1  c2  c3  c4

    N = len(pattern)
    ct_time_diag = [None]*N
    for i,t in enumerate(pattern):
      catch_index = (t+i)%N  # at what index t is catched
      ct_time_diag[catch_index] = (t, pattern[catch_index])   # pattern[catch_index]: what is thrown at the index t is catched
    return ct_time_diag

  def getCatchThrowTimeHandDiagram(self, ct_time_diag):
    """Computs the smallest required period of catch-throws for the nr of hands.

    Args:
        ct_time_diag ([type]): catch-throw time(only) diagram, where the nr of hands are not considered.

    Returns:
        [np.array((M_h, self.nh))]: catch-throw time-hand diagram, considering the nr of hands
    """
    N = len(ct_time_diag)
    # Calculate nr of beats required for each hand to perform perform all c-t it will ever face at least once(least common multiple)
    M = lcm(N, self.nh)
    # nr of c-t a hand performs in one period
    M_h = M//self.nh
    # fill a M_h x nh matrix with (c,t) pairs
    ct_period_per_hand = np.zeros((M_h, self.nh, 2), dtype=np.int)
    for i in range(M_h):
      for j in range(self.nh):
        k = (j + i*self.nh) % N  # index for ct_time_diag
        ct_period_per_hand[i,j] = ct_time_diag[k]

    return ct_period_per_hand


  def plot(self, ct_period_per_hand, repetition=1):
    N = ct_period_per_hand.shape[0]
    N_plot = N * repetition
    N_beats = N*self.nh

    def getConnection(ct, actual_beat_, y0_, y1_):
      n_t = ct[1]
      n_c = ct[0]

      same_hand_t = (n_t % self.nh) == 0
      same_hand_c = (n_c % self.nh) == 0


      def func(actual_beat, catch_beat, same_hand, y0, y1, throw):
        actual_beat, catch_beat, same_hand = float(actual_beat), float(catch_beat), float(same_hand)
        start_beat = max(min(actual_beat, N_beats), 0)
        end_beat = max(min(catch_beat, N_beats), 0)
        x = np.linspace(actual_beat, catch_beat)  # make sure it is cuted properly
        if same_hand:  # parable that starts at 0 and ets at 0 with height 1.0
          middle_beat = (actual_beat + catch_beat)/2
          n_t_2 = (catch_beat - actual_beat)/2
          y = (1.0 - ((x-middle_beat)/n_t_2)**2)
          y = y0 - math.copysign(1, y1-y0)*y
        else:
          y0, y1 = (y1, y0) if not throw else (y0, y1)
          a = (y1-y0)/(catch_beat - actual_beat)
          b = y0 - a*actual_beat
          y = a*x + b # straight line starting at actual_beat and ending at actual_beat+nt with y=1
        return x, y

      x_throw, y_throw = func(actual_beat_, actual_beat_+n_t, same_hand_t, y0_, y1_, True)
      x_catch, y_catch = func(actual_beat_-n_c, actual_beat_, same_hand_c, y0_, y1_, False)
      return x_throw, y_throw, x_catch, y_catch


    fig, ax = plt.subplots(1, 1)
    # start with Right hand
    r_height = 5
    r_x = self.nh * np.arange(0, N_plot+1)
    r_y = r_height * np.ones_like(r_x)
    r_ct =[str(tuple(ct_period_per_hand[n%N,0])) for n in range(N_plot+1)]
    ax.axhline(y=r_height, color='k', linestyle='-')
    ax.scatter(r_x, r_y)
    for i, txt in enumerate(r_ct):
      ax.annotate(txt, (r_x[i], r_y[i]), (r_x[i]-0.2, r_y[i]-0.5))

    for n in range(N+1):
      ct = ct_period_per_hand[n%N,0]
      actual_beat = n*2
      x1,y1, x2, y2 = getConnection(ct, actual_beat,5, 10)
      plt.plot(x1, y1, x2, y2, color='k')


    # Left hand
    if self.nh > 1:
      l_height = 10
      l_x = self.nh * np.arange(1, N_plot+1) - 1
      l_y = l_height * np.ones_like(l_x)
      l_ct =[str(tuple(ct_period_per_hand[n%N,1])) for n in range(N_plot)]
      ax.axhline(y=l_height, color='k', linestyle='-')
      ax.scatter(l_x, l_y)
      for i, txt in enumerate(l_ct):
        ax.annotate(txt, (l_x[i], l_y[i]), (l_x[i]-0.2, l_y[i]+0.2))

      for n in range(N+1):
          ct = ct_period_per_hand[n%N,1]
          actual_beat = n*2 +1
          x1,y1, x2, y2 = getConnection(ct, actual_beat, 10, 5)
          plt.plot(x1, y1, x2, y2, color='k')


    ax.grid(axis='x')
    ax.set_xticks(np.arange(0, N_beats+1, 1))
    # ax.set_xticklabels(["{:d}".format(int(v)) for v in x_axis])
    ax.set_xlim((-0.2, N_beats+0.2))   # set the xlim to left, right
    ax.set_ylim((3, 12))   # set the xlim to left, right
    plt.show()

if __name__ == "__main__":
  jp = JugglingPlanner(nh=2)
  jp.plan((3,3,3))
  jp.plan((4,4,4,4))
  jp.plan((3, 3, 4, 2, 3, 3, 4, 2))

  # import numpy as np
  # import matplotlib.pyplot as plt

  # # the data:
  # x_axis = np.array(
  #     [ 30.,  40.,  50.,  60.,  50.,  40.,  30.,  20.,  10.,   0.,  10.,
  #       20.,  30.,  50.,  60.,  50.,  40.,  30.,  20.,  10.,   0.,  20.,
  #       30.,  40.,  50.,  60.,  50.,  40.,  30.])
  # y_axis = np.array(
  #     [ 0.65029748,  0.65766552,  0.66916997,  0.69591696,  0.68025489,
  #     0.66391214,  0.65369247,  0.64442389,  0.63778457,  0.62621366,
  #     0.63776906,  0.64300699,  0.64674221,  0.66853973,  0.69495993,
  #     0.67951167,  0.6633013 ,  0.65302631,  0.64497037,  0.63772925,
  #     0.62649416,  0.64300646,  0.64669281,  0.65656247,  0.66868399,
  #     0.69528981,  0.68042033,  0.66314567,  0.65407716])

  # x=range(len(x_axis))

  # fig, ax = plt.subplots(1, 1)
  # ax.set_xticks(x) # set tick positions
  # # Labels are formated as integers:
  # ax.set_xticklabels(["{:d}".format(int(v)) for v in x_axis])
  # ax.plot(x, y_axis)

  # fig.canvas.draw() # actually draw figure
  # plt.show() # enter GUI loop (for non-interactive interpreters)