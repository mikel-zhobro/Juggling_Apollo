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

def plotConnection(ax, actual_beat, catch_beat, y0, y1, same_hand):
  y0, y1 = float(y0), float(y1)
  actual_beat, catch_beat = float(actual_beat), float(catch_beat)
  x = np.linspace(actual_beat, catch_beat)  # make sure it is cuted properly
  if same_hand:  # parable that starts at 0 and ets at 0 with height 1.0
    middle_beat = (actual_beat + catch_beat)/2
    n_t_2 = (catch_beat - actual_beat)/2
    y = 0.4*(1.0 - ((x-middle_beat)/n_t_2)**2)
    y = y0 - math.copysign(1, y1-y0)*y
  else:
    a = (y1-y0)/(catch_beat - actual_beat)
    b = y0 - a*actual_beat
    y = a*x + b # straight line starting at actual_beat and ending at actual_beat+nt with y=1
  ax.plot(x, y, color='k')

class CatchThrow():
  def __init__(self, T, h, beat_nr, n_c, n_t, h_c, h_t):
    self.T = T
    self.h = h
    self.beat_nr = beat_nr
    self.n_c = n_c
    self.n_t = n_t
    self.h_c = h_c
    self.h_t = h_t


  def plotTimeDiagram(self, ax, period=0):
    actual_beat = self.beat_nr + period*self.T
    x , y = actual_beat, self.h
    # Plot point and annonate
    ax.scatter(x, y)
    ax.annotate(str((self.n_c, self.n_t)), (x, y), (x-0.2, y-0.5))
    # Plot catch and throw at this catch-throw point
    plotConnection(ax, actual_beat-self.n_c, actual_beat, self.h_c, self.h, self.h_c==self.h)
    plotConnection(ax, actual_beat, actual_beat+self.n_t, self.h, self.h_t, self.h==self.h_t)


class JugglingHand():
  def __init__(self, h, ct_period, hand_positions, Th):
    """Represent one juggling hand. Holds trajectory information about how tha hand should move.

    Args:
        ct_period ([np.array(T, 6)]): where T is nr of beats in one period and 6 are the ct elements that describe a catch-throw
                                        (h, beat_nr, n_c, n_t, h_c, h_t)
        hand_positions  ([np.array(nh, 3)]): the x,y,z positions for each hand
    """
    self.h = h
    self.Th = Th  # == nr of hands, == nr of beats from catch to catch
    self.T = len(ct_period)

    self.ct_period = ct_period
    self.hand_positions = hand_positions

  def plot(self, ax):
    # Plot hand horizontal lines
    ax.axhline(y=self.h)
    # Plot catche-throw point&trajectories
    self.ct_period[0].plotTimeDiagram(ax, 1) # 1 more than the period(first ct is included twice) for visual effects
    for ct in self.ct_period:
      ct.plotTimeDiagram(ax)


class JugglingPlan():
  def __init__(self, hands, hand_positions, tB, r_dwell):
    self.Nh = len(hands)
    self.T = hands[0].T
    self.hands = hands
    self.handPositions = hand_positions

    # Calculate hand-time (the time between two catches of the same hand)
    self.tH = self.Nh * tB
    # Compute dwell-time: time in one hand-time where the ball is on the hand.
    self.tDwell = r_dwell * self.tH

  def plotHandTrajectories(self):
    # Plot hands
    fig, ax = plt.subplots(1, 1)
    ax.scatter(self.handPositions[:,0], self.handPositions[:,1])
    # [h.plot(ax) for h in self.hands]
    plt.show()

  def plotTimeDiagram(self):
    fig, ax = plt.subplots(1, 1)
    N_beats = self.T * self.Nh
    heights = np.arange(self.Nh)

    [h.plot(ax) for h in self.hands]

    ax.grid(axis='x')
    ax.set_xticks(np.arange(0, N_beats+1, 1))
    ax.set_yticks(heights)
    ax.set_yticklabels([r"H$_{}$".format(n) for n in range(self.Nh)])
    ax.set_xlim((-0.2, N_beats+0.2))
    ax.set_ylim((heights[0]-0.5, heights[-1]+0.5))
    plt.show()


class JugglingPlanner():
  def __init__(self, h=0.3, w=0.4, r_dwell=0.5):
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

        The goal of this class is to only make sure that the siteswap pattern is correct, parse it into per-hand information
        and initialize the jugglingPlan with it.

        The jugglingPlan can be then further used to compute the hand trajectories for realizing the desired pattern.
    """
    # assert 0 < nh <= 2, 'For the moment we accept only 2 hands or less.'

    self.hands = None  # this variable keeps the juggling trajectories for each hand
    self.h = h
    self.w = w

    # We calculate the beat time from the fly time of a 3 ball using 2 hands and r_dwell.
    # It is just a definition, in general the nr of hands can be different.
    # tf3 = (3*tB - 2*r_dwell*tH) = (3*tB - 4*r_dwell*tB) = tB(3 - 4*r_dwell)
    # where 2*r_dwell*tH is the dwell time for the two hands for one ball throw
    self.r_dwell = r_dwell
    tf3 = math.sqrt(2.0*h/g)  # time of flight for a 3-throw
    self.tB = tf3 / (3.0 - 4*self.r_dwell)

  def plan(self, nh, pattern=(3,), rep=1):
    """Plan the juggling trajectories for the hand

    Args:
        pattern (tuple, optional): The siteswap pattern to be juggled. Defaults to (3,).
    """
    pattern = tuple(pattern)

    # Check if pattern is jugglable
    assert self.isJugglable(pattern), 'The given pattern {} is not jugglable'.format(pattern)

    # Get catch-throw time diagram
    ct_time_diagram = self.getCatchThrowTimeDiagram(nh, pattern) * rep

    # Create time diagram (T x nr hands x 6) where T is the periode and each element(6,) gives: (h, beat_nr, n_c, n_t, h_c, h_t))
    ct_period_per_hand = self.getCatchThrowTimeHandDiagram(nh, ct_time_diagram)

    # Caresian positions of the hands. Are assumed to lie on the edges of a nh-polygon with side-length = w
    hand_positions = self.getHandPositions(nh)

    # List of JugglingHand objects, one for each hand.
    hands = self.initHands(nh, ct_period_per_hand, hand_positions)

    # The juggling plan which is intended for further usage(check its API)
    juggPlan = JugglingPlan(hands, hand_positions, self.tB, self.r_dwell)

    return juggPlan

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

  def getCatchThrowTimeDiagram(self, nh, pattern):
    # t1  t2  t3  t4
    #+ 0   1   2   3  modn
    # c1  c2  c3  c4
    # for each step 1..N: save (n_c, n_t):
    # (length of throw we are catching in nr of beats, length of throw in nr of beats)
    N = len(pattern)
    ct_time_diag = [None]*N
    for i, n_t in enumerate(pattern):
      # at what index throw t is catched
      catch_index = (n_t+i)%N
      # pattern[catch_index]: what is thrown at the timebeat where t is catched
      ct_time_diag[catch_index] = (n_t, pattern[catch_index])
    return ct_time_diag

  def getCatchThrowTimeHandDiagram(self, nh, ct_time_diag):
    """Computs the smallest required period of catch-throws for the nr of hands.

    Args:
        ct_time_diag ([type]): catch-throw time(only) diagram, where the nr of hands are not considered.

    Returns:
        [np.array((M_h, nh, 3))]: holds (hand_nr, beat_nr, n_c, n_t, h_c, h_t)
    """
    N = len(ct_time_diag) # Period
    # Calculate nr of beats required for each hand to perform perform all c-t it will ever face at least once(least common multiple)
    M = lcm(N, nh)
    # Nr of c-t a hand performs in one period
    M_h = M//nh
    # Fill a M_h x nh matrix with (c,t) pairs
    ct_period_per_hand = np.zeros((M_h, nh, 6), dtype=np.int)
    for i in range(M_h):
      for j in range(nh):
        time_beat = j + i*nh
        k = time_beat % N  # index for ct_time_diag
        # the rule is to always throw to higher hands(modulo)
        # and to be thrown to from lower hands(modulo)
        catch_from = (j-ct_time_diag[k][0]) % nh
        throw_to = (j+ct_time_diag[k][1]) % nh
        ct_period_per_hand[i,j] = (j, time_beat,) + ct_time_diag[k] + (catch_from, throw_to)
    return ct_period_per_hand

  def initHands(self, nh, ct_period_per_hand, hand_positions):
    hands = [None] * nh
    T = ct_period_per_hand.shape[0]
    for h in range(nh):
      hands[h] = JugglingHand(h, [CatchThrow(T*nh, *ct) for ct in ct_period_per_hand[:, h]], hand_positions=hand_positions, Th=nh)
    return hands

  def getHandPositions(self, nh):
    hand_positions = np.zeros((nh, 3))
    angle_step = np.pi*(float(nh) - 2.0)/float(nh) - np.pi
    angles  = np.arange(nh)*angle_step + np.pi
    x,y,z = 0.0, 0.0, 0.0
    for h in range(nh):
      hand_positions[h] = (x,y,z)
      # update x,y for next hand, assume hands are on the same height
      x = x + self.w * math.cos(angles[h])
      y = y + self.w * math.sin(angles[h])
    return hand_positions


if __name__ == "__main__":
  jp = JugglingPlanner()
  # jp.plan(1, pattern=(1,), rep=2).plotTimeDiagram()
  jp.plan(3, pattern=(4,4,4,4)).plotTimeDiagram()
  jp.plan(2, pattern=(3, 3, 4, 2, 3, 3, 4, 2)).plotTimeDiagram()