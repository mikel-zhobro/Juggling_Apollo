from numpy.lib import ufunclike
from utils import flyTime2HeightAndVelocity, plt
from MinJerk import get_min_jerk_trajectory, plotMinJerkTraj, get_minjerk_trajectory, get_minjerk_xyz
from utils import plot_intervals, plt, steps_from_time
from settings import g, dt
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

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

class MinJerkTraj():
  def __init__(self, dt, tt, xx, vv):
    self.dt = dt
    self.tt = tt
    self.xx = xx
    self.vv = vv
    self.N_Whole = steps_from_time(tt[-1] - tt[0], dt)
    self.ttt = np.arange(0, self.N_Whole) * self.dt
    self.ttt = np.linspace(0, self.N_Whole, self.N_Whole) * self.dt
    xxx, vvv, aaa, jjj = get_minjerk_xyz(dt, tt, xx, vv, i_a_end=0, only_pos=False)
    self.xxx = np.array(xxx).T
    self.vvv = np.array(vvv).T
    self.aaa = np.array(aaa).T
    self.jjj = np.array(jjj).T


  def plot(self, ax):
    # Plot path
    a = ax.plot3D(self.xxx[:,0], self.xxx[:,1], self.xxx[:,2])

    # Plot arrows
    # ts = [self.N_Whole//4, 3*self.N_Whole//4]
    ts = [0, self.N_Whole//2]
    ax.quiver(self.xxx[ts,0], self.xxx[ts,1], self.xxx[ts,2],
              self.vvv[ts,0], self.vvv[ts,1], self.vvv[ts,2],
              length=0.07, normalize=True, color=a[0].get_color())
    return a[0].get_color()


class BallTraj():
  def __init__(self, t_t, p_t, v_t):
    self.t_t, self.p_t, self.v_t = t_t, p_t, v_t
    self.tt = np.linspace(0, self.t_t, 50)
    self.xxxBall = np.zeros((50, 3))
    self.xxxBall[:] = p_t
    self.xxxBall[:,0] += self.tt * self.v_t[0]
    self.xxxBall[:,1] += self.tt * self.v_t[1]
    self.xxxBall[:,2] += self.tt * self.v_t[2] - g*self.tt**2*0.5

  def plot(self, ax, col):
    # plot path
    ax.plot3D(self.xxxBall[:,0], self.xxxBall[:,1], self.xxxBall[:,2], color=col, linestyle='--')

    # plot arrows
    # ts = [0, 25, -1]
    ts = [25]
    ax.quiver(self.xxxBall[ts,0], self.xxxBall[ts,1], self.xxxBall[ts,2],
              self.v_t[0]+self.tt[ts]*0, self.v_t[1]+self.tt[ts]*0,  self.v_t[2] - self.tt[ts]*g,   # direction
              length=0.08, normalize=True, color=col)


class CatchThrow():
  def __init__(self, h, beat_nr, n_c, n_t):
    # We throw from self.h.position.
    # We catch 'swingSize' behind the self.h.position in the direction of the ct that throws to us.

    self.h = h                  # hand this ct belongs to
    self.beat_nr = beat_nr      # beat nr during which this ct is performed
    self.n_c = n_c              # length of throw we are catching in nr of beats
    self.n_t = n_t              # length of throw we are throwing in nr of beats
    self.ct_c = None            # ct where catch comes from
    self.ct_t = None            # ct where throw goes to

    self.t_t = None             # flight time of the ball we catch
    self.t_dwell = None         # time we have from the catch to the throw
    self.t_hand = None          # time we have from the catch to the next catch
    self.p_t = np.zeros(3)      # (x, y, z) position where we catch
    self.v_t = np.zeros(3)      # (vx, vy, vz) of the catching ball


    self.traj = None            # the catch-throw trajectory
    self.balltraj = None

  def addCatchThrowCTs(self, ct_c, ct_t):
    self.ct_c = ct_c            # ct where catch comes from
    self.ct_t = ct_t            # ct where throw goes to

  def initThrow(self, t_dwell, tB, swingSize):
    # we assume catch and throw points lie all at z=0
    self.t_t = self.n_t * tB - t_dwell  # TODO
    if self.t_t <= 0.0:
      self.t_t = 0.5 * self.n_t * tB
    self.t_dwell = self.n_t * tB - self.t_t  # time we have from the catch to the throw
    self.t_hand = self.h.Th * tB

    c_delta_x, c_delta_y = self.P[:2] - self.ct_c.P[:2]
    theta = math.atan2(c_delta_y, c_delta_x)
    self.p_t[0] = self.P[0] - math.cos(theta)*swingSize
    self.p_t[1] = self.P[1] - math.sin(theta)*swingSize
    self.p_t[2] = self.P[2]

    self.v_t[0] = (self.ct_c.P[0] - self.p_t[0]) / self.t_t
    self.v_t[1] = (self.ct_c.P[1] - self.p_t[1]) / self.t_t
    self.v_t[2] = 0.5 * g * self.t_t

  def initTraj(self):
    tt = [0.0, self.t_dwell, self.t_hand]
    xx = [[self.P[0], self.p_t[0], self.P[0]],
          [self.P[1], self.p_t[1], self.P[1]],
          [self.P[2], self.p_t[2], self.P[2]]]
    vv = [[self.ct_c.v_t[0], self.v_t[0], self.ct_c.v_t[0]],
          [self.ct_c.v_t[1], self.v_t[1], self.ct_c.v_t[1]],
          [-self.ct_c.v_t[2], self.v_t[2], -self.ct_c.v_t[2]],
          ]
    self.traj = MinJerkTraj(dt, tt, xx ,vv)
    self.ballTraj = BallTraj(self.t_t, self.p_t, self.v_t)

  def plotTimeDiagram(self, ax, period=0):
    actual_beat = self.beat_nr + period*self.T
    x , y = actual_beat, self.h.h
    # Plot point and annonate
    ax.scatter(x, y)
    ax.annotate(str((self.n_c, self.n_t)), (x, y), (x-0.2, y-0.5))
    # Plot catch and throw at this catch-throw point
    plotConnection(ax, actual_beat-self.n_c, actual_beat, self.h_c.h, self.h.h, self.h_c.h==self.h.h)
    plotConnection(ax, actual_beat, actual_beat+self.n_t, self.h.h, self.h_t.h, self.h.h==self.h_t.h)

  def plotTrajectories(self, ax, period=0):
    ax.scatter(*self.p_t, color='k')
    ax.scatter(*self.P, color='k')
    ax.text(self.p_t[0], self.p_t[1], self.p_t[2], 'throw', size=11, zorder=1,  color='k')
    ax.text(self.P[0], self.P[1], self.P[2], 'catch', size=11, zorder=1,  color='k')
    col = self.traj.plot(ax)
    self.ballTraj.plot(ax, col)


  @property
  def P(self):
    return self.h.position

  @property
  def T(self):
    return self.h.T

  @property
  def h_c(self):
    return self.ct_c.h

  @property
  def h_t(self):
    return self.ct_t.h

class JugglingHand():
  def __init__(self, h, N, hand_positions):
    """Represent one juggling hand. Holds trajectory information about how tha hand should move.

    Args:
        ct_period ([np.array(N, 6)]): where N is nr of ct in one period and 6 are the ct elements that describe each ct
                                        (h, beat_nr, n_c, n_t, h_c, h_t)
        hand_positions  ([np.array(nh, 3)]): the x,y,z positions for each hand
    """
    self.h = h                            # index of this hand
    self.Th = len(hand_positions)         # == nr of hands, == nr of beats from catch to catch
    self.N = N                            # nr of ct this hands perform in one period
    self.T  = self.N * self.Th            # nr of beats for this hand takes(time after which the same ct has to be performed)
    self.position = hand_positions[h]     # (x,y,z) position of this hand

    self.ct_period = [None]*N                 # list of the ct this hand performs
    self.hand_positions = hand_positions  # position of all existing hands

  def initThrows(self, tDwell, tB, swingSize):
    [ct.initThrow(tDwell, tB, swingSize) for ct in self.ct_period]
  def initTraj(self):
    [ct.initTraj() for ct in self.ct_period]
    pass

  def addCT(self, i, ct):
    self.ct_period[i] = ct

  def plotTimeDiagram(self, ax):
    # Plot hand horizontal lines
    ax.axhline(y=self.h)
    # Plot catche-throw point&trajectories
    self.ct_period[0].plotTimeDiagram(ax, 1) # 1 more than the period(first ct is included twice) for visual effects
    [ct.plotTimeDiagram(ax) for ct in self.ct_period]

  def plotHandTrajectories(self, ax):
    ax.scatter(self.position[0], self.position[1])
    [ct.plotTrajectories(ax) for ct in self.ct_period]


class JugglingPlan():
  def __init__(self, hands, hand_positions, tB, r_dwell, swing_size):
    self.Nh = len(hands)                  # nr of hands
    self.T = hands[0].T                   # period length in nr of beats( nr of beats after the first ct from the first hand gets repeated)
    self.hands = hands                    # list of hands in this plan
    self.handPositions = hand_positions   # cartesian positions of each hand in this plan
    self.swingSize = swing_size           # how much we can swing for the throw(normally 1.5*diameter_ball)

    # Beat length in seconds
    self.tB = tB  #[seconds]
    # Calculate hand-time (the time between two catches of the same hand)
    self.tH = self.Nh * tB  #[seconds]
    # Compute dwell-time: time in one hand-time where the ball is on the hand.
    self.r_dwell = r_dwell
    self.tDwell = r_dwell * self.tH  #[seconds]

  def initTrajectories(self):
    [h.initThrows(self.tDwell, self.tB, self.swingSize) for h in self.hands]
    [h.initTraj() for h in self.hands]

  def plotHandTrajectories(self):
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    # Plot hands
    [h.plotHandTrajectories(ax) for h in self.hands]

    # [h.plot(ax) for h in self.hands]
    # ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required
    ax.grid()
    plt.show()

  def plotTimeDiagram(self):
    fig, ax = plt.subplots(1, 1)
    N_beats = self.T
    heights = np.arange(self.Nh)

    [h.plotTimeDiagram(ax) for h in self.hands]

    ax.grid(axis='x')
    ax.set_xticks(np.arange(0, N_beats+1, 1))
    ax.set_yticks(heights)
    ax.set_yticklabels([r"H$_{}$".format(n) for n in range(self.Nh)])
    ax.set_xlim((-0.2, N_beats-0.2))
    ax.set_ylim((heights[0]-0.5, heights[-1]+0.5))
    plt.show()


class JugglingPlanner():
  def __init__(self, h=0.3, w=0.4, r_dwell=0.5, D=0.07):
    """[summary]

    Args:
        nh ([int]): number of hands
        h ([float]): height of the standard 3-throw
        w ([float]): width of the hand positions
        r_dwell ([float]): ratio of time the ball stays in hand and time between two catches of the same hand
        D ([float]): diameter of the ball

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
    # tf3 = 0.5*(3*tH - 2*r_dwell*tH) = (3*tB - 2*r_dwell*tB) = tB(3 - 2*r_dwell)
    # where 2*r_dwell*tB is the dwell time with which we have to shorten the flight time.
    self.r_dwell = r_dwell
    tf3 = math.sqrt(2.0*h/g)  # time of flight for a 3-throw
    self.tB = tf3 / (3.0 - 2.0*self.r_dwell)
    self.swing_size = D*1.5

  def plan(self, nh, pattern=(3,), rep=1):
    """Plan the juggling trajectories for the hand

    Args:
        pattern (tuple, optional): The siteswap pattern to be juggled. Defaults to (3,).
    """
    pattern = tuple(pattern)

    # Check if pattern is jugglable
    assert self.isJugglable(pattern), 'The given pattern {} is not jugglable'.format(pattern)

    # Get catch-throw sequence
    ct_time_sequence = self.getCatchThrowSequence(nh, pattern) * rep

    # Computes the ct period-sequence for each hand. Returns matrix (T x nr hands x 6),
    # where T is the periode and each element, matrix(t, h) describes a ct: (h, beat_nr, n_c, n_t, h_c, h_t))
    # Here each ct knows the hands the ball is coming from and which hand the ball is going to.
    ct_period_per_hand = self.getCatchThrowHandPeriod(nh, ct_time_sequence)

    # Caresian positions of the hands. Are assumed to lie on the edges of a nh-polygon with side-length = w
    hand_positions = self.getHandPositions(nh)

    # List of JugglingHand objects, one for each hand.
    hands = self.initHands(nh, ct_period_per_hand, hand_positions)

    # The juggling plan which is intended for further usage(check its API)
    juggPlan = JugglingPlan(hands, hand_positions, self.tB, self.r_dwell, self.swing_size)
    juggPlan.initTrajectories()

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

  def getCatchThrowSequence(self, nh, pattern):
    # return: ct_time_seq: ct_time_seq[i] = (c_i, t_i),  saves the catch throw happening at the time beat i.
    # (length of throw we are catching in nr of beats, length of throw in nr of beats)
    N = len(pattern)
    ct_time_seq = [None]*N
    for i, n_c in enumerate(pattern):
      # at what index  n_c must be catched if it is thrown at index i
      catch_index = (n_c+i)%N
      # pattern[catch_index]: what is thrown at the timebeat where t is catched
      ct_time_seq[catch_index] = (n_c, pattern[catch_index])
    return ct_time_seq

  def getCatchThrowHandPeriod(self, nh, ct_time_seq):
    """Computs the smallest sequence of ct that a hand has to perform(called period) for each hand.
       For each of these ct the information describing the ct is saved.

    Args:
        ct_time_seq ([ct]): catch-throw sequence where each element is (n_c, n_t)

    Returns:
        [np.array((M_h, nh, 3))]: holds (hand_nr, beat_nr, n_c, n_t, h_c, h_t)
    """
    N = len(ct_time_seq) # Period
    # Calculate nr of beats required for each hand to perform perform all c-t it will ever face at least once(least common multiple)
    M = lcm(N, nh)
    # Nr of ct a hand performs in one period
    M_h = M//nh
    # Fill a M_h x nh matrix with (c,t) pairs
    ct_period_per_hand = np.zeros((M_h, nh, 8), dtype=np.int)
    for i in range(M_h):  # i: ct index for the hand
      for h in range(nh): # h: hand index
        time_beat = h + i*nh
        k = time_beat % N  # k: index for ct_time_seq
        # the rule is to always throw to higher hands(modulo)
        # and to be thrown to from lower hands(modulo)
        catch_from = (h-ct_time_seq[k][0]) % nh
        throw_to = (h+ct_time_seq[k][1]) % nh

        # compute beat_nr for the cts this ct is connected to
        catch_throw_beat = (time_beat - ct_time_seq[k][0]) % M
        throw_catch_beat = (time_beat + ct_time_seq[k][1]) % M

        # compute indexes of cts this ct is connected to.
        # to be used in the lists correspondinh hands,
        # i.e. ct_catch = ct_period_per_hand[catch_ct_index, catch_from]
        catch_ct_index = (catch_throw_beat - catch_from) // nh
        throw_ct_index = (throw_catch_beat - throw_to) // nh
                                                    # (n_c, n_t)
        ct_period_per_hand[i,h] = (h, time_beat,) + ct_time_seq[k] + (catch_from, throw_to) + (catch_ct_index, throw_ct_index)
    return ct_period_per_hand

  def initHands(self, nh, ct_period_per_hand, hand_positions):
    N = ct_period_per_hand.shape[0] # nr of ct a hand has to perform before starting to repeat
    # initialize hands
    hands = [JugglingHand(h, N, hand_positions=hand_positions) for h in range(nh)]
    # Initialize cts
    cts = [ [ CatchThrow(hands[h], *ct[1:4]) for ct in ct_period_per_hand[:, h] ] for h in range(nh) ]
    # Add CTs to each hand
    for h in range(nh):
      for i, ct in enumerate(ct_period_per_hand[:, h]):
        # add catch_ct and throw_ct to each ct
        h_c, h_t, ct_c_i, ct_t_i = ct[-4:]
        cts[h][i].addCatchThrowCTs(cts[h_c][ct_c_i], cts[h_t][ct_t_i])
        # add ct to corresponding hand
        hands[h].addCT(i, cts[h][i])
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
  p = jp.plan(4, pattern=(7,), rep=2)
  # p.plotTimeDiagram()
  # p.plotHandTrajectories()

  # jp.plan(3, pattern=(4,4,4,4)).plotTimeDiagram()
  p = jp.plan(2, pattern=(3, 3, 4, 2, 3, 3, 4, 2))
  # p.plotTimeDiagram()
  p.plotHandTrajectories()