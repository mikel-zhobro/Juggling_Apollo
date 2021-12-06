import numpy as np
import matplotlib.pyplot as plt
import math

from settings import g, dt  # globals
import MinJerk


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


class Traj():
  def __init__(self):
    self.N_Whole = None                     # nr of timesteps in trajectory
    self.ttt = None                         # timesteps in trajectory
    self.xxx = None                         # position trajectory
    self.vvv = None                         # velocity trajectory
    self.aaa = None                         # acceleration trajectory
    self.jjj = None                         # jerk trajectory
    self.thetas = None

  def init_traj(self, ttt, xxx, vvv, aaa, jjj, set_thetas=False):
    # self.N_Whole = utils.steps_from_time(ttt.size, dt)
    self.N_Whole = ttt.size
    self.ttt = ttt
    self.xxx = xxx
    self.vvv = vvv
    self.aaa = aaa
    self.jjj = jjj
    if set_thetas:
      # orientation is in the direction of the acceleration
      # only that we take the absolute value of the z-direction so the ball doesn't fall.
      self.thetas = self.vvv.copy() / np.linalg.norm(self.vvv, axis=1, keepdims=True)
      self.thetas[:,2] = abs(self.thetas[:,2])

  def get(self, get_thetas=False):
    return (self.N_Whole, self.xxx, self.vvv, self.aaa, self.jjj) + ((self.thetas,) if get_thetas else ())


class MinJerkTraj(Traj):
  def __init__(self, tt, xx, vv):
    """Computes MinJerk trajectory in 3D cartesian coordinates and saves this information.

    Args:
        tt ([type]): time points for mj, of length 3 ( catch, throw, catch2)
        xx ([type]): positions mj trajectory should go through at the given time points
        vv ([type]): velocities mj trajectory should have at the given time points
    """
    Traj.__init__(self)

    self.tt = tt
    self.xx = np.array(xx).T
    self.vv = np.array(vv).T

    self.catch_throw = MinJerk.get_min_jerk_xyz(dt, tt[0], tt[1],
                                                self.xx[0], self.xx[1],
                                                self.vv[0], self.vv[1], lambdas=True)
    self.throw_catch2 = MinJerk.get_min_jerk_xyz(dt, tt[1], tt[2],
                                                 self.xx[1], self.xx[2],
                                                 self.vv[1], self.vv[2], lambdas=True)

  def initTraj(self, ttt):
    self.init_traj(*self._getTraj(ttt), set_thetas=True)
    return self.get()

  def _getTraj(self, t):
    # Mask out only concerning time steps
    ttt = t[(self.tt[0] <= t) &  (t <= self.tt[2])]
    mask = ttt <= self.tt[1]
    # MJ traj for the throw part
    x0, v0, a0, j0 = self.catch_throw(ttt[mask])
    # MJ traj for the catch part
    x1, v1, a1, j1 = self.throw_catch2(ttt[np.invert(mask)])
    return ttt, np.vstack((x0, x1)), np.vstack((v0, v1)), np.vstack((a0, a1)), np.vstack((j0, j1))

  def plot(self, ax, ttt, i, h_i):
    # Plot path
    a = ax.plot3D(self.xxx[:,0], self.xxx[:,1], self.xxx[:,2], label='{}_{}'.format(h_i, i))

    # Plot arrows
    # ts = [self.N_Whole//4, 3*self.N_Whole//4]
    ts = [0, self.N_Whole//2, -1]
    ax.quiver(self.xxx[ts,0], self.xxx[ts,1], self.xxx[ts,2],
              self.vvv[ts,0], self.vvv[ts,1], self.vvv[ts,2],
              length=0.07, normalize=True, color=a[0].get_color())
    tmp = 2
    ax.quiver(self.xxx[0::tmp,0], self.xxx[0::tmp,1], self.xxx[0::tmp,2],
              self.thetas[::tmp,0], self.thetas[::tmp,1], abs(self.thetas[::tmp,2]),
              length=0.07, normalize=True, color='k', alpha=0.2)
    return a[0].get_color()


class BallTraj(Traj):
  def __init__(self, t_t, p_t, v_t):
    """Creates the ideal ball trajectory for the given fly_time throw velocity and throw position.

    Args:
        t_t (float): fly time of the ball
        p_t (np.array(3,)): throw position
        v_t (np.array(3,)): throw velocity
    """
    Traj.__init__(self)
    self.t_t, self.p_t, self.v_t = t_t, p_t, v_t
    self.ttt = np.linspace(0, self.t_t, 50).reshape(-1,1)
    self.aaa = np.zeros((50, 3)); self.aaa[:,-1] = -g
    self.vvv = self.v_t + self.aaa*self.ttt
    self.xxx = p_t + self.ttt*v_t + 0.5*self.aaa*self.ttt**2

  def plot(self, ax, col):
    # plot path
    ax.plot3D(self.xxx[:,0], self.xxx[:,1], self.xxx[:,2], color=col, linestyle='--')

    # plot arrows
    ts = [0, 25, -1]
    ax.quiver(self.xxx[ts,0], self.xxx[ts,1], self.xxx[ts,2],  # position of arrow
              self.vvv[ts,0], self.vvv[ts,1], self.vvv[ts,2],  # direction of arrow
              length=0.08, normalize=True, color=col)


class CatchThrow():
  def __init__(self, i, h, beat_nr, n_c, n_t):
    # We catch at self.h.position.
    # We throw 'swingSize' in the direction of the ct the ball goes to.
    self.i = i                  # order index of this ct in the corresponding hand
    self.h = h                  # hand this ct belongs to
    self.beat_nr = beat_nr      # beat nr during which this ct is performed
    self.n_c = n_c              # length of throw we are catching in nr of beats
    self.n_t = n_t              # length of throw we are throwing in nr of beats
    self.ct_c = None            # ct where catch comes from
    self.ct_t = None            # ct where throw goes to

    self.t_catch = None         # time we catch at this ct(corresponds to the beatnr)
    self.t_throw = None         # time of throw (t_throw = t_catch + t_dwell)
    self.t_catch2 = None        # time of the next catch ( t_catch2 = t_catch + t_hand)

    self.t_fly = None           # flight time of the ball we throw
    self.p_t = np.zeros(3)      # (x, y, z) position where we throw
    self.v_t = np.zeros(3)      # (vx, vy, vz) velocity we throw the ball with

    self.traj = None            # the catch-throw-catch trajectory
    self.balltraj = None        # ball trajectory, only for visualisation

  def addCatchThrowCTs(self, ct_c, ct_t):
    self.ct_c = ct_c            # ct where catch comes from
    self.ct_t = ct_t            # ct where throw goes to

  def initThrow(self, t_dwell, tB, swingSize):
    self.tB = tB
    # we assume catch and throw points lie all at z=0
    self.t_fly = self.n_t * tB - t_dwell
    if self.t_fly <= 0.0:
      print('Warning: This throw requires more time beats than we have if we take away the dwell time.')
      print('that is why we are we split the hand-time equally between both.')
      self.t_fly = 0.5 * self.n_t * tB
    t_dwell = self.n_t * tB - self.t_fly  # time we have from the catch to the throw
    t_hand = self.h.Th * tB  # time we have from the catch to the next catch

    self.t_catch = self.beat_nr * tB
    self.t_throw = self.t_catch + t_dwell
    self.t_catch2 = self.t_catch + t_hand

    # throw position is swingSize in the direction of the hand the ball goes to
    c_delta_x, c_delta_y = self.ct_c.P[:2] - self.P[:2]
    theta = math.atan2(c_delta_y, c_delta_x)
    self.p_t[0] = self.P[0] + math.cos(theta)*swingSize
    self.p_t[1] = self.P[1] + math.sin(theta)*swingSize
    self.p_t[2] = self.P[2]

    # throw velocity
    self.v_t[0] = (self.ct_t.P[0] - self.p_t[0]) / self.t_fly
    self.v_t[1] = (self.ct_t.P[1] - self.p_t[1]) / self.t_fly
    self.v_t[2] = 0.5 * g * self.t_fly

  def initTraj(self, ttt, ct_next):
    tt = [self.t_catch, self.t_throw, self.t_catch2]
    xx = [[self.P[0], self.p_t[0], ct_next.P[0]],
          [self.P[1], self.p_t[1], ct_next.P[1]],
          [self.P[2], self.p_t[2], ct_next.P[2]]]
    vv = [[ self.ct_c.v_t[0], self.v_t[0],  ct_next.ct_c.v_t[0]],
          [ self.ct_c.v_t[1], self.v_t[1],  ct_next.ct_c.v_t[1]],
          [-self.ct_c.v_t[2], self.v_t[2], -ct_next.ct_c.v_t[2]],
          ]
    self.ballTraj = BallTraj(self.t_fly, self.p_t, self.v_t)
    self.traj = MinJerkTraj(tt, xx ,vv)
    return self.traj.initTraj(ttt)

  def plotTimeDiagram(self, ax, period=0):
    actual_beat = self.beat_nr + period*self.T
    x , y = actual_beat, self.h.h
    # Plot point and annonate
    ax.scatter(x, y)
    ax.annotate(str((self.n_c, self.n_t)), (x, y), (x-0.2, y-0.5))
    # Plot catch and throw at this catch-throw point
    plotConnection(ax, actual_beat-self.n_c, actual_beat, self.h_c.h, self.h.h, self.h_c.h==self.h.h)
    plotConnection(ax, actual_beat, actual_beat+self.n_t, self.h.h, self.h_t.h, self.h.h==self.h_t.h)

  def plotTrajectories(self, ax, ttt, h_i, period=0):
    ax.scatter(*self.p_t, color='k')
    ax.scatter(*self.P, color='k')
    ax.text(self.p_t[0], self.p_t[1], self.p_t[2], 'throw', size=11, zorder=1,  color='k')
    ax.text(self.P[0], self.P[1], self.P[2], 'catch', size=11, zorder=1,  color='k')
    col = self.traj.plot(ax, ttt, self.i, h_i)
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


class JugglingHand(Traj):
  def __init__(self, h, N, hand_positions):
    """Represent one juggling hand. Holds trajectory information about how tha hand should move.

    Args:
        ct_period ([np.array(N, 6)]): where N is nr of ct in one period and 6 are the ct elements that describe each ct
                                        (h, beat_nr, n_c, n_t, h_c, h_t)
        hand_positions  ([np.array(nh, 3)]): the x,y,z positions for each hand
    """
    Traj.__init__(self)
    self.h = h                            # index of this hand
    self.Th = len(hand_positions)         # == nr of hands, == nr of beats from catch to catch
    self.N = N                            # nr of ct this hands perform in one period
    self.T = (self.N) * self.Th           # nr of beats for this hand takes(time after which the same ct has to be performed)
    self.position = hand_positions[h]     # (x,y,z) position of this hand

    self.ct_period = [None]*N             # list of the ct this hand performs
    self.hand_positions = hand_positions  # position of all existing hands

  def initThrows(self, tDwell, tB, swingSize):
    # self.ttt = np.arange(0.0, self.Th*tB, dt)
    self.ttt = np.linspace(self.h*tB, self.T*tB+self.h*tB, self.T*tB//dt)
    self.N_Whole = self.ttt.size
    [ct.initThrow(tDwell, tB, swingSize) for ct in self.ct_period]

  def getTraj(self):
    return self.ttt,

  def initTraj(self):
    xxx = np.zeros((self.N_Whole, 3))
    vvv = np.zeros((self.N_Whole, 3))
    aaa = np.zeros((self.N_Whole, 3))
    jjj = np.zeros((self.N_Whole, 3))
    N0 = 0
    for i, ct in enumerate(self.ct_period):
     N, xx, vv, aa, jj = ct.initTraj(self.ttt, self.ct_period[(i+1)%self.N])
     xxx[N0:N0+N]  = xx
     vvv[N0:N0+N]  = vv
     aaa[N0:N0+N]  = aa
     jjj[N0:N0+N]  = jj

     N0 += N
    self.init_traj(self.ttt, xxx, vvv, aaa, jjj)

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
    for ct in self.ct_period:
      ct.plotTrajectories(ax, self.ttt, self.h)


class JugglingPlan():
  def __init__(self, pattern, hands, hand_positions, tB, r_dwell, swing_size):
    self.pattern = pattern
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

  def plotHandTrajectories(self, subplot=111):
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
    # fig = plt.figure()
    ax = plt.subplot(subplot, projection='3d')
    # Plot hands
    [h.plotHandTrajectories(ax) for h in self.hands]

    set_axes_equal(ax) # IMPORTANT - this is also required
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.grid()
    ax.legend()
    ax.set_title('Cartesian Trajectories')
    if subplot==111:
      plt.show()

  def plotTimeDiagram(self, subplot=111):
    ax = plt.subplot(subplot)
    N_beats = self.T
    heights = np.arange(self.Nh)

    [h.plotTimeDiagram(ax) for h in self.hands]

    ax.grid(axis='x')
    ax.set_xticks(np.arange(0, N_beats+1, 1))
    ax.set_yticks(heights)
    ax.set_yticklabels([r"H$_{}$".format(n) for n in range(self.Nh)])
    ax.set_xlim((-0.2, N_beats-0.2))
    ax.set_ylim((heights[0]-0.5, heights[-1]+0.5))
    ax.set_title('Time Diagram')
    if subplot==111:
      plt.show()

  def plot(self):
    fig = plt.figure(figsize=(13,6))
    self.plotHandTrajectories(122)
    self.plotTimeDiagram(323)
    fig.tight_layout()
    plt.suptitle('{} hands plan for pattern: {}'.format(self.Nh, self.pattern))
    plt.show()


class JugglingPlanner():
  def __init__(self, h=0.3, w=0.3, r_dwell=0.5, D=0.07):
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
    tf3 = 2.0*math.sqrt(2.0*h/g)  # time of flight for a 3-throw
    self.tB = tf3 / (3.0 - 2.0*self.r_dwell)
    self.swing_size = D*2.5

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

    # List of JugglingHand objects, one for each hand
    hands = self.initHands(nh, ct_period_per_hand, hand_positions)

    # The juggling plan which is intended for further usage(check its API)
    juggPlan = JugglingPlan(pattern, hands, hand_positions, self.tB, self.r_dwell, self.swing_size)
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
    cts = [ [ CatchThrow(i, hands[h], *ct[1:4]) for i, ct in enumerate(ct_period_per_hand[:, h]) ] for h in range(nh) ]
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
  jp.plan(1, pattern=(3,3,3), rep=1).plot()
  jp.plan(2, pattern=(3,), rep=1).plot()
  p = jp.plan(3, pattern=(4,), rep=1)
  N_Whole0, x0, v0, a0, j0, thetas = p.hands[0].get(True)  # get plan for hand0
  p.plot()
