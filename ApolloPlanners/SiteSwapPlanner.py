#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   SiteSwapPlanner.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   Defines a juggling pattern parser for arbitrary patterns and hands.
             The parsing is organized hierarchichally where the plan consists of a sequence
             of catch-throws which are then organized in hand planes.
             We use Minimum Jerk approach for the default interpolation between catch-throws.
'''

import numpy as np
import matplotlib.pyplot as plt
import math

import utils
import MinJerk


def rollavg_cumsum_edges(a,n):
  """ Used to smooth out trajectories.
      Similar to the numpy.cumsum but with edge handling.
      Simple sliding-window spatial filter that replaces the center value in the window
      with the average (mean) of all the values in the window.

  Args:
      a (np.array(N, M)): Trajectory to be smoothed where N is the length and M the dimension of each element.
      n (float):          Size of the smooth window

  Returns:
      (np.array(N, M)): Smoothed out trajectory
  """
  n = n if n%2==1 else n+1
  N = len(a)
  cumsum_vec = np.cumsum(np.insert(np.pad(a,((n-1,n-1),(0,0)),'wrap'), 0, 0, 0), axis=0)
  d = np.hstack((np.arange(n//2+1,n),np.ones(N-n)*n,np.arange(n,n//2,-1))).reshape(-1,1)
  return (cumsum_vec[n+n//2:-n//2+1] - cumsum_vec[n//2:-n-n//2]) / d

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

def plotConnection(ax, actual_beat, catch_beat, y0, y1, same_hand, color='k'):
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
  ax.plot(x, y, color=color, linewidth=1)


class BallTraj():
  def __init__(self, t_t, p_t, v_t):
    """Creates the ideal ball trajectory for the given fly_time throw velocity and throw position.
       It uses a simple model without air drag or friction.

    Args:
        t_t (float):        fly time of the ball
        p_t (np.array(3,)): throw position
        v_t (np.array(3,)): throw velocity
    """
    self.t_t, self.p_t, self.v_t = t_t, p_t, v_t
    self.ttt = np.linspace(0, self.t_t, 50).reshape(-1,1)
    self.aaa = np.zeros((50, 3)); self.aaa[:,-1] = -utils.g
    self.vvv = self.v_t + self.aaa*self.ttt
    self.xxx = p_t + self.ttt*v_t + 0.5*self.aaa*self.ttt**2

  def plotBallTraj(self, ax, col):
    # plot path
    ax.plot3D(self.xxx[:,0], self.xxx[:,1], self.xxx[:,2], color=col, linestyle='--')

    # plot arrows
    ts = [0, 25, -1]
    ax.quiver(self.xxx[ts,0], self.xxx[ts,1], self.xxx[ts,2],  # position of arrow
              self.vvv[ts,0], self.vvv[ts,1], self.vvv[ts,2],  # direction of arrow
              length=0.08, normalize=True, color=col)


class Traj():
  def __init__(self):
    """ This represent the base class for interpolated trajectories.
    """
    self.N_Whole = None                     # nr of timesteps in trajectory
    self.ttt = None                         # timesteps in trajectory
    self.xxx = None                         # position trajectory
    self.vvv = None                         # velocity trajectory
    self.aaa = None                         # acceleration trajectory
    self.jjj = None                         # jerk trajectory
    self.thetas = None

  def init_traj(self, ttt, xxx, vvv, aaa, jjj, set_thetas=False, v_throw=None):
    """ Initializes the trajectory.

    Args:
        ttt (np.array(N,1)): timesteps in trajectory
        xxx (np.array(N,3)): position trajectory
        vvv (np.array(N,3)): velocity trajectory
        aaa (np.array(N,3)): acceleration trajectory
        jjj (np.array(N,3)): jerk trajectory
        set_thetas (bool): Whether to initialize the orientation trajectory.
                           Since the trajectory lies in a 2D plane, we can
                           define the orientation by setting the z-axis corresponding
                           to the tangential of the position trajectory. The y-axis should then be set
                           so that it corresponds to the normal of the position trajectory and the x axis
                           to the cross-product of the above axis.
                           More precisely, for the z-axis we use the velocity trajectory with the z-part forced
                           to be positive(so that the hand doesn't show downwards.)
                           The y-axis (the normal to the position) can then be calculated as the cross-product
                           of the tangential and the acceleration trajectory.
                           In order end up with smooth orientation trajectory use smooth out the velocity
                           and acceleration trajecrories with a simple mean-filter defined in the function 'rollavg_cumsum_edges'.
        v_throw (np.array(3,1)): In case the throw-velocity vector is set, we initialize the orientation trajectory
                                 we define the z-axis as a weighted average between this vector and the velocity trajectory
                                 mentioned before. This helps to further reduce changes in the orientation trajectory.
    """
    self.N_Whole = ttt.size
    self.ttt = ttt
    self.xxx = xxx.squeeze()
    self.vvv = vvv.squeeze()
    self.aaa = aaa.squeeze()
    self.jjj = jjj.squeeze()
    if set_thetas:
      # Create the axis
      N_smooth_window = len(self.vvv)//2
      vvv_smoothed =rollavg_cumsum_edges(self.vvv, N_smooth_window)
      aaa_smoothed =rollavg_cumsum_edges(self.aaa, N_smooth_window)

      tmp = 0.7
      _z = vvv_smoothed / np.linalg.norm(vvv_smoothed, axis=1, keepdims=True)
      _z[:,2] = abs(_z[:,2])
      if v_throw is not None:
        _z = (1-tmp)*_z + tmp*v_throw
      else:
        _z[:,2] = np.clip(_z[:,2], 0.8, np.inf)
      _z /= np.linalg.norm(_z, axis=1, keepdims=True)

      _y = np.cross(vvv_smoothed, aaa_smoothed)
      _y /= np.linalg.norm(_y, axis=1, keepdims=True)
      _x = np.cross(_y, _z)

      # Fill the rotation trajectory
      self.thetas = np.zeros((xxx.shape[0],3,3))
      self.thetas[:,:,0] = _x
      self.thetas[:,:,1] = _y
      self.thetas[:,:,2] = _z

  def get(self, get_thetas=False):
    """ Returns the trajectory.
    """
    return (self.N_Whole, self.xxx, self.vvv, self.aaa, self.jjj) + ((self.thetas,) if get_thetas else ())


class MinJerkTraj(Traj):
  def __init__(self, dt, tt, xx, vv):
    """Computes MinJerk trajectory in 3D cartesian coordinates and saves this information.

    Args:
        tt ([type]): time points for mj, of length 3 ( catch, throw, catch2)
        xx ([3, 3]): positions mj trajectory should go through at the given time points (where catch, throw, catch2 are rows)
        vv ([type]): velocities mj trajectory should have at the given time points (velocities at each point are the rows)
    """
    Traj.__init__(self)

    self.tt = tt
    self.xx = xx
    self.vv = vv
    self.ctc_traj_lambda = MinJerk.get_multi_interval_multi_dim_minjerk(dt, tt, self.xx, self.vv, smooth_acc=True, i_a_end=0, lambdas=True)


  def initMJTraj(self, ttt, last=False):
    # Mask out only concerning time steps
    ttt_ = ttt[(self.tt[0] <= ttt) & ( (ttt < self.tt[2]) if last else (ttt <= self.tt[2]))]
    x, v, a, j = self.ctc_traj_lambda(ttt_)
    self.init_traj(ttt_, x, v, a, j, set_thetas=True, v_throw=self.vv[1])

    return self.get()

  def plotMJTraj(self, ax, ttt, i, h_i, orientation):
    # Plot path
    a = ax.plot3D(self.xxx[:,0], self.xxx[:,1], self.xxx[:,2], label='{}_{}'.format(h_i, i))

    # Plot arrows
    # ts = [self.N_Whole//4, 3*self.N_Whole//4]
    ts = [0, self.N_Whole//2, -1]
    ax.quiver(self.xxx[ts,0], self.xxx[ts,1], self.xxx[ts,2],
              self.vvv[ts,0], self.vvv[ts,1], self.vvv[ts,2],
              length=0.07, normalize=True, color=a[0].get_color())

    if orientation:
      tmp = 9
      ax.quiver(self.xxx[0::tmp,0], self.xxx[0::tmp,1], self.xxx[0::tmp,2],
                -self.thetas[::tmp,0,0], -self.thetas[::tmp,1,0], -self.thetas[::tmp,2,0],
                length=0.07, normalize=True, color='r', alpha=0.2, label='_'*int(bool(i or h_i)) + '-x direction')

      ax.quiver(self.xxx[0::tmp,0], self.xxx[0::tmp,1], self.xxx[0::tmp,2],
                self.thetas[::tmp,0,1], self.thetas[::tmp,1,1], self.thetas[::tmp,2,1],
                length=0.07, normalize=True, color='g', alpha=0.2, label='_'*int(bool(i or h_i)) + 'y direction')
      ax.quiver(self.xxx[0::tmp,0], self.xxx[0::tmp,1], self.xxx[0::tmp,2],
                self.thetas[::tmp,0,2], self.thetas[::tmp,1,2], self.thetas[::tmp,2,2],
                length=0.07, normalize=True, color='b', alpha=0.2, label='_'*int(bool(i or h_i)) + 'z direction')

    print('a',np.max(self.aaa, axis=0))
    print('v',np.max(self.vvv, axis=0))
    return a[0].get_color()


class CatchThrow():
  def __init__(self, i, h, beat_nr, n_c, n_t):
    """ Initializes a catch throw.

    Args:
        i (int): order index of this ct in the corresponding hand
        h (int): hand this ct belongs to
        beat_nr (int): beat nr during which this ct is performed
        n_c (int): length of throw we are catching in nr of beats
        n_t (int): length of throw we are throwing in nr of beats
    """
    # We catch at self.h.position.
    # We throw 'swingSize' in the direction of the ct the ball goes to.
    self.i = i                  # order index of this ct in the corresponding hand
    self.h = h                  # hand this ct belongs to
    self.beat_nr = beat_nr      # beat nr during which this ct is performed
    self.n_c = n_c              # length of throw we are catching in nr of beats
    self.n_t = n_t              # length of throw we are throwing in nr of beats
    self.ct_c = None            # ct where catch comes from
    self.ct_t = None            # ct where throw goes to
    self.ct_next = None         # ct where next catch takes place

    self.t_catch = None         # time we catch at this ct(corresponds to the beatnr)
    self.t_throw = None         # time of throw (t_throw = t_catch + t_dwell)
    self.t_catch2 = None        # time of the next catch ( t_catch2 = t_catch + t_hand)

    self.t_fly = None           # flight time of the ball we throw
    self.p_t = np.zeros(3)      # (x, y, z) position where we throw
    self.v_t = np.zeros(3)      # (vx, vy, vz) velocity we throw the ball with

    self.traj = None            # the catch-throw-catch trajectory
    self.ballTraj = None        # ball trajectory, only for visualisation

  def addCatchThrowNextCTs(self, ct_c, ct_t, ct_next):
    """ Add cts where we catch from, throw to and where the next catch takes place.

    Args:
        ct_c (CatchThrow): ct where catch comes from
        ct_t (CatchThrow): ct where throw goes to
        ct_next (CatchThrow): ct where the next catch takes place
    """
    self.ct_c    = ct_c            # ct where catch comes from
    self.ct_t    = ct_t            # ct where throw goes to
    self.ct_next = ct_next         # ct where the next catch takes place

  def initCTThrow(self, t_dwell, tB, swingSize, throw_height):
    """ Initializes physical parameters of the catch-throw,
        so that it satisfyies the laws of physic.

        After we catch the ball we have t_dwell time to throw again.
        To throw we can swing for a distance of 'swingSize' in the direction
        of the CT we want to throw to.
        According to the desired flight time(defined by the throw type, tells how high the ball flies)
        and receiving CT we can compute the throw velocity.

    Args:
        t_dwell (float): dwell time
        tB (float): length in seconds of a beat
        swingSize (float): the size we are allowed to swing for
        throw_height (float): the height the ball should be thrown from
    """
    self.tB = tB
    # we assume catch and throw points lie all at z=throw_height
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
    if self.ct_c != self:
      theta = math.atan2(c_delta_y, c_delta_x)
      self.p_t[0] = self.P[0] + math.cos(theta)*swingSize
      self.p_t[1] = self.P[1] + math.sin(theta)*swingSize
    else: # when juggling to the same hand we choose to throw in -x direction
      self.p_t[0] = self.P[0] - swingSize
      self.p_t[1] = self.P[1]
    self.p_t[2] = self.P[2] + throw_height
    # throw velocity
    self.v_t[0] = (self.ct_t.P[0] - self.p_t[0]) / self.t_fly
    self.v_t[1] = (self.ct_t.P[1] - self.p_t[1]) / self.t_fly
    self.v_t[2] = 0.5 * utils.g * self.t_fly  +  (self.ct_t.P[2]- throw_height)/self.t_fly

  def initCTTraj(self, dt, ttt, last):
    """ Here we define the edge condition that the hand should satisfy for this CT.
        These consist of time, position and velocity requirements for the catching, throwing and catching again.
        This conditions allow us to compute the ball trajectory and interpolate the hand trajectory with Min Jerk.

    Args:
        dt (float): time steop
        ttt (np.array(N,1)): time points along the hand trajectory for this CT used to evaluate
                             the functions returned by the Min Jerk module.
        last (bool): whether this is the last CT in the hand trajectory.

    Returns:
        _type_: _description_
    """
    # Init plan points
    rat = 0.0
    tt = np.array([self.t_catch,       self.t_throw,      self.t_catch2])
    xx = np.array([[self.P[0],         self.P[1],         self.P[2]],          # catch position
                   [self.p_t[0],       self.p_t[1],       self.p_t[2]],        # throw position
                   [self.ct_next.P[0], self.ct_next.P[1], self.ct_next.P[2]]]) # next ct of this hand

    vv = np.array([[self.ct_c.v_t[0]*rat,         self.ct_c.v_t[1]*rat,         -self.ct_c.v_t[2]*rat],
                   [self.v_t[0],                  self.v_t[1],                   self.v_t[2]],
                   [self.ct_next.ct_c.v_t[0]*rat, self.ct_next.ct_c.v_t[1]*rat, -self.ct_next.ct_c.v_t[2]*rat]])

    self.ballTraj = BallTraj(self.t_fly, self.p_t, self.v_t)
    self.traj = MinJerkTraj(dt, tt, xx, vv)
    return self.traj.initMJTraj(ttt, last=last)

  # plotting methods and helper functions

  def plotCTTimeDiagram(self, ax, period=0):
    actual_beat = self.beat_nr + period*self.T
    x , y = actual_beat, self.h.h
    # color = None
    color = 'b'
    # if actual_beat in [2, 5]:
    #   print(actual_beat)
    #   color = 'r'
    # elif (actual_beat) %3 ==1:
    #   color = 'g'

    # Plot point and annonate
    ax.scatter(x, y, color=color)
    # ax.annotate(str((self.n_c, self.n_t)), (x, y), (x-0.2, y-0.5))
    # Plot catch and throw at this catch-throw point
    plotConnection(ax, actual_beat-self.n_c, actual_beat, self.h_c.h, self.h.h, self.h_c.h==self.h.h, color=color)
    plotConnection(ax, actual_beat, actual_beat+self.n_t, self.h.h, self.h_t.h, self.h.h==self.h_t.h, color=color)

  def plotCTTrajectory(self, ax, ttt, h_i, orientation):
    ax.scatter(*self.p_t, color='k')
    ax.scatter(*self.P, color='k')
    ax.text(self.p_t[0], self.p_t[1], self.p_t[2], 'throw', size=6, zorder=1,  color='k')
    ax.text(self.P[0], self.P[1], self.P[2], 'catch', size=6, zorder=1,  color='k')
    col = None
    # col = self.traj.plotMJTraj(ax, ttt, self.i, h_i, orientation)
    self.ballTraj.plotBallTraj(ax, col)

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
    """Represent one juggling hand. Holds trajectory information about how the hand should move

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

  def initHandThrows(self, dt, tDwell, tB, swingSize, throw_height):
    """ Initializes the CTs included in the hand's trajectory.

    Args:
        dt (float): time step
        tDwell (float): dwell time
        tB (float): length in seconds of a beat
        swingSize (float): swing distance
        throw_height (float): the height from where the throw should be performed
    """
    # self.ttt = np.arange(0.0, self.Th*tB, dt)
    self.ttt = np.linspace(self.h*tB, self.T*tB+self.h*tB, self.T*tB//dt)
    self.N_Whole = self.ttt.size
    [ct.initCTThrow(tDwell, tB, swingSize, throw_height) for ct in self.ct_period]

  def initHandTraj(self, dt):
    """ Initializes the hand trajectory by putting together the trajectories of all CTs in the hand.

    Args:
        dt (float): time step
    """
    xxx = np.zeros((self.N_Whole, 3))
    vvv = np.zeros((self.N_Whole, 3))
    aaa = np.zeros((self.N_Whole, 3))
    jjj = np.zeros((self.N_Whole, 3))
    N0 = 0
    for i, ct in enumerate(self.ct_period):
     N, xx, vv, aa, jj = ct.initCTTraj(dt, self.ttt, last=i==self.N-1)
     xxx[N0:N0+N]  = xx
     vvv[N0:N0+N]  = vv
     aaa[N0:N0+N]  = aa
     jjj[N0:N0+N]  = jj

     N0 += N
    self.init_traj(self.ttt, xxx, vvv, aaa, jjj, set_thetas=True)

  def addCT(self, i, ct):
    """ Add i-th CT

    Args:
        i (int): index of the CT
        ct (CatchThrow): CT to be added
    """
    self.ct_period[i] = ct

  # plotting methods

  def plotHandTimeDiagram(self, ax):
    # Plot hand horizontal lines
    ax.axhline(y=self.h)
    # Plot catche-throw point&trajectories
    self.ct_period[0].plotCTTimeDiagram(ax, 1) # 1 more than the period(first ct is included twice) for visual effects
    [ct.plotCTTimeDiagram(ax) for ct in self.ct_period]

  def plotHandTrajectory(self, ax, orientation):
    ax.scatter(self.position[0], self.position[1])
    for ct in self.ct_period:
      ct.plotCTTrajectory(ax, self.ttt, self.h, orientation)

  def getTimesPositionsVelocities(self):
    """ Puts together the plan of the hand.
        It can then be used in different trajectory planning algorithms such as Minjerk in cartesian or joint space.
    Returns:
        (tts, xxs, vvs): time-points(N, ) and their corresponding positions and velocities that the hand should achieve
                         all of size (N, 3) where N is the nr of time points.
    """
    tts = np.zeros((self.N*3,  ))
    xxs = np.zeros((self.N*3, 3))
    vvs = np.zeros((self.N*3, 3))
    for i, ct in enumerate(self.ct_period):
      j = i*3
      tts[j:j+3] = ct.traj.tt
      xxs[j:j+3] = ct.traj.xx
      vvs[j:j+3] = ct.traj.vv
    return tts, xxs, vvs

class JugglingPlan():
  def __init__(self, pattern, hands, hand_positions, tB, r_dwell, swing_size):
    """Class to represent a juggling plan consisting of several hand-juggling plans.

    Args:
        pattern (tuple): Siteswap pattern
        hands (_type_): list of hands in this plan
        hand_positions (_type_): cartesian positions of each hand in this plan
        tB (float): length in seconds of a beat
        r_dwell (float): dwell ratio (r_dwell = t_dwell/t_hand).
                         For what part of the hand time we can keep the ball in the hand.
        swing_size (float): swing distance
    """
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

  def initHands(self, dt, throw_height):
    """ Initializes the hand trajectory plans

    Args:
        dt (float): time step
        throw_height (float): the height from where the throw should be performed
    """
    [h.initHandThrows(dt, self.tDwell, self.tB, self.swingSize, throw_height) for h in self.hands]
    [h.initHandTraj(dt) for h in self.hands]

  def getFlightTimes(self):
    """ Returns list of flight times for each CT in each hand.
    """
    return [[ ct.t_fly for ct in h.ct_period] for h in self.hands]

  def getBallNPlatte(self):
    """ Returns list of hand and ball trajectories for each CT in each hand.
    """
    return [[ (ct.traj, ct.ballTraj) for ct in h.ct_period] for h in self.hands]

  # plotting methods

  def plotTrajectories(self, subplot=111, orientation=False):
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
    # fig = plt.figure()
    ax = plt.subplot(subplot, projection='3d')
    # Plot hands
    [h.plotHandTrajectory(ax, orientation) for h in self.hands]

    utils.set_axes_equal(ax) # IMPORTANT - this is also required
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.grid()
    ax.legend()
    ax.set_title('Cartesian Trajectories')
    if subplot==111:
      plt.show()

  def plotTimeDiagram(self, subplot=111):
    # plt.figure(figsize=(14,8))
    ax = plt.subplot(subplot)
    N_beats = self.T
    heights = np.arange(self.Nh)

    [h.plotHandTimeDiagram(ax) for h in self.hands]

    ax.grid(axis='x')
    ax.set_xticks(np.arange(0, N_beats+1, 1))
    ax.set_yticks(heights)
    ax.set_yticklabels([r"H$_{}$".format(n) for n in range(self.Nh)])
    ax.set_xlim((-0.2, N_beats-0.2))
    ax.set_ylim((heights[0]-0.5, heights[-1]+0.5))
    ax.set_title('Time Diagram')
    # plt.savefig("44" + '.pdf', bbox_inches='tight')

    if subplot==111:
      plt.show()

  def plot(self, orientation=False):
    fig = plt.figure(figsize=(13,6))
    self.plotTrajectories(122, orientation=orientation)
    self.plotTimeDiagram(323)
    fig.tight_layout(rect=[0., 0.16, 1, 0.95])
    plt.suptitle('{} hands plan for pattern: {}'.format(self.Nh, self.pattern))
    plt.show()


class JugglingPlanner():
  def __init__(self):
    pass

  def plan(self, dt, nh, pattern=(3,), h=0.5, r_dwell=0.5, throw_height=0.2, swing_size=0.2, w=0.3, slower=1.0, rep=1):
    """ Plans the juggling trajectories for each hand encapsulated in a JugglingPlan instance.
    Args:
        dt (float): time step
        nh (int): number of hands
        pattern (tuple): The siteswap pattern to be juggled. Defaults to (3,).
        h (float): height of the standard 3-throw
        r_dwell (float): ratio of time the ball stays in hand and time between two catches of the same hand
        throw_height (float): the height from where the throws should be performed
        swingSize (float): swing distance
        w (float): Width between the hand positions.
                   We assume hands are placed on the edges of a regular polygon with all sides equal to w
        slower (float): Tells how many times slower the return-plans should run compared to the real juggling plan.
        rep (int): Nr of periods the returned paln should repeat for.

        We assume here that hands throw the balls one after the other, as such they are not allowed to throw simultaniously.
        That means that a hand can throw at most every second step. We call this time, a hand period,
        i.e. the time between two catches of the same hand if we are are performing a cascade or shower with 2+ balls and 2 hands.
        tH = 2*tB

        h----> tf3 --r_dwell --> tB, t_handperiode

        The goal of this class is to only make sure that the siteswap pattern is correct, parse it into per-hand information
        and initialize the jugglingPlan with it.

        The jugglingPlan can be then further used to compute the hand trajectories for realizing the desired pattern.
    Returns:
        JugglingPlan: encapsulates the hand trajectories for the desired siteswap juggling pattern.
    """

    dt = dt/slower


    # We calculate the beat time from the fly time of a 3 ball using 2 hands and r_dwell.
    # It is just a definition, in general the nr of hands can be different.
    # tf3 = 0.5*(3*tH - 2*r_dwell*tH) = (3*tB - 2*r_dwell*tB) = tB(3 - 2*r_dwell)
    # where 2*r_dwell*tB is the dwell time with which we have to shorten the flight time. TODO: (not doing that right now, but could do)
    nb = 3
    tf3 = 2.0*math.sqrt(nb*(h-throw_height)/utils.g)  # time of flight for a 3-throw

    nb = 3
    v = math.sqrt(2.0*(h-throw_height)*utils.g)
    tf = (v + math.sqrt(v**2 + 2.*throw_height*utils.g))/utils.g  # consider that we throw from a higher position than we catch
    tB = tf / (nb - nh*r_dwell)


    pattern = tuple(pattern)

    # Check if pattern is jugglable
    assert self.isJugglable(pattern), 'The given pattern {} is not jugglable'.format(pattern)

    # Get catch-throw sequence
    ct_time_sequence = self.getCatchThrowSequence(pattern) * rep

    # Computes the ct period-sequence for each hand. Returns matrix (T x nr hands x 8),
    # where T is the periode and each element, matrix(t, h) describes a ct: (h, beat_nr, n_c, n_t, h_c, h_t, i_c, i_t))
    # Here each ct knows the hands the ball is coming from and which hand the ball is going to.
    ct_period_per_hand = self.getCatchThrowHandPeriod(nh, ct_time_sequence)

    # Caresian positions of the hands. Are assumed to lie on the edges of a nh-polygon with side-length = w
    hand_positions = self.getHandPositions(nh, w)

    # List of JugglingHand objects, one for each hand
    hands = self.initHands(nh, ct_period_per_hand, hand_positions)

    # The juggling plan which is intended for further usage(check its API)
    juggPlan = JugglingPlan(pattern, hands, hand_positions, tB, r_dwell, swing_size)
    juggPlan.initHands(dt, throw_height=throw_height)

    return juggPlan

  def isJugglable(self, pattern):
    """ Checks whether the siteswap pattern is juggable

    Args:
        pattern (tuple): Siteswap pattern
    """
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

  def getCatchThrowSequence(self, pattern):
    """ Computes the catch-throw numbers for each beat
    Args:
        pattern (tuple): The siteswap pattern to be juggled.

    Returns:
        ct_time_seq (list): ct_time_seq[i] = (c_i, t_i), saves the catch-throw number happening at the time beat i.
    """
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
        nh(int): number of hands
        ct_time_seq ([ct]): catch-throw sequence where each element is (n_c, n_t)

    Returns:

        [np.array((M_h, nh, 8))]: Where M_h is the number of CTs a hand performs in one period.

                                  Each element in this matrix holds (h_nr, beat_nr, n_c, n_t, h_c, h_t, i_c, i_t) where:
                                    h_nr is the index of the hand this CT is part of
                                    beat_nr is the beat number of the actual CT
                                    n_c/n_t are number of the receiving/throwin ball
                                    h_c/h_t are indexes of the hand we receive from/throw to
                                    i_c/i_t are indexes of the CTs we receive from/throw to

    """
    N = len(ct_time_seq) # Period
    # Calculate nr of beats required for each hand to perform perform all c-t it will ever face at least once(least common multiple)
    M = lcm(N, nh)
    # Nr of ct a hand performs in one period
    M_h = M//nh
    # Fill a M_h x nh matrix with (c,t) pairs
    ct_period_per_hand = np.zeros((M_h, nh, 8), dtype=np.int)
    for i in range(M_h):  # i: ct index for the CT
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
    """ Initializes the hands and their corresponding CTs(catch-throws)

    Args:
        nh (int): nr of hands
        ct_period_per_hand (list((n_c, n_t))): number of catch and throw for each CT
        hand_positions (np.array(n_h,3)): cartesian positions of the hand

    Returns:
        (list(JugglingHand)): list of JugglingHand objects, one for each hand
    """
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
        cts[h][i].addCatchThrowNextCTs(cts[h_c][ct_c_i], cts[h_t][ct_t_i], cts[h][(i+1)%N])
        # add ct to corresponding hand
        hands[h].addCT(i, cts[h][i])
    return hands

  def getHandPositions(self, nh, w):
    """ Initializes th cartesian positions of the hands assuming they
        lie on the edges of a regular polygon with equal sides of length w.

    Args:
        nh (int): nr of hands
        w (float): side length(distance between two neighboring hands)

    Returns:
        hand_positions (np.array(n_h,3)): cartesian positions of the hand
    """
    hand_positions = np.zeros((nh, 3))
    angle_step = np.pi*(float(nh) - 2.0)/float(nh) - np.pi
    angles  = np.arange(nh)*angle_step + np.pi
    x,y,z = 0.0, 0.0, 0.0
    for h in range(nh):
      hand_positions[h] = (x,y,z)
      # update x,y for next hand, assume hands are on the same height
      x = x + w * math.cos(angles[h])
      y = y + w * math.sin(angles[h])
    return hand_positions


if __name__ == "__main__":
  dt = 0.004
  jp = JugglingPlanner()
  # jp.plan(1, pattern=(3,3,3), rep=1).plot()
  # p = jp.plan(dt, 3, pattern=(3,5,3,5), slower=5, rep=1)
  p = jp.plan(dt, 3, h=0.4, pattern=(4,), slower=5, w=0.7, rep=1)
  p.plot()

  # jp.plan(4, pattern=(3,), rep=1).plot()
  # jp.plan(dt, 4, pattern=(5,), slower=5, rep=1).plot()
  # N_Whole0, x0, v0, a0, j0, thetas = p.hands[0].get(True)  # get plan for hand0
  # plt.plot(a0[:,0])
  # plt.plot(a0[:,1])
  # plt.plot(a0[:,2])
  # plt.show()
  # p.plot()
  # print(p.hands[0].getTimesPositionsVelocities())