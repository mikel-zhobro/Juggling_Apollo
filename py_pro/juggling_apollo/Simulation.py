
import numpy as np
from utils import steps_from_time, find_continuous_intervals, plot_intervals, plt
from settings import m_b, m_p, k_c, g, ABS
from juggling_apollo.Visual import Paddle


class Simulation:
  # Constructor
  def __init__(self, input_is_force, air_drag, x0, sys=None, plate_friction=False):
    # true if input is force, false if input is velocity
    self.input_is_force = input_is_force
    # dynamic sys used to get state space matrixes of system(TODO)
    self.sys = sys
    # bool whether we should add air drag to the ball
    self.air_drag = air_drag
    # bool whether we should add some friction to the plate movement
    self.plate_friction = plate_friction
    # visualisation
    self.vis = None
    
    # state
    self.x_b0 = np.asarray(x0[0]).reshape(-1)
    self.x_p0 = np.asarray(x0[1]).reshape(-1)
    self.u_b0 = np.asarray(x0[2]).reshape(-1)
    self.u_p0 = np.asarray(x0[3]).reshape(-1)
    self.x_b = None
    self.x_p = None
    self.u_b = None
    self.u_p = None

  def reset(self, x0=None):
    if x0 is not None:
      self.x_b0 = np.asarray(x0[0]).reshape(-1)
      self.x_p0 = np.asarray(x0[1]).reshape(-1)
      self.u_b0 = np.asarray(x0[2]).reshape(-1)
      self.u_p0 = np.asarray(x0[3]).reshape(-1)
    self.x_b = self.x_b0
    self.x_p = self.x_p0
    self.u_b = self.u_b0
    self.u_p = self.u_p0

  def force_from_velocity(self, u_des_p, u_p):
    F_p = m_p * k_c * (u_des_p - u_p)
    return F_p

  def simulate_one_iteration(self, dt, T, u, x0=None, repetitions=1, d=None, visual=False, pause_on_hight=None, it=0, slow=1):
    """ Simulates the system from the time interval 0->T

    Args:
        dt ([double]): [description]
        T ([double]): [description]
        x0 ([list]):
        u ([np.array(double)]): [description]
        d ([np.array(double)], optional): [description]. Defaults to None.
        repetitions (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec of shape [1, N]
    """
    if visual:
      if self.vis is None:
        self.vis = Paddle(self.x_b, self.x_p, dt, colors=['red', 'blue'])
    if self.vis:
      self.vis.reset(self.x_b, self.x_p, it)

    assert abs(T-len(u)*dt) <= dt, "Input signal is of length {} instead of length {}".format(len(u)*dt ,T)
    N0 = len(u)
    if d is None:
      d = np.zeros(u.shape)  # disturbance
    d = np.squeeze(np.tile(d.reshape(u.shape), [repetitions, 1]))
    u = np.squeeze(np.tile(u, [repetitions, 1]))
    # Vectors to collect the history of the system states
    # N0 = steps_from_time(T, dt)-1
    
    N = N0 * repetitions + 1
    if x0 is not None:
      self.reset(x0)
    x_b = np.zeros((N,) + self.x_b.shape); x_b[0] = self.x_b
    u_b = np.zeros((N,) + self.u_b.shape); u_b[0] = self.u_b
    x_p = np.zeros((N, 1)); x_p[0] = self.x_p
    u_p = np.zeros((N, 1)); u_p[0] = self.u_p
    # Vector to collect extra info for debugging
    gN_vec = np.zeros_like(x_b);   gN_vec[0] = self.x_b - self.x_p
    dP_N_vec = np.zeros_like(x_b)
    u_vec = np.zeros((N-1, 1))

    # Simulation
    repetition = 0
    old_repetition = -1
    old_repetition2 = -1
    throw = False
    for i in range(N-1):
      # one step simulation
      if i%N0==1:
        repetition +=1
      x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_i, contact_impact = self.simulate_one_step(dt, u[i], d[i])
      if visual:
        self.visual(i, N0, repetition, contact_impact, old_repetition, old_repetition2 ,slow)
      # collect state of the system
      x_b[i+1] = x_b_new
      x_p[i+1] = x_p_new
      u_b[i+1] = u_b_new
      u_p[i+1] = u_p_new
      # collect helpers
      dP_N_vec[i+1] = dP_N
      gN_vec[i+1] = gN
      u_vec[i] = u_i
    if x0 is not None:
      return x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec
    else:
      return x_b[1:], u_b[1:], x_p[1:], u_p[1:], dP_N_vec[1:], gN_vec[1:], u_vec

  def simulate_one_step(self, dt, u_i, plate_force_disturbance):
    # 1.Generalized to multiple balls: self.x_b and self.u_b can be vectors
    # 2.This works only with force as input: so if we get speed we transform it to force.
    F_i = u_i
    if not self.input_is_force:  # did we get speed?
      F_i = self.force_from_velocity(u_i, self.u_p)

    # disturbances
    plate_friction_force_disturbance = self.get_plate_force_friction(self.u_p)
    ball_friction_force_disturbance = self.get_ball_force_friction(self.u_b)
    gravity_force = dt*(g + ball_friction_force_disturbance/m_b)
    F_i = F_i + plate_force_disturbance - plate_friction_force_disturbance

    x_b_1_2 = self.x_b + 0.5*dt*self.u_b
    x_p_1_2 = self.x_p + 0.5*dt*self.u_p

    # gN = x_b_1_2 - x_p_1_2
    gN = self.x_b - self.x_p
    gamma_n_i = self.u_b - self.u_p
    # && (((-gamma_n_i + g*dt + u_i*dt/m_p))>=0)
    contact_impact = gN <= ABS
    dP_N = np.where(contact_impact, np.maximum(0, (-gamma_n_i + gravity_force + F_i*dt/m_p) / (m_b ** -1 + m_p ** -1)), 0)

    self.u_b  = self.u_b - gravity_force + dP_N / m_b
    self.u_p = self.u_p + F_i * dt / m_p - np.sum(dP_N / m_p)
    self.x_b = x_b_1_2 + 0.5 * dt * self.u_b 
    self.x_p = x_p_1_2 + 0.5 * dt * self.u_p
    return self.x_b, self.x_p, self.u_b, self.u_p, dP_N, gN, F_i, contact_impact

  def get_ball_force_friction(self, v):
    # v can be both a vector or a scalar
    f_drag = 0
    if self.air_drag:
      # D is the diameter of the ball
      D = 0.4  # ball has diameter of 5cm
      p = 1.225  # [kg/m]  air density
      c = np.pi/16*p* (D**2)  # c = 1/4*p*A = pi/16*p*D**2
      f_drag = np.sign(v)*c* (v**2)
    return f_drag

  def get_plate_force_friction(self, v):
      f_drag = 0
      if self.plate_friction:
        c = 20
        f_drag = np.sign(v)*c* (v**2)
      return f_drag

  def visual(self,i, N0, repetition, contact_impact, old_repetition, old_repetition2 ,slow):
    self.vis.update_repetition(repetition)
    self.vis.run_frame(self.x_b, self.x_p, self.u_b, self.u_p, slow)
    if i%N0>N0/4 and any(contact_impact) and old_repetition != repetition:
      old_repetition = repetition  # stop only once per repetition
      self.vis.plot_catch_line()
    if i%N0<N0/4 and not any(contact_impact) and old_repetition2 != repetition:
      old_repetition2 = repetition  # stop only once per repetition
      self.vis.plot_throw_line()

def plot_simulation(dt, u, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, x_p_des=None, title=None, vertical_lines=None, horizontal_lines=None):
  # Everything are column vectors
  intervals = find_continuous_intervals(gN_vec)
  # print("INTERVALS: ", np.array(intervals[0])*dt, np.array(intervals[1])*dt)
  fig, axs = plt.subplots(5, 1)

  timesteps = np.arange(u_p.size) * dt
  axs[0].plot(timesteps, x_b, label='Ball [m]')
  axs[0].plot(timesteps, x_p, 'r', label='Plate [m]')
  # axs[0].axhline(y=0.0, color='y', linestyle='-')
  if horizontal_lines is not None:
        for pos, label in horizontal_lines.items():
          axs[0].axhline(pos, linestyle='--', color='brown')  # , label=label
  if x_p_des is not None:
    axs[0].plot(timesteps, x_p_des, color='green', linestyle='dashed', label='Desired')
  axs[1].plot(timesteps, u_b, label='Ball [m/s]')
  axs[1].plot(timesteps, u_p, 'r', label='Plate [m/s]')
  axs[2].plot(timesteps, dP_N_vec, label='dP_N')
  axs[3].plot(timesteps, gN_vec, label='g_{N_{vec}} [m]')
  axs[4].plot(timesteps, u, 'b', label='F [N]')
  for ax in axs:
    ax = plot_intervals(ax, intervals, dt)
    if vertical_lines is not None:
      for pos in vertical_lines:
        ax.axvline(pos, linestyle='--', color='k')
    ax.legend(loc=1)

  if title is not None:
    fig.suptitle(title)
  plt.show(block=False)
