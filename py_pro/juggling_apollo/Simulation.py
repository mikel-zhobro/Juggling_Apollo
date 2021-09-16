
import numpy as np
from utils import steps_from_time, find_continuous_intervals, plot_intervals, plt
from settings import m_b, m_p, k_c, g
from juggling_apollo.Visual import Paddle


class Simulation:
  # Constructor
  def __init__(self, input_is_force, air_drag, sys=None, plate_friction=False):
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

  def force_from_velocity(self, u_des_p, u_p):
    F_p = m_p * k_c * (u_des_p - u_p)
    return F_p

  def simulate_one_iteration(self, dt, T, x_b0, x_p0, u_b0, u_p0, u, repetitions=1, d=None, visual=False, pause_on_hight=None, it=0):
    """ Simulates the system from the time interval 0->T

    Args:
        dt ([double]): [description]
        T ([double]): [description]
        x_b0 ([double]): [description]
        x_p0 ([double]): [description]
        u_b0 ([double]): [description]
        u_p0 ([double]): [description]
        u ([np.array(double)]): [description]
        d ([np.array(double)], optional): [description]. Defaults to None.
        repetitions (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec of shape [1, N]
    """
    x_b0 = np.asarray(x_b0)
    u_b0 = np.asarray(u_b0)
    x_p0 = np.asarray(x_p0)
    u_p0 = np.asarray(u_p0)
    if visual:
      if self.vis is None:
        self.vis = Paddle(x_b0, x_p0, dt)
    if self.vis:
      self.vis.reset(x_b0, x_p0, it)

    if d is None:
      d = np.zeros(u.shape)  # disturbance

    d = np.squeeze(np.tile(d.reshape(u.shape), [repetitions, 1]))
    u = np.squeeze(np.tile(u, [repetitions, 1]))
    # Vectors to collect the history of the system states
    N0 =steps_from_time(T, dt)-1
    N = N0 * repetitions + 1
    x_b = np.zeros((N,) + x_b0.shape); x_b[0] = x_b0
    u_b = np.zeros((N,) + u_b0.shape); u_b[0] = u_b0
    x_p = np.zeros((N, 1)); x_p[0] = x_p0
    u_p = np.zeros((N, 1)); u_p[0] = u_p0
    # Vector to collect extra info for debugging
    dP_N_vec = np.zeros_like(x_b)
    gN_vec = np.zeros_like(x_b)
    u_vec = np.zeros((N, 1))

    # Simulation
    repetition = 0
    old_repetition = -1
    old_repetition2 = -1
    for i in range(N-1):
      # one step simulation
      if i%N0==1:
        repetition +=1
        if visual:
          self.vis.update_repetition(repetition)
      x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_i, contact_impact = self.simulate_one_step(dt, u[i], x_b[i], x_p[i], u_b[i], u_p[i], d[i])
      if visual:
        self.vis.run_frame(x_b_new, x_p_new, u_b_new, u_p_new)
        if pause_on_hight is not None and i%N0>N0/4 and any(contact_impact) and old_repetition != repetition:
          old_repetition = repetition  # stop only once per repetition
          self.vis.plot_catch_line()
        if pause_on_hight is not None and i%N0<N0/4 and not any(contact_impact) and old_repetition2 != repetition:
          old_repetition2 = repetition  # stop only once per repetition
          self.vis.plot_throw_line()
      # collect state of the system
      x_b[i+1] = x_b_new
      x_p[i+1] = x_p_new
      u_b[i+1] = u_b_new
      u_p[i+1] = u_p_new
      # collect helpers
      dP_N_vec[i] = dP_N
      gN_vec[i] = gN
      u_vec[i] = u_i
    return x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec

  def simulate_one_step(self, dt, u_i, x_b_i, x_p_i, u_b_i, u_p_i, plate_force_disturbance):
    # 1.Generalized to multiple balls: x_b_i and u_b_i can be vectors
    # 2.This works only with force as input: so if we get speed we transform it to force.
    F_i = u_i
    if not self.input_is_force:  # did we get speed?
      F_i = self.force_from_velocity(u_i, u_p_i)

    # disturbances
    plate_friction_force_disturbance = self.get_plate_force_friction(u_p_i)
    ball_friction_force_disturbance = self.get_ball_force_friction(u_b_i)
    gravity_force = dt*(g + ball_friction_force_disturbance/m_b)
    F_i = F_i + plate_force_disturbance - plate_friction_force_disturbance

    x_b_1_2 = x_b_i + 0.5*dt*u_b_i
    x_p_1_2 = x_p_i + 0.5*dt*u_p_i

    # gN = x_b_1_2 - x_p_1_2
    gN = x_b_i - x_p_i
    gamma_n_i = u_b_i - u_p_i
    # && (((-gamma_n_i + g*dt + u_i*dt/m_p))>=0)
    contact_impact = gN <= 1e-5
    dP_N = np.where(contact_impact, np.maximum(0, (-gamma_n_i + gravity_force + F_i*dt/m_p) / (m_b ** -1 + m_p ** -1)), 0)

    u_b_new = u_b_i - gravity_force + dP_N / m_b
    u_p_new = u_p_i + F_i * dt / m_p - np.sum(dP_N / m_p)
    x_b_new = x_b_1_2 + 0.5 * dt * u_b_new
    x_p_new = x_p_1_2 + 0.5 * dt * u_p_new
    return x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, F_i, contact_impact

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


def plot_simulation(dt, u, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, x_p_des=None, title = None, vertical_lines = None):
  # Everything are column vectors
  intervals = find_continuous_intervals(gN_vec)
  # print("INTERVALS: ", np.array(intervals[0])*dt, np.array(intervals[1])*dt)
  fig, axs = plt.subplots(5, 1)

  timesteps = np.arange(u_p.size) * dt
  axs[0].plot(timesteps, x_b, label='Ball position [m]')
  axs[0].plot(timesteps, x_p, 'b', label='Plate position [m]')
  axs[0].axhline(y=0.0, color='y', linestyle='-')
  if x_p_des is not None:
    axs[0].plot(timesteps, x_p_des, color='green', linestyle='dashed', label='Desired Plate position [m]')
  axs[1].plot(timesteps, u_b, label='Ball velocity [m/s]')
  axs[1].plot(timesteps, u_p, 'b', label='Plate velocity [m/s]')
  axs[2].plot(timesteps, dP_N_vec, label='dP_N')
  axs[3].plot(timesteps, gN_vec, label='g_{N_{vec}} [m]')
  axs[4].plot(timesteps, u, 'b', label='F [N]')
  for ax in axs:
    ax = plot_intervals(ax, intervals, dt)
    if vertical_lines is not None:
      for pos, label in vertical_lines.items():
        ax.axvline(pos, linestyle='--', color='k', label=label)
    ax.legend(loc=1)

  if title is not None:
    fig.suptitle(title)
  plt.show()
