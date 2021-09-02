
import numpy as np
from utils import steps_from_time, find_continuous_intervals, plot_intervals, plt


class Simulation:

  # Constructor      m_b, m_p, k_c, g, input_is_force, air_drag
  def __init__(self, m_b, m_p, k_c, g, input_is_force, air_drag, sys=None, plate_cos_dis=False):
    self.m_b = m_b                         # mass of ball
    self.m_p = m_p                         # mass of plate
    self.k_c = k_c                         # force coefficient
    self.g = g                             # gravitational acceleration constant
    self.input_is_force = input_is_force   # true if input is force, false if input is velocity
    self.sys = sys                         # dynamic sys used to get state space matrixes of system
    self.air_drag = air_drag               # bool whether we should add air drag to the ball
    self.plate_cos_dis = plate_cos_dis     # bool whether we should add some cosinus form disturbance on the plate trajectory

  def force_from_velocity(self, u_des_p, u_p):
    F_p = self.m_p * self.k_c * (u_des_p - u_p)
    return F_p

  def simulate_one_iteration(self, dt, T, x_b0, x_p0, u_b0, u_p0, u, d=None, repetitions=1):
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
    if d is None:
      d = np.zeros(u.shape)  # disturbance

    u = np.repeat(u, repetitions)
    d = np.repeat(d, repetitions)
    # Vectors to collect the history of the system states
    N = steps_from_time(T, dt) * repetitions
    x_b = np.zeros(N); x_b[1] = x_b0
    u_b = np.zeros(N); u_b[1] = u_b0
    x_p = np.zeros(N); x_p[1] = x_p0
    u_p = np.zeros(N); u_p[1] = u_p0
    # Vector to collect extra info for debugging
    dP_N_vec = np.zeros(N)
    gN_vec = np.zeros(N)
    u_vec = np.zeros(N)

    # Simulation
    # for i = (1:N-1)
    for i in range(N-1):
        # one step simulation
        x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_i = \
          self.simulate_one_step(dt, u[i], x_b[i], x_p[i], u_b[i], u_p[i], d[i])
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
    # this works only with force as input
    # so if we get speed we transform it to force.
    F_i = u_i
    if not self.input_is_force:  # did we get speed?
      F_i = self.force_from_velocity(u_i, u_p_i)

    # disturbances
    ball_friction_force_disturbance = self.get_ball_force_friction(u_b_i)
    gravity_force = dt*(self.g + ball_friction_force_disturbance/self.m_b)
    F_i = F_i + plate_force_disturbance

    x_b_1_2 = x_b_i + 0.5*dt*u_b_i
    x_p_1_2 = x_p_i + 0.5*dt*u_p_i

    # gN = x_b_1_2 - x_p_1_2
    gN = x_b_i - x_p_i
    gamma_n_i = u_b_i - u_p_i
    contact_impact = gN<=1e-5  # && (((-gamma_n_i + self.g*dt + u_i*dt/self.m_p))>=0)
    if contact_impact:
      dP_N = max(0, (-gamma_n_i + gravity_force + F_i*dt/self.m_p)/ (self.m_b^-1 + self.m_p^-1))
      # dP_N = (-gamma_n_i + self.g*dt + u_i*dt/self.m_p)/ (self.m_b^-1 + self.m_p^-1)
    else:
      dP_N = 0

    u_b_new = u_b_i - gravity_force + dP_N / self.m_b
    u_p_new = u_p_i + F_i * dt / self.m_p - dP_N / self.m_p
    x_b_new = x_b_1_2 + 0.5 * dt * u_b_new
    x_p_new = x_p_1_2 + 0.5 * dt * u_p_new
    return x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, F_i

  def get_ball_force_friction(self, v):
    # D is the diameter of the ball
    # c = 1/4*p*A = pi/16*p*D^2
    f_drag = 0
    if self.air_drag:
        D = 0.4  # ball has diameter of 5cm
        p = 1.225  # [kg/m]  air density
        c = np.pi/16*p*D^2
        f_drag = np.sign(v)*c*v^2
    return f_drag


def plot_simulation(dt, u, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec):
  intervals = find_continuous_intervals(np.argwhere(np.squeeze(gN_vec)[1:-2] <= 1e-5))
  fig, axs = plt.subplots(5, 1)

  timesteps = np.arange(u_p.size) * dt
  axs[0].plot(timesteps, x_b[0], 'r', label='Ball position [m]')
  axs[0].plot(timesteps, x_p, 'b', label='Plate position [m]')
  axs[1].plot(timesteps, u_b, 'r', label='Ball velocity [m/s]')
  axs[1].plot(timesteps, u_p, 'b', label='Plate velocity [m/s]')
  axs[2].plot(timesteps, dP_N_vec, 'b', label='dP_N')
  axs[3].plot(timesteps, gN_vec, 'b', label='g_{N_{vec}} [m]')
  axs[4].plot(timesteps, u, 'b', label='F [N]')
  for ax in axs:
    ax = plot_intervals(ax, intervals, dt)
    # Position where legend can be put
    # ===============   =============
    # Location String   Location Code
    # ===============   =============
    # 'best'            0
    # 'upper right'     1
    # 'upper left'      2
    # 'lower left'      3
    # 'lower right'     4
    # 'right'           5
    # 'center left'     6
    # 'center right'    7
    # 'lower center'    8
    # 'upper center'    9
    # 'center'          10
    # ===============   =============
    ax.legend(loc=1)
  plt.show()


def main():
  dt = 0.04
  N = 225
  u = np.random.rand(N, 1)
  tt = np.arange(0, 225*dt, dt)
  x_b = np.sin(tt).reshape(1, -1)
  u_b = np.random.rand(N, 1)
  x_p = np.random.rand(N, 1)
  u_p = np.random.rand(N, 1)
  dP_N_vec = np.random.rand(N, 1)
  gN_vec = np.random.rand(N, 1) -0.5
  plot_simulation(dt, u, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)


if __name__ == "__main__":
  main()
