
import numpy as np
from utils import steps_from_time, find_continuous_intervals, plot_intervals, plt
from settings import m_b, m_p, k_c, g, ABS
from Visual import Paddle


class Apollo_it:
  # Constructor
  def __init__(self, x0):
    # state
    self.reset(x0)

  def reset(self, x0=None):
    if x0 is not None:
      # apollo.goTo(x0)
      # self.x0 = FK(apollo.pulse)
      pass

  def simulate_one_iteration(self, dt, T, u, x0=None, repetitions=1, it=0):
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

    assert abs(T-len(u)*dt) <= dt, "Input signal is of length {} instead of length {}".format(len(u)*dt ,T)
    N0 = len(u)
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

    # Action Loop
    repetition = 0
    for i in range(N-1):
      # one step simulation
      if i%N0==1:
        repetition +=1
      x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_i, contact_impact = self.run_one_step(dt, u[i])
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

  def run_one_step(self, dt, u_i):
    # apollo.send_command(u_i)
    # apollo.observation() -> x_TCP
    # measure in the meantime the force sensors?
    p_tcp = 0.0
    R_tcp = 0.0
    impulse = 0.0
    return p_tcp, R_tcp, impulse