import path_setter
import numpy as np
from juggling_apollo.LiftedStateSpace import LiftedStateSpace
from juggling_apollo.OptimLss import OptimLss


def test1():
  class MySys:
    def __init__(self, A, B, C, S, c):
      self.Ad = A
      self.Ad_impact = A  # [nx, nx]
      self.Bd = B
      self.Bd_impact = B  # [nx, nu]
      self.Cd = C         # [ny, nx]
      self.S = S          # [nx, ndup]
      self.c = c
      self.c_impact = c   # constants from gravity ~ dt, g, mp mb

  # Desired Input optimization example
  # Create Ad, Bd, Cd, S, c and x0
  N = 4
  A = np.diag([1, 2, 3])
  B = 2*np.ones((3, 1))
  C = np.ones((2, 3)); C[0, 0] = 0
  S = np.ones((3, 1))
  x0 = 0.2*np.ones((3, 1))  # initial state (xb0, xp0, ub0, up0)
  c = 0.1*np.ones((3, 1))

  sys = MySys(A, B, C, S, c)
  lss = LiftedStateSpace(sys, x0)
  set_of_impact_timesteps = np.ones(N)
  # set_of_impact_timesteps = (1, 0, 1)
  lss.updateQuadrProgMatrixes(set_of_impact_timesteps)

  optim = OptimLss(lss)
  # Test solving the quadratic problem
  dup = np.zeros((N, 1))
  y_des = np.ones((2*N, 1))
  u_des = optim.calcDesiredInput(dup=dup, y_des=y_des, print_norm=True)
  print("desired input is: ", u_des)


def test_all():
  test1()


if __name__ == "__main__":
  test_all()
