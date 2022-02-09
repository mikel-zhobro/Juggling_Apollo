import numpy as np
import __add_path__
from ApolloILC.LiftedStateSpace import LiftedStateSpace


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

  A = np.diag([1, 2, 3])
  B = np.ones((3, 1))
  C = np.zeros((1, 3)); C[0, 0] = 1
  S = np.ones((3, 1))
  x0 = np.ones((3, 1))-2  # initial state (xb0, xp0, ub0, up0)
  c = np.ones((3, 1))

  sys = MySys(A, B, C, S, c)
  lss = LiftedStateSpace(sys, x0)
  # set_of_impact_timesteps = (True, False, True)
  set_of_impact_timesteps = (1, 0, 1)
  lss.updateQuadrProgMatrixes(set_of_impact_timesteps)

  print('GF', lss.GF)
  print('GK', lss.GK)
  print('Gd0', lss.Gd0)


def test_all():
  test1()


if __name__ == "__main__":
  test_all()
