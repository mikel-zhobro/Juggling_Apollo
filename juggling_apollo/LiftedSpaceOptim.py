#!/usr/bin/env python3
import numpy as np


class LiftedSpaceOptim:
  def __init__(self, lifted_state_space):
    self.lss = lifted_state_space

  def calcDesiredInput(obj, dup, y_des):
    # TODO: Add penalty on the input
    u_des = np.linalg.lstsq((obj.lss.GF.T).dot(obj.lss.GF), obj.lss.GF.T.dot((obj.lss.GK*dup + obj.lss.Gd0 - y_des)))
    # np.norm( (obj.lss.GF.T).dot(obj.lss.GF)* u_des + obj.lss.GF.T.dot((obj.lss.GK*dup + obj.lss.Gd0 - y_des) )
    return u_des


def main():
  #    We are trying to find x for  H*x=f
  H1 = np.random.rand(4, 4)
  H2 = np.arange(16).reshape(4, 4)
  f = np.ones((4, 1))

  x1 = np.linalg.lstsq(H1, f)[0]
  x2 = np.linalg.lstsq(H2, f)[0]
  print("Random 4x4 matrix")
  print(f"x1={x1}")
  print(f"H*x1={H1.dot(x1)}")
  print("A singular matrix 4x4 matrix")
  print(f"x2={x2}")
  print(f"H*x2={H2.dot(x2)}")


if __name__ == "__main__":
    main()
