import numpy as np

class LiftedSpaceOptim:
    properties
    lss;
  # Constructor
  def __init__(lifted_state_space):
      self.lss = lifted_state_space

  def calcDesiredInput(obj, dup, y_des):
    # u_des = linsolve(GF, -transpose(GF)*(GK*dup + Gd0 - y_des));
    # Add penalty on the input
    u_des = quadprog((transpose(obj.lss.GF) * obj.lss.GF), transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des));
    # check
    # norm( (transpose(obj.lss.GF) * obj.lss.GF)* u_des + transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des) )

