import turtle as t
import time
import numpy as np


class Paddle():
  def __init__(self, x0b, x0p, dt, colors=None):

    self.it = 0
    # meter to pixel
    self.fact = 150
    self.const = -125
    self.delay = dt

    # Setup Background
    self.win = t.Screen()
    self.win.title('Platte and ball setup')
    self.win.bgcolor('black')
    self.win.setup(width=600, height=600)
    self.win.tracer(0)

    # Ball
    self.balls = [t.Turtle() for _ in np.array(x0b).reshape(-1, 1)]
    if colors is None:
      colors = ['red'] * len(self.balls)
    for ball, color in zip(self.balls, colors):
      ball.speed(0)
      ball.shape('circle')
      ball.color(color)
      ball.penup()

    # Paddle
    self.paddle = t.Turtle()
    self.paddle.speed(0)
    self.paddle.shape('square')
    self.paddle.shapesize(stretch_wid=1, stretch_len=5)
    self.paddle.color('white')
    self.paddle.penup()

    # Info
    self.info = t.Turtle()
    self.info.speed(0)
    self.info.color('white')
    self.info.penup()
    self.info.hideturtle()
    self.info.goto(0, 250)
    self.info.write("Iteration: {}".format(self.it), align='center', font=('Courier', 30, 'bold'))

    # The 0-line
    self.line = t.Turtle()
    self.line.shape('square')
    self.line.goto(0, self.const)
    self.line.shapesize(stretch_wid=0.03, stretch_len=25)
    self.line.color('blue')
    self.line.penup()

    self.reset(x0b, x0p)

  def run_frame(self, x_b, x_p, u_b, u_p):
    self.win.update()
    # Update
    for ball, xb in zip(self.balls, np.array(x_b).reshape(-1, 1)):
      ball.sety(self.m2pixel(xb)+20)
      # self.ball.dy = u_b

    self.paddle.sety(self.m2pixel(x_p))
    self.paddle.dy = u_p

    time.sleep(self.delay)
    # self.win.delay(int(self.delay*1000))

  def m2pixel(self, x):
    return int(x*self.fact + self.const)

  def reset(self, x_b, x_p):
    self.it +=1
    self.info.clear()
    self.info.write("Iteration: {}".format(self.it), align='center', font=('Courier', 30, 'bold'))

    self.paddle.clear()
    self.paddle.goto(0, self.m2pixel(x_p))
    self.paddle.dy = 0

    for ball, xb in zip(self.balls, np.array(x_b).reshape(-1, 1)):
      ball.clear()
      ball.goto(0, self.m2pixel(xb)+20)
      ball.dy = 0



#!/usr/bin/env python3

def main():
  env  = Paddle([0, 0.2, -0.3], -0.4, 0.4, colors = ['red', 'yellow', 'green'])
  for i in range(100):
    x_b_new = np.random.rand(3, 1)
    x_p_new = np.random.rand(1)
    env.run_frame(x_b_new, x_p_new, 0, 0)

##############################################################################

if __name__ == "__main__":
    main()
