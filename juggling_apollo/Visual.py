import turtle as t
import time
import numpy as np


class Paddle():
  def __init__(self, x0b, x0p, dt, colors=None):
    self.pause = True

    # meter to pixel
    self.fact = 150
    self.const = -125
    self.delay = dt
    self.it = 0

    # Setup Background
    self.win = t.Screen()
    self.win.onscreenclick(self.clickHandler)
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
    self.set_info()

    # The 0_line
    self.line = t.Turtle()
    self.line.shape('square')
    self.line.goto(0, self.const)
    self.line.shapesize(stretch_wid=0.03, stretch_len=240)
    self.line.color('blue')
    self.line.goto(0, self.const)

    # The catch_line
    self.line2 = t.Turtle()
    self.line2.shape('square')
    self.line2.shapesize(stretch_wid=0.04, stretch_len=240)
    self.line2.color('red')
    self.line2.hideturtle()

    # The throw_line
    self.line3 = t.Turtle()
    self.line3.shape('square')
    self.line3.shapesize(stretch_wid=0.03, stretch_len=240)
    self.line3.color('green')
    self.line3.hideturtle()

    self.reset(x0b, np.asarray(x0p), 0)

  def run_frame(self, x_b, x_p, u_b, u_p, slow=1):
    if self.pause:
      self.wait_for_keypress()

    self.win.update()
    # Update
    for ball, xb in zip(self.balls, np.array(x_b).reshape(-1, 1)):
      ball.sety(self.m2pixel(xb)+20)
      # self.ball.dy = u_b

    self.paddle.sety(self.m2pixel(x_p))
    self.paddle.dy = u_p

    time.sleep(self.delay*slow)

  def update_repetition(self, repetition):
    self.info_rep.clear()
    self.info_rep.write("Repetition: {}".format(repetition), align='center', font=('Courier', 20, 'bold'))

  def m2pixel(self, x):
    return int(x*self.fact + self.const)

  def reset(self, x_b, x_p, it):
    self.it = it
    self.info.clear()
    self.info.write("Iteration: {}".format(it), align='center', font=('Courier', 30, 'bold'))

    self.paddle.clear()
    self.paddle.goto(0, self.m2pixel(x_p))
    self.paddle.dy = 0

    for ball, xb in zip(self.balls, np.array(x_b).reshape(-1, 1)):
      ball.clear()
      ball.goto(0, self.m2pixel(xb)+20)
      ball.dy = 0

  def plot_catch_line(self):
    self.line2.clear()
    self.line2.showturtle()
    self.line2.penup()
    self.line2.goto(0, self.paddle.pos()[1])

  def plot_throw_line(self):
    self.line3.clear()
    self.line3.showturtle()
    self.line3.penup()
    self.line3.goto(0, self.paddle.pos()[1])

  def wait_for_keypress(self):
    # Wait
    self.info.clear()
    self.info_rep.clear()
    self.info.write("WAITING FOR KEYPRESS", align='center', font=('Courier', 30, 'bold'))
    while self.pause:
      self.win.update()

  def clickHandler(self, x, y):
    self.pause = False
    self.info.clear()
    self.info.write("Iteration: {}".format(self.it), align='center', font=('Courier', 30, 'bold'))
    

  def set_info(self):
    # Info
    self.info = t.Turtle()
    self.info.color('white')
    self.info.penup()
    self.info.hideturtle()
    self.info.goto(0, 250)

    self.info_rep = t.Turtle()
    self.info_rep.color('white')
    self.info_rep.penup()
    self.info_rep.hideturtle()
    self.info_rep.goto(0, 220)

    self.info0 = t.Turtle()
    self.info0.color('blue')
    self.info0.penup()
    self.info0.hideturtle()
    self.info0.goto(130, 200)
    self.info0.write("----- 0 line", align='left', font=('Courier', 12, 'italic'))

    # self.info_throw = t.Turtle()
    # self.info_throw.color('green')
    # self.info_throw.penup()
    # self.info_throw.hideturtle()
    # self.info_throw.goto(130, 170)
    # self.info_throw.write("----- throw line", align='left', font=('Courier', 12, 'italic'))

    # self.info_catch = t.Turtle()
    # self.info_catch.color('red')
    # self.info_catch.penup()
    # self.info_catch.hideturtle()
    # self.info_catch.goto(130, 140)
    # self.info_catch.write("----- catch line", align='left', font=('Courier', 12, 'italic'))

def main():
  env  = Paddle([0, 0.2, -0.3], -0.4, 0.4, colors = ['red', 'yellow', 'green'])
  env.wait_for_keypress()
  for i in range(3):
    x_b_new = np.random.rand(3, 1)
    x_p_new = np.random.rand(1)
    env.run_frame(x_b_new, x_p_new, 0, 0)
  env.wait_for_keypress()


if __name__ == "__main__":
    main()
