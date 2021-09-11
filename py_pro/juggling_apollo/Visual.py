import turtle as t
import time


class Paddle():
  def __init__(self, x0b, x0p, dt):

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

    # Paddle
    self.paddle = t.Turtle()
    self.paddle.speed(0)
    self.paddle.shape('square')
    self.paddle.shapesize(stretch_wid=1, stretch_len=5)
    self.paddle.color('white')
    self.paddle.penup()

    # Ball
    self.ball = t.Turtle()
    self.ball.speed(0)
    self.ball.shape('circle')
    self.ball.color('red')
    self.ball.penup()

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
    self.ball.sety(self.m2pixel(x_b)+20)
    self.ball.dy = u_b

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

    self.ball.clear()
    self.ball.goto(0, self.m2pixel(x_b)+20)
    self.ball.dy = 0
