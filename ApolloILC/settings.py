global g, dt, m_b, m_p, k_c


# Constants
g = 9.80665     # [m/s^2] gravitational acceleration constant
dt = 0.004      # [s] discretization time step size

# Params
m_b = 0.1       # [kg] mass of ball
m_p = 10.0        # [kg] mass of plate
k_c = 10.0        # [1/s] force coefficient
ABS = 1e-5

# Apollo
alpha = 10