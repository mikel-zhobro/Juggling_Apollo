import numpy as np
from math import cos, sin

d1 = 0.2
d3 = 0.3
d5 = 0.4
d7 = 0.1

pi_2 = np.pi/2
off1 = 0.0
off2 = -pi_2
off3 = +pi_2
off4 = 0.0
off5 = -pi_2
off6 = 0.0
off7 = 0.0

def FK(q1, q2, q3, q4, q5, q6, q7):
    
    q1 += off1
    q2 += off2
    q3 += off3
    q4 += off4
    q5 += off5
    q6 += off6
    q7 += off7
    
    c1 = cos(q1); s1 = sin(q1)
    c2 = cos(q2); s2 = sin(q2)
    c3 = cos(q3); s3 = sin(q3)
    c4 = cos(q4); s4 = sin(q4)
    c5 = cos(q5); s5 = sin(q5)
    c6 = cos(q6); s6 = sin(q6)
    c7 = cos(q7); s7 = sin(q7)
    
    
    A1 = c1*c2*c3 + s1*s3;  Aa1 = A1*c4 - c1*s2*s4
    A2 = -c1*c2*s3 + c3*s1; Aa2 = A1*s4 - c1*c4*s2; Aa12 = Aa1*c5 + A2*s5

    B1 = -c1*s3 + c2*c3*s1; Bb1 = B1*c4 - s1*s2*s4
    B2 = -c1*c3 - c2*s1*s3; Bb2 = -B1*s4 - c4*s1*s2; Bb12 = Bb1*c5 + B2*s5

    C1 = c2*s4 + c3*c4*s2; Cc1 = C1*c5 - s2*s3*s5
    C2 = c2*c4 - c3*s2*s4

    Xa = Aa12*c6 + Aa2*s6
    Xb = -Aa1*s5 + A2*c5

    Ya = Bb12*c6 + Bb2*s6
    Yb = -Bb1*s5 + B2*c5

    Za = Cc1*c6 + C2*s6
    Zb = -C1*s5 - c5*s2*s3


    n0_7 = np.array([Xa*c7 + Xb*s7,
                    Ya*c7 + Yb*s7,	
                    Za*c7 + Zb*s7], dtype='float')


    s0_7 = np.array([Aa12*s6 - Aa2*c6,
                    Bb12*s6 - Bb2*c6,
                    Cc1*s6 - C2*c6], dtype='float')


    a0_7 = np.array([Xa*s7 - Xb*c7,
                    Ya*s7 - Yb*c7,
                    Za*s7 - Zb], dtype='float')


    p0_7 = np.array([
        (Aa12*s6 - Aa2*c6)*d7 + (A1*s4 + c1*c4*s2)*d5 + c1*d3*s2,
        (Bb12*s6 - Bb2*c6)*d7 + (B1*s4 + c4*s1*s2)*d5 + d3*s1*s2,
        (Cc1*s6 - C2*c6)*d7 + (-c2*c4 + c3*s2*s4)*d5 - c2*d3 + d1], dtype='float')

    T =  np.vstack([np.vstack([n0_7, s0_7, a0_7, p0_7]).T, [0.0, 0.0, 0.0, 1.0]])
    
    return T

np.set_printoptions(precision=3, suppress=True)
print(FK(0, 0, 0, 0, 0, 0, 0))