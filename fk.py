import numpy as np
from math import cos, sin, acos, atan2, sqrt

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

q2_fixed = 0.0
q3_fixed = 0.0
q5_fixed = pi_2
q7_fixed = -pi_2


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
    A2 = -c1*c2*s3 + c3*s1; Aa2 = A1*s4 + c1*c4*s2


    B1 = -c1*s3 + c2*c3*s1; Bb1 = B1*c4 - s1*s2*s4
    B2 = -c1*c3 - c2*s1*s3; Bb2 = B1*s4 + s1*s2*c4

    C1 = c2*s4 + c3*s2*c4; 
    C2 = -c2*c4 + c3*s2*s4; Cc1 = C1*c5 - s2*s3*s5

    Aa12 = Aa1*c5 + A2*s5
    Bb12 = Bb1*c5 + B2*s5
    
    Xa = Aa12*c6 - Aa2*s6
    Xb = -Aa1*s5 + A2*c5

    Ya = Bb12*c6 - Bb2*s6
    Yb = -Bb1*s5 + B2*c5

    Za = Cc1*c6 - C2*s6
    Zb = -C1*s5 - s2*s3*c5

    P7a = Aa12*s6 + Aa2*c6
    P7b = Bb12*s6 + Bb2*c6
    P7c = Cc1*s6 + C2*c6
    n0_7 = np.array([Xa*c7 + Xb*s7,
                     Ya*c7 + Yb*s7,
                     Za*c7 + Zb*s7], dtype='float')


    s0_7 = np.array([-Aa12*s6 - Aa2*c6,
                     -Bb12*s6 - Bb2*c6,
                     -Cc1*s6 - C2*c6], dtype='float')



    a0_7 = np.array([-Xa*s7 + Xb*c7,
                     -Ya*s7 + Yb*c7,
                     -Za*s7 + Zb*c7], dtype='float')


    p0_7 = np.array([P7a*d7 + Aa2*d5 + c1*s2*d3,
                     P7b*d7 + Bb2*d5 + s1*s2*d3,
                     P7c*d7 + C2*d5 - c2*d3 + d1], dtype='float')
    
    T =  np.vstack([np.vstack([n0_7, s0_7, a0_7, p0_7]).T, [0.0, 0.0, 0.0, 1.0]])
    
    return T


def FK_reduced(q1, q4, q6):
    # FK with fixed 2,3,5 and 7 joint
    return FK(q1, q2_fixed, q3_fixed, q4, q5_fixed, q6, q7_fixed)


def IK_reduced(x_TCP, y_TCP, theta_TCP):
    # The workspace is:
    # 
    x_j6 = x_TCP + d7*cos(theta_TCP)
    y_j6 = y_TCP + d7*sin(theta_TCP)
    
    p_j6 = x_j6**2 + y_j6**2
    d3_2 = d3**2
    d5_2 = d5**2
    
    assert sqrt(p_j6) < d3 + d5, "The required position lies outside the reachable workspace of Apollo"
    # acos returns in range [0, pi]
    q4 = acos((p_j6 - d3_2 - d5_2)/(2*d3*d5))
    
    # atan2 retunrs in range[-pi, pi] depending on the quadrant
    q1 = atan2(y_j6, x_j6) + np.pi - acos((d3_2 - d5_2 + p_j6)/(2*d3*sqrt(p_j6)))
    
    q6 = theta_TCP - q1 - q4
    return q1, q4, q6


def J(q1, q2, q3, q4, q5, q6, q7):
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
    A2 = -c1*c2*s3 + c3*s1; Aa2 = A1*s4 + c1*c4*s2; Aa12 = Aa1*c5 + A2*s5


    B1 = -c1*s3 + c2*c3*s1; Bb1 = B1*c4 - s1*s2*s4
    B2 = -c1*c3 - c2*s1*s3; Bb2 = B1*s4 + c4*s1*s2; Bb12 = Bb1*c5 + B2*s5

    C1 = c2*s4 + c3*c4*s2; 
    C2 = -c2*c4 + c3*s2*s4; Cc1 = C1*c5 - s2*s3*s5

    Xa = Aa12*c6 - Aa2*s6
    Xb = -Aa1*s5 + A2*c5

    Ya = Bb12*c6 - Bb2*s6
    Yb = -Bb1*s5 + B2*c5

    Za = Cc1*c6 - C2*s6
    Zb = -C1*s5 - c5*s2*s3

    P7a = Aa12*s6 + Aa2*c6
    P7b = Bb12*s6 + Bb2*c6
    P7c = Cc1*s6 + C2*c6


    # Calc zi and pi
    z0 = np.array([[0, 0, 1]], dtype='float').T
    p0 = np.array([[0, 0, 0]], dtype='float').T
    
    z1 = np.array([[s1, -c1, 0]], dtype='float').T
    p1 = np.array([[0, 0, d1]], dtype='float').T
    
    z2 = np.array([[c1*s2, s1*s2, -c2]], dtype='float').T
    p2 = np.array([[0, 0, d1]], dtype='float').T
    
    z3 = np.array([[A2, B2, -s2*s3]], dtype='float').T
    p3 = np.array([[c1*d3*s2,
                   d3*s1*s2,
                   -c2*d3 + d1]], dtype='float').T
    z4 = np.array([[Aa2, Bb2, C2]], dtype='float').T
    p4 = p3

    z5 = np.array([[-Aa1*s5 + A2*c5,
                   -Bb1*s5 + B2*c5,
                   -C1*s5 - c5*s2*s3]], dtype='float').T
    p5 = np.array([[Aa2*d5 + c1*d3*s2,
                   Bb2*d5 + d3*s1*s2,
                   C2*d5 - c2*d3 + d1]], dtype='float').T

    z6 = np.array([[P7a, P7b, P7c]], dtype='float').T
    p6 = p5


    z7 = np.array([[-Xa*s7 + Xb*c7,
                   -Ya*s7 + Yb*c7,
                   -Za*s7 + Zb*c7]], dtype='float').T
    p7 = np.array([[P7a*d7, P7b*d7, P7c*d7]], dtype='float').T + p5   

    Jp1 = np.cross(z0, p7-p0, axis=0)
    Jp2 = np.cross(z1, p7-p1, axis=0)
    Jp3 = np.cross(z2, p7-p2, axis=0)
    Jp4 = np.cross(z3, p7-p3, axis=0)
    Jp5 = np.cross(z4, p7-p4, axis=0)
    Jp6 = np.cross(z5, p7-p5, axis=0)
    Jp7 = np.cross(z6, p7-p6, axis=0)
    
    
    JP = np.hstack([Jp1, Jp2, Jp3, Jp4, Jp5, Jp6, Jp7])
    JO = np.hstack([z0, z1, z2, z3, z4, z5, z6])
    
    J = np.vstack([JP, JO])
    
    return J


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    # print(FK(0, 0, 0, pi_2, 0, 0, 0))
    print(FK_reduced(0, 0, -pi_2))


    # q1, q2 ,q3 = IK_reduced(-0.3, -0.3, pi_2)
    # print(q1 + q2 + q3, pi_2)
    print(FK_reduced(*IK_reduced(-0.5, -0.5, pi_2)))