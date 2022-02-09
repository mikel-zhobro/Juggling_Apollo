# %%
import numpy as np
from math import cos, sin, acos, atan2, sqrt
from scipy.linalg import logm

# Define position/orientation of base from of kuka arm to the world frame
c = cos(np.pi/6); s = sin(np.pi/6)
R_origin_base = np.array([[1.0, 0.0, 0.0],
                          [0.0,  c,  -s],
                          [0.0,  s,   c]], dtype='float')
p_origin_base = np.array([1.0, 0.0, 0.0], dtype='float').reshape(3,1)

d_bs = 0.1
d_se = 0.4
d_ew = 0.39
d_wt = 0.1
d_bs = 0.378724; d_se = 0.4; d_ew = 0.438; d_wt = 0.138

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
signs = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # alphas in DH table 1 if pi/2, -1 if -pi/2
ds = [d_bs, 0.0, d_se, 0.0, d_ew, 0.0, d_wt]
JOINTS_LIMITS = {
    "R_SFE":(-2.96,2.96),
    "R_SAA":(-3.1-off2, -0.1-off2),  # (-1.529, 1.47)
    "R_HR": (-1.9-off3, 4.0-off3),  # (-3.47, 2.4292)
    "R_EB": (-2.09,2.09),
    "R_WR": (-3.1-off5, 1.35-off5),  # (-1.5292, 2.9208)
    "R_WFE":(-2.09,2.09),
    "R_WAA":(-2.96,2.96),

    "L_SFE":(-2.96,2.96),
    "L_SAA":(-3.1,-0.1),
    "L_HR": (-1.9,4.0),
    "L_EB": (-2.09,2.09),
    "L_WR": (-3.1,1.35),
    "L_WFE":(-2.09,2.09),
    "L_WAA":(-2.96,2.96)
}


def FK_DH(q):
    q = np.array(q).reshape(-1, 1) + np.array([off1, off2, off3, off4, off5, off6, off7]).reshape(-1, 1)
    cs = np.cos(q)
    ss = np.sin(q)

    def getT(i):
        return np.array([
                        [cs[i], 0.0,        signs[i]*ss[i],     0.0],
                        [ss[i], 0.0,        -signs[i]*cs[i],    0.0],
                        [0.0,   signs[i],   0.0,                ds[i]],
                        [0.0,   0.0,        0.0,                1.0]]
                        , dtype='float')

    T0_7 = np.eye(4)
    for i in range(7):
        T0_7 = T0_7.dot(getT(i))

    # Make up for axes mismatch to real apollo  x_apollo->x_dh, y_apollo->z_dh, z_apollo-> -y_dh
    T0_7 = T0_7[:, [0, 2, 1, -1]]
    T0_7[:,2] *= -1
    return T0_7


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

    C1 = c2*s4 + c3*s2*c4
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


    s0_7 = np.array([-Xa*s7 + Xb*c7,
                     -Ya*s7 + Yb*c7,
                     -Za*s7 + Zb*c7], dtype='float')


    a0_7 = -np.array([-Aa12*s6 - Aa2*c6,
                      -Bb12*s6 - Bb2*c6,
                      -Cc1*s6 - C2*c6], dtype='float')


    p0_7 = np.array([P7a*d_wt + Aa2*d_ew + c1*s2*d_se,
                     P7b*d_wt + Bb2*d_ew + s1*s2*d_se,
                     P7c*d_wt + C2*d_ew - c2*d_se + d_bs], dtype='float')

    T =  np.vstack([np.vstack([n0_7, s0_7, a0_7, p0_7]).T, [0.0, 0.0, 0.0, 1.0]])

    return T

def FK_reduced_5(q1, q4, q6):
    # TODO
    # FK with fixed 2,3,5 and 7 joint
    return FK(q1, q2_fixed, q3_fixed, q4, q5_fixed, q6, q7_fixed)

def IK_reduced_5(x_TCP, y_TCP, theta_TCP):
    # TODO
    # The workspace is:
    pass

def FK_reduced_3(q1, q4, q6):
    # FK with fixed 2,3,5 and 7 joint
    return FK(q1, q2_fixed, q3_fixed, q4, q5_fixed, q6, q7_fixed)


def IK_reduced_3(x_TCP, y_TCP, theta_TCP):
    # The workspace is:
    #
    x_j6 = x_TCP + d_wt*cos(theta_TCP)
    y_j6 = y_TCP + d_wt*sin(theta_TCP)

    p_j6 = x_j6**2 + y_j6**2
    d3_2 = d_se**2
    d5_2 = d_ew**2

    assert sqrt(p_j6) < d_se + d_ew, "The required position lies outside the reachable workspace of Apollo"
    # acos returns in range [0, pi]
    q4 = acos((p_j6 - d3_2 - d5_2)/(2*d_se*d_ew))

    # atan2 retunrs in range[-pi, pi] depending on the quadrant
    q1 = atan2(y_j6, x_j6) + np.pi - acos((d3_2 - d5_2 + p_j6)/(2*d_se*sqrt(p_j6)))

    q6 = theta_TCP - q1 - q4
    return q1, q4, q6


def J(q1, q2, q3, q4, q5, q6, q7):
    """ Local Jacobian at the seventh joint

    Args:
        qi ([double]): value of ith joint

    Returns:
        [np.array()]: J in [6 x 7] where Jp = J[:3, :] is the positional jacobian and Jo = J[3:, :] is the orientational jacobian
    """
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

    C1 = c2*s4 + c3*c4*s2
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
    p1 = np.array([[0, 0, d_bs]], dtype='float').T

    z2 = np.array([[c1*s2, s1*s2, -c2]], dtype='float').T
    p2 = np.array([[0, 0, d_bs]], dtype='float').T

    z3 = np.array([[A2, B2, -s2*s3]], dtype='float').T
    p3 = np.array([[c1*d_se*s2,
                   d_se*s1*s2,
                   -c2*d_se + d_bs]], dtype='float').T
    z4 = np.array([[Aa2, Bb2, C2]], dtype='float').T
    p4 = p3

    z5 = np.array([[-Aa1*s5 + A2*c5,
                   -Bb1*s5 + B2*c5,
                   -C1*s5 - c5*s2*s3]], dtype='float').T
    p5 = np.array([[Aa2*d_ew + c1*d_se*s2,
                   Bb2*d_ew + d_se*s1*s2,
                   C2*d_ew - c2*d_se + d_bs]], dtype='float').T

    z6 = np.array([[P7a, P7b, P7c]], dtype='float').T
    p6 = p5


    z7 = np.array([[-Xa*s7 + Xb*c7,
                   -Ya*s7 + Yb*c7,
                   -Za*s7 + Zb*c7]], dtype='float').T
    p7 = np.array([[P7a*d_wt, P7b*d_wt, P7c*d_wt]], dtype='float').T + p5

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


def so3_2_SO3(theta, w):
    S = np.array([[0,    w[0], w[1]],
                  [w[0], 0.0,  w[2]],
                  [w[1], w[2], 0.0]], dtype='float')

    R = np.eye(3) + sin(theta)*S + (1-cos(theta)*S.dot(S))
    return R

def SO3_2_so3(R):
    theta = acos(np.clip((np.trace(R)-1.0)/2.0, -1, 1))
    if np.abs(theta) < 1e-6: return np.array([0.0, 0.0, 0.0], dtype='float').reshape(3,1), theta

    w = 0.5 * np.array([R[2,1] - R[1,2],
                        R[0,2] - R[2,0],
                        R[1,0] - R[0,1]], dtype='float').reshape(3,1)
    return w, theta

def orientation_error2(R_i, R_goal):
    R_e = R_i.T.dot(R_goal)  # error rotation

    n_e, theta_e = SO3_2_so3(R_e)
    return n_e*theta_e

def orientation_error(R_i, R_goal):
    R_e = R_i.T.dot(R_goal)  # error rotation

    w_e = logm(R_e)
    wx = w_e[-1, 1]
    wy = w_e[0, -1]
    wz = w_e[1, 0]
    n_e = np.array([wx, wy, wz]).reshape(3,1)
    return n_e

def IK(pos_goal, R_goal, q_joints_state, max_steps=1000, verbose=False):
    """ Calculates the IK from a certain joint configuration.

    Args:
        pos_goal ([np.array]): Goal position [x, y, z]
        R_goal ([np.array]): [nx, ny, nz] unit vector showing the goal orientaiton of TCP in base frame
        q_joints_state ([np.array]): [q1, q2, q3, q4, q5, q6, q7] actual joint conifguration
    """
    # max_steps = 1000
    v_step_size = 0.05  # 5mm
    vn_step_size = 0.0031  # 0.01 rad ~ 2grad
    theta_max_step = 0.04
    Q_j = q_joints_state.reshape(-1, 1)  # Array containing the starting joint angles
    T_j = FK(*Q_j.copy())  # x, y, z, nx, ny, nz coordinate of the position of the end effector in the base frame

    p_j = T_j[:3, 3:]
    R_j = T_j[:3, :3]
    delta_p = pos_goal - p_j  # delta_x, delta_y, delta_z between start position and desired final position of end effector
    delta_n = orientation_error(R_j, R_goal)  # delta_nx, delta_ny, delta_nz between start position and desired final position of end effector
    j = 0  # Initialize the counter variable

    # While the magnitude of the delta_p vector is greater than 0.01
    border_p = 0.008  # 1cm
    border_n = 0.01
    # and we are less than the max number of steps
    not_converged = np.linalg.norm(delta_p) > border_p or np.linalg.norm(delta_n) > border_n
    while not_converged and j<max_steps:
        # print('j{}:\n Q[{}],\n P[{}],\n O[{}]'.format(j, Q_j.T, p_j.T, theta_err))  # Print the current joint angles and position of the end effector in the global frame
        #  or np.linalg.norm(delta_n) > border_n
        if j%20 == 0 and verbose:
            print('j{}:'.format(j))
            print('x_norm={},  {}'.format(delta_p.T,  np.linalg.norm(delta_p)))
            print('n_norm={},  {}'.format(delta_n.T,  np.linalg.norm(delta_n)))
        # Reduce the delta_p 3-element delta_p vector by some scaling factor
        # delta_p represents the distance between where the end effector is now and our goal position.
        # v_p = np.nan_to_num(delta_p * v_step_size / np.linalg.norm(delta_p))
        # v_n = np.nan_to_num(delta_n * vn_step_size / np.linalg.norm(delta_n+1e-6))

        v_n = delta_n
        v_p = delta_p
        # Get the jacobian matrix given the current joint angles
        # np.block([[R_j.T, np.zeros_like(R_j)],[np.zeros_like(R_j), R_j.T]])
        J_j = J(*Q_j.copy())  # if we don't give the transpose qi=[qi] so all qi_s are 1x1 matrixes

        # Calculate the pseudo-inverse of the Jacobian matrix
        J_invj = np.linalg.pinv(J_j)
        # J_invj = np.linalg.pinv(J_j[:3,:])
        # J_invj = np.linalg.pinv(J_j[3:,:])
        # J_invj  = J_j[:3,:].T.dot( np.linalg.inv(J_j[:3,:].dot(J_j[:3,:].T)) )

        # Multiply the two matrices together
        v_Q = np.matmul(J_invj, np.vstack([0.0*v_p, v_n]))
        # v_Q = np.matmul(J_invj, v_p)
        # v_Q = np.matmul(J_invj, v_n)

        # Move the joints to new angles
        # We use the np.clip method here so that the joint doesn't move too much. We
        # just want the joints to move a tiny amount at each time step because
        # the full motion of the end effector is nonlinear, and we're approximating the
        # big nonlinear motion of the end effector as a bunch of tiny linear motions.
        # Q_j = Q_j + np.clip(v_Q, -1*theta_max_step, theta_max_step)  # [:self.N_joints]
        Q_j = Q_j + v_Q * 0.006  # [:self.N_joints]

        # Get the current position of the end-effector in the global frame
        T_j = FK(*Q_j.copy())
        p_j = T_j[:3, 3:]
        R_j = T_j[:3, :3]

        # Increment the time step
        j = j + 1

        # Determine the difference between the new position and the desired end position
        delta_p = pos_goal - p_j
        delta_n = orientation_error2(R_j, R_goal)

        # print("THETA: {}  {}".format(np.linalg.norm(delta_n), delta_p.T)
        pos_not_converged = np.linalg.norm(delta_p) > border_p
        orient_not_converged = np.linalg.norm(delta_n) > border_n
        not_converged = pos_not_converged or orient_not_converged
    if verbose:
        print('j{}:\n x_norm[{}],\n n_norm[{}]'.format(j, np.linalg.norm(delta_p), np.linalg.norm(delta_n)))
    if not_converged:
        print("IK did not converge delta_n = {}, {}".format(delta_n.T, np.linalg.norm(delta_n)))
    # Return the final angles for each joint
    return Q_j

def CartesianMinJerk2JointSpace(position_traj, thetas, q_joints_state_start):
    """ Calculates the joint space trajectories that correspond to cartesian trajectories

    Args:
        position_traj ([np.array(dtype='float')]): [N,3] a xyz relative trajectory of length N from the start position specified by q_joints_state_start
        thetas ([np.array(dtype='float')]): [N,] describing the orientation of the cup in the 2D plane(deviation from upright position)
                                            z_cup = [0, -1, 0] is constant as we want to rotate only in the 2D plane
                                            R = [[ s, 0, c],
                                                 [ 0,-1, 0],
                                                 [-c, 0, s]]
        q_joints_state_start ([type]): starting joint configuration which should correspond to the first pose
    """
    # 1. Make sure the home position of the hand matches the start theta of the trajectory
    T_start = FK(*q_joints_state_start.copy())
    s = sin(thetas[0]); c = cos(thetas[0])
    R_goal_start = np.array([[ s,   c,  0.0,],
                             [0.0, 0.0, 1.0,],
                             [-c,   s,  0.0,]], dtype='float')
    assert np.allclose(T_start[:3, :3], R_goal_start), "Home position must match the start theta of the trajectory"
    assert np.allclose(T_start[:3, -1], position_traj[0,:]), "Home position must match the start position of the trajectory"
    ## If needed to corrects
    ## a. Update joint_start so that the home position/orientation matches the first position/orientation of the trajectory
    # q_joints_state_start = IK(T_start[:3, 3:], R_goal_start, q_joints_state_start.copy())

    N = position_traj.shape[0]
    joints_traj = np.zeros((N, q_joints_state_start.size))
    joints_traj[0] = q_joints_state_start


    q_joint_state_i = q_joints_state_start.copy()
    for i, (pos_goal_i, theta_i) in enumerate(zip(position_traj[1:], thetas[1:])):
        s = sin(theta_i); c = cos(theta_i)
        R_goal_i = np.array([[ s,   c,  0.0,],
                             [0.0, 0.0, 1.0,],
                             [-c,   s,  0.0,]], dtype='float')
        q_joint_state_i = IK(pos_goal_i.reshape(3,1), R_goal_i, q_joint_state_i.copy(), verbose=False)
        joints_traj[i+1,:] = q_joint_state_i.T
    return joints_traj



# %%
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    home_pose = np.array([np.pi/4, 0.0, 0.0, np.pi/4, np.pi/2, np.pi/2, -np.pi/2])
    # print(FK(*home_pose))
    # print(FK_DH(home_pose))



    # q_joints_goal = np.array([0.2, 1.21, 1.1, pi_2/2, 1.1, 0.2, 0.2]).reshape(-1, 1)
    # T_goal = FK(*q_joints_goal)
    # q_joints_state = np.array([0, 1.0, 0, pi_2/2, 0, 0, 0]).reshape(-1, 1)
    # pos_goal = T_goal[:3, 3:]
    # R_goal = T_goal[:3,:3]

    # print(pos_goal)
    # q = IK(pos_goal, R_goal, q_joints_state)
    # print(T_goal)
    # print(FK(*q))

    # home_new = np.array([0.0, -1.0, 3.0, -np.pi/4, -np.pi/4, np.pi/2, 0.0])
    home_new = np.array([np.pi/4, -0.4, 0.0, np.pi/4, np.pi/2, np.pi/2, -np.pi/2])
    T = FK(*home_new)
    print(T)
    pos_goal = T[:3, 3:]
    R = T[:3, :3]
    R[0,0:2] = 0.0
    R[:,0] /= np.linalg.norm(R[:,0])
    R[:,1] /= np.linalg.norm(R[:,1])
    R[:,2]  = np.array([1.0,  0.0,  0.0], dtype='float').T
    print(R, np.linalg.norm(R[:,0]), np.linalg.norm(R[:,1]))

    R_goal = np.array([ [0.0,  0.0,  1.0],
                        [0.0,  1.0,  0.0],
                        [-1.0, 0.0,  0.0]], dtype='float')
    print(IK(pos_goal, R, home_new, max_steps=10000, verbose=True))