import numpy as np
import time

import __add_path__
import configs
from juggling_apollo.utils import steps_from_time, plt
from juggling_apollo.JugglingPlanner import calc
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.MinJerk import plotMJ, get_minjerk_trajectory
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys2
from juggling_apollo import SiteSwapPlanner
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from kinematics import utilities
from utils import plot_A, save, colors, line_types, print_info, plot_info

np.set_printoptions(precision=4, suppress=True)

# PARAMS
print("juggling_apollo")
########################################################################################################
########################################################################################################
########################################################################################################
FREQ_DOMAIN=True
NF=15

SAVING = False
UB = 0.87

CARTESIAN_ERROR = False
NOISE=0.0

ILC_it = 8  # number of ILC iteration
end_repeat = 0  if not FREQ_DOMAIN else 0 # repeat the last position value this many time

# Learnable Joints
learnable_joints = [0,1,2,3,4,5,6]

# ILC Params
n_ms  = np.ones((7,))* 3e-5;   n_ms[3:] = 1e-5  # covariance of noise on the measurment
n_ds  = [1e-2]*7    # initial disturbance covariance
ep_s  = [1e-3]*7    # covariance of noise on the disturbance
alpha = np.ones((7,)) * 18.0; alpha[3:] = 90.0; alpha[0:] = 33.0
syss  = [ApolloDynSys2(dt, x0=np.zeros((2,1)), alpha_=a, freq_domain=FREQ_DOMAIN) for a in alpha]

# Cartesian Error propogation params
damp            = 1e-12
mu              = 0.35

# Home Configuration
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

R_dhtcp_tcp = np.array([[0.0, -1.0, 0.0],
                        [0.0,  0.0, 1.0],
                        [-1.0, 0.0, 0.0]]).T
########################################################################################################
########################################################################################################
########################################################################################################

# A. COMPUTE TRAJECTORIES IN CARTESIAN AND JOINT SPACE
jp = SiteSwapPlanner.JugglingPlanner(w=0.7, D=0.152)
plan = jp.plan(1, pattern=(3,), rep=1)
N, x0, v0, a0, j0, thetas = plan.hands[0].get(get_thetas=True)  # get plan for hand0

# From direction vector to theta
# thetas = R_dhtcp_tcp.dot(thetas.T).T

#- thetas[0] + np.array([0,0,1.]).reshape(1,3)
print(thetas[-1,2])
thetas = np.arccos(np.clip(thetas[:, 2], -1.0, 1.0))
thetas2 = np.arccos(np.clip(-(v0/ np.linalg.norm(v0, axis=1, keepdims=True))[:, 2], -1.0, 1.0)) - thetas[0]
thetas -= thetas[0]
plt.plot(thetas*180./np.pi)
plt.plot(thetas2*180./np.pi)
# plt.show()

plan.plot()



# From positions get delta position
delta_xyz_traj_des = (x0 - x0[0])
# delta_xyz_traj_des[:,[0, 1]] = delta_xyz_traj_des[:,[1, 0]]
xyz_traj_des = delta_xyz_traj_des + T_home[:3, -1]
print(np.max(delta_xyz_traj_des, 0) - np.min(delta_xyz_traj_des, 0))


T_FULL = N*dt - 0.002
N_1 = N-1

# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=NOISE)  ## kinematics with noise (used for its (wrong)IK calculations)
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)               ## kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)



cartesian_traj_des, q_traj_des, q_start, psi_params = rArmKinematics.seqIK(delta_xyz_traj_des, thetas, T_home, considered_joints=learnable_joints[:])  # [N, 7]
# cartesian_traj_des, q_traj_des_nn, q_start_nn, _    = rArmKinematics_nn.seqIK(delta_xyz_traj_des, thetas, T_home)  # [N, 7]


# Set to 0 the non learning joints
for i in set(range(7)) - set(learnable_joints):
  q_traj_des[:,i] = 0.0
  # q_traj_des_nn[:,i] = 0.0


if False:
  rArmKinematics.plot(q_traj_des, *psi_params)

Nf = N-1
if FREQ_DOMAIN:
  Nf = NF
# B. Initialize ILC
def kf_params(n_m=0.02, epsilon=1e-5, n_d=0.06):
  kf_dpn_params = {
    'M': n_m*np.ones(Nf, dtype='float'),       # covariance of noise on the measurment
    'P0': n_d*np.ones(Nf, dtype='float'),      # initial disturbance covariance
    'd0': np.zeros((Nf, 1), dtype='float'),    # initial disturbance value
    'epsilon0': epsilon,                       # covariance of noise on the disturbance
    'epsilon_decrease_rate': 0.95               # the decreasing factor of noise on the disturbance
  }
  return kf_dpn_params


# ILC Works on differences(ie delta)
# Changing (possibly wrong/noisy) desired trajectory to make up for kinematics errors
q_traj_des_i       = q_traj_des.copy()
# q_start_i          = q_start.copy()
q_start_i = q_traj_des_i[1]; #q_start_i[-1,0] = np.pi/4.
print("START", q_start_i.shape, q_start_i.T)
delta_q_traj_des_i = q_traj_des_i[1:] - q_start_i

my_ilcs = [
  ILC(sys=syss[i], y_des=delta_q_traj_des_i[:, i], kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i]), freq_domain=FREQ_DOMAIN, Nf=Nf)                # make sure to make up for the initial state during learning
  for i in range(N_joints)]



# C. LEARN BY ITERATING
# Data collection
q_traj_des_vec = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# a. Measurments
joints_q_vec   = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_vq_vec  = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_aq_vec  = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joint_torque_vec     = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# b. ILC Trajectories
u_ff_vec       = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
disturbanc_vec = np.zeros([ILC_it, Nf, N_joints], dtype='float')
# c. Trajectory errors
cartesian_error_norms    = np.zeros([ILC_it, 6, 1], dtype='float')  # (x,y,z,nx,ny,nz)
joint_error_norms    = np.zeros([ILC_it, N_joints, 1], dtype='float')
joints_d_vec   = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
d_xyz_vec      = np.zeros([ILC_it, N, 3], dtype='float')





# D. Main Loop
every_N = 5


# Use linear model to compute first input
u_ff   = np.zeros([N-1, N_joints, 1], dtype='float')
for i in learnable_joints:
  u_ff[:,i] = my_ilcs[i].init_uff_from_lin_model()

for j in range(ILC_it):
  # Limit Input
  # u_ff = np.clip(u_ff,-UB,UB)

  # Main Simulation
  q_traj, q_v_traj, q_a_traj, F_N_vec, _, q0 = rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL, u=u_ff, joint_home_config=q_start_i, repetitions=3 if FREQ_DOMAIN else 1, it=j)
  q_traj   = np.average(  q_traj[0:], axis=0)
  q_v_traj = np.average(q_v_traj[0:], axis=0)
  q_a_traj = np.average(q_a_traj[0:], axis=0)
  F_N_vec  = np.average( F_N_vec[0:], axis=0)
  
  delta_y_meas = q_traj - q_traj[0]  #  np.average(q0[0,], axis=0) #np.average(q0[0:], axis=0, weights=[3,2,1,1]) # calc delta of executed traj
  q_traj = q_start_i + delta_y_meas   # rebase executed traj as if it would start from the exact home position

  # Update feed-forward signal
  for i in learnable_joints:
    if FREQ_DOMAIN:
      u_ff[:,i] = my_ilcs[i].updateStep2(y_meas=delta_y_meas[:,i],  y_des=delta_q_traj_des_i[:, i], 
                                        #  lb=-UB,ub=UB
                                        verbose=False)
    else:
      u_ff[:,i] = my_ilcs[i].updateStep(y_meas=delta_y_meas[:,i],  y_des=delta_q_traj_des_i[:, i], 
                                        #  lb=-UB,ub=UB
                                        verbose=False)

  # Collect Data
  cartesian_traj_i = rArmKinematics_nn.seqFK(q_traj)
  delta = np.array([utilities.errorForJacobianInverse(T_i=rArmKinematics_nn.FK(q0), T_goal=cartesian_traj_des[0])]+ 
                   [utilities.errorForJacobianInverse(T_i=cartesian_traj_i[i], T_goal=cartesian_traj_des[i+1]) for i in range(N_1)]).reshape(-1, 6)
  d_xyz = delta[:, :3]  # measured cartesian error: calculated using the noise-less FK
  # d_xyz = xyz_traj_des[1:] - rArmKinematics_nn.seqFK(q_traj)[:, :3, -1]  # measured cartesian error: calculated using the noise-less FK
  # a. Meas
  joints_q_vec[j]       = q_traj
  joints_vq_vec[j]      = q_v_traj
  joints_aq_vec[j]      = q_a_traj
  joint_torque_vec[j]   = F_N_vec
  # b. ILC
  u_ff_vec[j]           = u_ff
  disturbanc_vec[j]     = np.squeeze([ilc.d for ilc in my_ilcs]).T  # learned joint space disturbances
  q_traj_des_vec[j]     = q_traj_des_i[1:]
  # c. Errors
  d_xyz_vec[j]          = d_xyz   # actual cartesian errors
  joints_d_vec[j]       = delta_q_traj_des_i-delta_y_meas         # actual joint space error
  joint_error_norms[j]  = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T
  cartesian_error_norms[j]  = np.linalg.norm(delta, axis=0, keepdims=True).T


  # For the next iteration
  # TODO: add orientation error
  if CARTESIAN_ERROR:
    for i in range(0, N):
      qi = q0 if i==0 else q_traj[i-1]
      di = delta[i].reshape(6,1)
      # J_invj          = np.linalg.pinv(rArmKinematics.J(qi)[:,:])
      Ji = rArmKinematics.J(qi)
      J_invj          = Ji.T.dot(np.linalg.inv(Ji.dot(Ji.T) + damp*np.eye(6)))
      q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(di[:])
    # Update desired trajectory
    q_start_i = q_traj_des_i[0]
    delta_q_traj_des_i = q_traj_des_i[1:] - q_start_i
    mu = 0.95*mu



  print_info(j, learnable_joints, joints_d_vec, d_xyz)

  if False and j%every_N==0: plot_info(dt, j, learnable_joints,
                             joints_q_vec, q_traj_des_i[1:],
                             u_ff_vec, q_v_traj[1:],
                             joint_torque_vec,
                             disturbanc_vec, d_xyz, joint_error_norms, cartesian_error_norms,
                             v=False, p=True, dp=False, e_xyz=False, e=True, torque=False)

  if False and j%every_N==0:  # How desired  trajectory changes
    plot_A([q_traj_des_nn, q_traj_des_vec[j], q_traj_des_vec[j-1], q_traj_des_vec[0]], learnable_joints, ["des", "it="+str(j), "it="+str(j-1), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
    plt.suptitle("Desired Joint Trajectories")
    plt.show(block=False)

  if False and j%every_N==0:  # Whether disturbance makes up for differences
    y_no_d_s = np.array([ilc.lss.GF.dot(u_ff[:, indx:indx+1]) + ilc.lss.Gd0  + q_start_i[indx] for indx, ilc in enumerate(my_ilcs)]).transpose(1,0,2) # predicted y
    y_with_d_s = np.array([ilc.lss.GF.dot(u_ff[:, indx:indx+1]) + ilc.lss.Gd0  + ilc.lss.GK.dot(ilc.d) + q_start_i[indx] for indx, ilc in enumerate(my_ilcs)]).transpose(1,0,2) # predicted y
    fillbetween = [y_no_d_s, y_with_d_s]
    plot_A([y_no_d_s, q_traj[1:]], labels=["aimed_trajectory", "realized_trajectory"], indexes_list=learnable_joints, fill_between=fillbetween)
    plt.show()








# After Main Loop has finished
if True:
  plot_info(dt, j, learnable_joints,
            joints_q_vec=joints_q_vec, q_traj_des=q_traj_des_i[1:],
            u_ff_vec=u_ff_vec, q_v_traj=q_v_traj, cartesian_error_norms = cartesian_error_norms,
            disturbanc_vec=disturbanc_vec, d_xyz=d_xyz, joint_error_norms=joint_error_norms,
            v=True, p=True, dp=False, e_xyz=True, e=True, N=min(4, ILC_it-1))

if SAVING:
  # Saving Results
  freq = 'freq_domain' if FREQ_DOMAIN else "time_domain"
  cartesian_err = 'cart_err_on' if CARTESIAN_ERROR else "cart_err_off"
  filename = "examples/data/AllJoints3/" + time.strftime("%Y_%m_%d-%H_%M_%S") + "joint_{}_alpha_{}_eps_{}_{}_{}".format(learnable_joints, alpha, ep_s[0], freq, cartesian_err)
  save(filename,
       dt=dt,
       q_start=q_start_i, T_home=T_home,                                                 # Home
       xyz_traj_des=xyz_traj_des, q_traj_des=q_traj_des_i,                               # Desired Trajectories
       joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
       joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
       disturbanc_vec=disturbanc_vec, u_ff_vec=u_ff_vec,                                 # Learned Trajectories (uff and disturbance)
       d_xyz_vec=d_xyz_vec, joints_d_vec=joints_d_vec,                                   # Progress Measurments
       joint_error_norms=joint_error_norms, cartesian_error_norms=cartesian_error_norms,
      #  mj=mj,                                                                            # Minjerk Params
       ilc_learned_params = [(ilc.d, ilc.P) for ilc in my_ilcs],
       learnable_joints=learnable_joints, alpha=alpha, n_ms=n_ms, n_ds=n_ds, ep_s=ep_s)  # ILC parameters

  with open('examples/data/AllJoints3/list_files_all.txt', 'a') as f:
      f.write(filename + "\n")



# Run Simulation with several repetition
rArmInterface.apollo_run_one_iteration2(dt=dt, T=end_repeat*dt, u=u_ff[:end_repeat], joint_home_config=q_start_i, repetitions=1, it=j)
rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL-end_repeat*dt, u=u_ff[end_repeat:], repetitions=5, it=j)
