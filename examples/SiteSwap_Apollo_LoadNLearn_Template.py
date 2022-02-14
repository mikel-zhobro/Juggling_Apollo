import numpy as np
import time
import matplotlib.pyplot as plt

import __add_path__
from ApolloILC.settings import dt
from ApolloILC.ILC import ILC
from ApolloILC.DynamicSystem import ApolloDynSys
from ApolloPlanners import SiteSwapPlanner
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloKinematics.ApolloKinematics import ApolloArmKinematics
from ApolloKinematics import utilities
from utils import plot_A, save, load, print_info, plot_info

np.set_printoptions(precision=4, suppress=True)


# PARAMS
print("juggling_apollo")
########################################################################################################
########################################################################################################
########################################################################################################
FREQ_DOMAIN = True
NF=18

SAVING = True
UB = 3.87

CARTESIAN_ERROR = False
NOISE=0.0


ILC_it = 1                                # number of ILC iteration
end_repeat = 0  if not FREQ_DOMAIN else 0 # repeat the last position value this many time

FILENAME= "examples/data/AllJoints3/2021_12_09-16_59_04siteswap_joint_[0, 1, 2, 3, 5]_alpha_[18. 18. 18. 18. 18. 18. 18.]_eps_0.001_freq_domain_cart_err_off"
ld = load(FILENAME)


# Learnable Joints
N_joints = 7
learnable_joints = ld.learnable_joints
non_learnable_joints = list(set(range(N_joints)) - set(learnable_joints))

# ILC Params
n_ms  = ld.n_ms
n_ds  = ld.n_ds
ep_s  = ld.ep_s
alpha = ld.alpha
syss  = [ApolloDynSys(dt, x0=np.zeros((2,1)), alpha_=a, freq_domain=FREQ_DOMAIN) for a in alpha]

# Cartesian Error propogation params
# rArmKinematics_nn:  kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
# rArmKinematics:     kinematics with noise (used for its (wrong)IK calculations)
damp            = 1e-12
mu              = 0.35
CARTESIAN_ERROR = False



# 0. Create Apollo objects
T_home = ld.T_home
T_dhtcp_tcp = np.eye(4)
T_dhtcp_tcp[:3,:3] = T_home[:3,:3].T
########################################################################################################
########################################################################################################
########################################################################################################

# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=NOISE)  ## kinematics with noise (used for its (wrong)IK calculations)
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)               ## kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)

# 1. Trajectory Planning
jp = SiteSwapPlanner.JugglingPlanner()
pattern=(1,); h=0.3; r_dwell=0.6; throw_height=0.15; swing_size=0.15; w=0.3; slower=1.0; rep=1
plan = jp.plan(dt, 1, pattern=ld.pattern, h=ld.h, r_dwell=ld.r_dwell, throw_height=ld.throw_height,
               swing_size=ld.swing_size, w=ld.w, slower=ld.slower, rep=ld.rep)
N, x0, v0, a0, j0, rot_traj_des = plan.hands[0].get(get_thetas=True)  # get plan for hand0

# Transform to dh frames and do IK
T_traj = utilities.pR2T(x0-x0[0]+T_home[:3, -1], rot_traj_des)
T_traj = rArmKinematics.transform_in_dh_frames(T_dhtcp_tcp, T_traj)
T_home = T_traj[0].copy()
q_traj_des, q_start, psi_params = rArmKinematics.seqIK(T_traj, considered_joints=learnable_joints)  # [N, 7]
# Remove trajectories we are not learning
q_traj_des[:,non_learnable_joints] = 0


print("T_Home\n",T_traj[0])
print("FLIGHT TIMES: ", plan.getFlightTimes())
print("1.Start: ", q_start.T)
print()
plan.plot()
if False:
  aa = plan.getBallNPlatte()
  rArmKinematics.plot(q_traj_des, *psi_params)

# INIT ILC
# ILC Works on differences(ie delta)
q_traj_des_i       = q_traj_des.copy()
q_start_i          = q_start.copy()
# q_start_i = q_traj_des_i[1]; #q_start_i[-1,0] = np.pi/4.
delta_q_traj_des_i = q_traj_des_i[1:] - q_traj_des_i[1]


# B. Initialize ILC
Nf = NF if FREQ_DOMAIN else N-1
def kf_params(n_m=0.02, epsilon=1e-5, n_d=0.06, d0=None, P0=None):
  P0 = n_d*np.ones(Nf, dtype='float') if P0 is None else P0
  d0 = np.zeros((Nf, 1), dtype='float') if d0 is None else d0
  kf_dpn_params = {
    'M': n_m*np.ones(Nf, dtype='float'),   # covariance of noise on the measurment
    'P0': P0,                              # initial disturbance covariance
    'd0': d0,                              # initial disturbance value
    'epsilon0': epsilon,                   # covariance of noise on the disturbance
    'epsilon_decrease_rate': 0.95          # the decreasing factor of noise on the disturbance
  }
  return kf_dpn_params


my_ilcs = [
  ILC(sys=syss[i], y_des=delta_q_traj_des_i[:, i], kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i], *ld.ilc_learned_params[i]), freq_domain=FREQ_DOMAIN, Nf=Nf)                # make sure to make up for the initial state during learning
  for i in range(N_joints)]




# C. LEARN BY ITERATING
T_FULL = N*dt - 0.002
N_1 = N-1
# Data collection
q_traj_des_vec        = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# a. Measurments
joints_q_vec          = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_vq_vec         = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joints_aq_vec         = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
joint_torque_vec      = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
# b. ILC Trajectories
u_ff_vec              = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
disturbanc_vec        = np.zeros([ILC_it, Nf, N_joints], dtype='float')
# c. Trajectory errors
cartesian_error_norms = np.zeros([ILC_it, 6, 1], dtype='float')  # (x,y,z,nx,ny,nz)
joint_error_norms     = np.zeros([ILC_it, N_joints, 1], dtype='float')
joints_d_vec          = np.zeros([ILC_it, N_1, N_joints, 1], dtype='float')
d_xyz_rpy_vec         = np.zeros([ILC_it, N, 6], dtype='float')






# D. Main Loop
every_N = 5

# Use linear model to compute first input
u_ff   = np.zeros([N-1, N_joints, 1], dtype='float')
for i in learnable_joints:
  u_ff[:,i] = my_ilcs[i].init_uff_from_lin_model()

for j in range(ILC_it):
  # Limit Input
  u_ff = np.clip(u_ff,-UB,UB)
  if True and j%every_N==0: plot_info(dt, j, learnable_joints,
                             joints_q_vec, q_traj_des_i[1:],
                            #  u_ff_vec, q_v_traj[1:],
                             joint_torque_vec,
                            #  disturbanc_vec,
                            #  d_xyz,
                             joint_error_norms, cartesian_error_norms,
                             v=False, p=True, dp=False, e_xyz=False, e=True, torque=False)

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
  delta = np.array([utilities.errorForJacobianInverse(T_i=rArmKinematics_nn.FK(q0), T_goal=T_traj[0])]+
                   [utilities.errorForJacobianInverse(T_i=cartesian_traj_i[i], T_goal=T_traj[i+1]) for i in range(N_1)]).reshape(-1, 6)
  d_xyz = delta[:, :3]  # measured cartesian error: calculated using the noise-less FK

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
  d_xyz_rpy_vec[j]      = delta   # actual cartesian errors
  joints_d_vec[j]       = delta_q_traj_des_i-delta_y_meas         # actual joint space error
  joint_error_norms[j]  = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T / N_1
  cartesian_error_norms[j]  = np.linalg.norm(delta, axis=0, keepdims=True).T / N_1


  # For the next iteration
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
            disturbanc_vec=disturbanc_vec, d_xyz_rpy_vec=d_xyz_rpy_vec, joint_error_norms=joint_error_norms,
            v=True, p=True, dp=False, e_xyz=True, e=True, N=min(4, ILC_it-1))

if SAVING:
  # Saving Results
  freq = 'freq_domain' if FREQ_DOMAIN else "time_domain"
  cartesian_err = 'cart_err_on' if CARTESIAN_ERROR else "cart_err_off"
  filename = "examples/data/AllJoints3/" + time.strftime("%Y_%m_%d-%H_%M_%S") + "siteswap_joint_{}_alpha_{}_eps_{}_{}_{}".format(learnable_joints, alpha, ep_s[0], freq, cartesian_err)
  save(filename,
       dt=dt,
       q_start=q_start_i, T_home=T_home,                                                 # Home
       T_traj=T_traj, q_traj_des_vec=q_traj_des_vec,                                     # Desired Trajectories
       joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
       joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
       disturbanc_vec=disturbanc_vec, u_ff_vec=u_ff_vec,                                 # Learned Trajectories (uff and disturbance)
       d_xyz_rpy_vec=d_xyz_rpy_vec, joints_d_vec=joints_d_vec,                                   # Progress Measurments
       joint_error_norms=joint_error_norms, cartesian_error_norms=cartesian_error_norms,
       pattern=pattern, h=h, r_dwell=r_dwell, throw_height=throw_height,                 # SiteSwap Params
       swing_size=swing_size, w=w, slower=slower, rep=rep,
       ilc_learned_params = [(ilc.d, ilc.P) for ilc in my_ilcs],
       learnable_joints=learnable_joints, alpha=alpha, n_ms=n_ms, n_ds=n_ds, ep_s=ep_s)  # ILC parameters

  with open('examples/data/AllJoints3/list_files_all.txt', 'a') as f:
      f.write(filename + "\n")



# Run Simulation with several repetition
rArmInterface.apollo_run_one_iteration2(dt=dt, T=end_repeat*dt, u=u_ff[:end_repeat], joint_home_config=q_start_i, repetitions=1, it=j)
rArmInterface.apollo_run_one_iteration2(dt=dt, T=T_FULL-end_repeat*dt, u=u_ff[end_repeat:], repetitions=5, it=j)
