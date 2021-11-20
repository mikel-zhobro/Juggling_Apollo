from operator import index
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
from apollo_interface.Apollo_It import ApolloInterface, plot_simulation
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, save, load, print_info, plot_info

np.set_printoptions(precision=4, suppress=True)

end_repeat = 200   # repeat the last position value this many time
SAVING = True
UB = 0.87

FILENAME = "joint_[0, 1, 2, 3, 4, 5, 6]_alpha_[16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]_eps_0.0001_no_knowledge2021_11_20-17_25_21"
ld = load('examples/data/AllJoints3/'+FILENAME)


# Learnable Joints
learnable_joints = ld.learnable_joints

# ILC Params
n_ms  = ld.n_ms
n_ds  = ld.n_ds
ep_s  = ld.ep_s
alpha = ld.alpha
syss  = [ApolloDynSys2(dt, alpha_=a) for a in ld.alpha]

# Cartesian Error propogation params
# rArmKinematics_nn:  kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
# rArmKinematics:     kinematics with noise (used for its (wrong)IK calculations)
damp            = 1e-12
mu              = 5e-2
CARTESIAN_ERROR = False



# 0. Create Apollo objects
print("juggling_apollo")
T_home = ld.T_home
# a) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)
# b) KINEMATICS: create rArmInterface and go to home position
rArmKinematics    = ApolloArmKinematics(r_arm=True, noise=0.0)  ## noise noisifies the forward dynamics only





# A. COMPUTE TRAJECTORIES IN CARTESIAN AND JOINT SPACE
N_1, delta_xyz_traj_des, thetas, mj = configs.get_minjerk_config(dt, end_repeat)
xyz_traj_des = ld.xyz_traj_des
q_traj_des = ld.q_traj_des
q_start = q_traj_des[0]

# Set to 0 the non learning joints
for i in set(range(7)) - set(learnable_joints):
  q_traj_des[:,i] = 0.0
 
if False:
  rArmKinematics.plot(q_traj_des, *psi_params)






# B. Initialize ILC
def kf_params(n_m=0.02, epsilon=1e-5, n_d=0.06, d0=None, P0=None):
  
  d0 = np.zeros((N_1+end_repeat, 1), dtype='float') if d0 is None else d0
  P0 = n_d*np.eye(N_1+end_repeat, dtype='float') if P0 is None else P0
  kf_dpn_params = {
    'M': n_m*np.eye(N_1+end_repeat, dtype='float'),   # covariance of noise on the measurment
    'P0': P0,                                         # initial disturbance covariance
    'd0': d0,                                         # initial disturbance value
    'epsilon0': epsilon,                              # initial variance of noise on the disturbance
    'epsilon_decrease_rate': 0.9                      # the decreasing factor of noise on the disturbance
  }
  return kf_dpn_params

# Initialize ILC from learned ilcparams
my_ilcs = [
  ILC(dt=dt, sys=syss[i], kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i], *ld.ilc_learned_params[i]), x_0=[0.0, 0.0])                # make sure to make up for the initial state during learning
  for i in range(N_joints)]




# C. LEARN BY ITERATING
ILC_it = 1  # number of ILC iteration
# Data collection
q_traj_des_vec = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
# a. Measurments
joints_q_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
joints_vq_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
joints_aq_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
joint_torque_vec     = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
# b. ILC Trajectories
u_ff_vec       = np.zeros([ILC_it, N_1+end_repeat, N_joints, 1], dtype='float')
disturbanc_vec = np.zeros([ILC_it, N_1+end_repeat, N_joints], dtype='float')
# c. Trajectory errors
error_norms    = np.zeros([ILC_it, N_joints, 1], dtype='float')
joints_d_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
d_xyz_vec      = np.zeros([ILC_it, N_1+1+end_repeat, 3], dtype='float')







# D. Main Loop
every_N = 2
# ILC Works on differences(ie delta)
# Changing (possibly wrong/noisy) desired trajectory to make up for kinematics errors
q_traj_des_i       = q_traj_des.copy()   
q_start_i          = q_start.copy()
delta_q_traj_des_i = q_traj_des_i - q_start_i

# Use linear model to compute first input
u_ff   = ld.u_ff_vec[-1]  # load last learned uff
for i in learnable_joints:
  u_ff[:,i] = my_ilcs[i].ff_from_lin_model(y_des=delta_q_traj_des_i[1:, i])

for j in range(ILC_it):  
  # Limit Input
  # u_ff = np.clip(u_ff,-UB,UB)

  # Main Simulation
  q_traj, q_v_traj, q_a_traj, F_N_vec, _ = rArmInterface.apollo_run_one_iteration(dt=dt, T=mj.T_FULL+end_repeat*dt, u=u_ff, joint_home_config=q_start_i, repetitions=1, it=j)
  delta_y_meas = q_traj - q_traj[0] # calc delta of executed traj
  q_traj = q_start_i + delta_y_meas   # rebase executed traj as if it would start from the exact home position

  # Update feed-forward signal
  for i in learnable_joints:
    u_ff[:,i] = my_ilcs[i].learnWhole(u_ff_old=u_ff[:, i], y_des=delta_q_traj_des_i[1:, i], y_meas=delta_y_meas[1:,i],
                                      #  lb=-UB,ub=UB
                                      verbose=False)

  # Collect Data
  d_xyz = xyz_traj_des - rArmKinematics.seqFK(q_traj)[:, :3, -1]  # measured cartesian error: calculated using the noise-less FK
  # a. Meas
  joints_q_vec[j]       = q_traj
  joints_vq_vec[j]      = q_v_traj
  joints_aq_vec[j]      = q_a_traj
  joint_torque_vec[j]   = F_N_vec
  # b. ILC
  u_ff_vec[j]           = u_ff
  disturbanc_vec[j]     = np.squeeze([ilc.d for ilc in my_ilcs]).T  # learned joint space disturbances
  q_traj_des_vec[j]     = q_traj_des_i + q_start_i
  # c. Errors
  d_xyz_vec[j]          = d_xyz   # actual cartesian errors
  joints_d_vec[j]       = delta_q_traj_des_i-delta_y_meas         # actual joint space error
  error_norms[j]        = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T


  # For the next iteration
  # TODO: add orientation error
  if CARTESIAN_ERROR:
    for i in range(N_1):
      J_invj          = np.linalg.pinv(rArmKinematics.J(q_traj[i+1])[:3,:])
      q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(d_xyz[i].reshape(3, 1))
    # Update desired trajectory
    q_start_i = q_traj_des_i[0]
    delta_q_traj_des_i = q_traj_des_i - q_start_i



  print_info(j, learnable_joints, joints_d_vec, d_xyz)

  if False and j%every_N==0: plot_info(dt, j, learnable_joints, 
                                       joints_q_vec, q_traj_des_i, 
                                       u_ff_vec, q_v_traj[1:], 
                                       joint_torque_vec,
                                       disturbanc_vec, d_xyz, error_norms,
                                       v=True, p=True, dp=True, e_xyz=False, e=False, torque=False)

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
            joints_q_vec=joints_q_vec, q_traj_des=q_traj_des_i, 
            u_ff_vec=u_ff_vec, q_v_traj=q_v_traj[1:,], 
            disturbanc_vec=disturbanc_vec, d_xyz=d_xyz, error_norms=error_norms,
            v=True, p=True, dp=False, e_xyz=True, e=True, N=1)

if SAVING:
  # Saving Results
  filename = "examples/data/AllJoints3/joint_{}_alpha_{}_eps_{}_no_knowledge".format(learnable_joints, alpha, ep_s[0]) + time.strftime("%Y_%m_%d-%H_%M_%S")
  save(filename,
        q_start=q_start_i, T_home=T_home,                                                 # Home
        xyz_traj_des=xyz_traj_des, q_traj_des=q_traj_des_i,                               # Desired Trajectories
        joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
        joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
        disturbanc_vec=disturbanc_vec, u_ff_vec=u_ff_vec,                                 # Learned Trajectories (uff and disturbance)
        d_xyz_vec=d_xyz_vec, joints_d_vec=joints_d_vec, error_norms=error_norms,          # Progress Measurments
        mj=mj,                                                                            # Minjerk Params
        ilc_learned_params = [(ilc.d, ilc.P) for ilc in my_ilcs],
        learnable_joints=learnable_joints, alpha=alpha, n_ms=n_ms, n_ds=n_ds, ep_s=ep_s)  # ILC parameters

  with open('examples/data/AllJoints3/list_files_all.txt', 'a') as f:
      f.write(filename + "\n")



# Run Simulation with several repetition
# rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_ff[:-end_repeat], joint_home_config=q_start_i, repetitions=25, it=j)
rArmInterface.apollo_run_one_iteration(dt=dt, T=end_repeat*dt, u=u_ff[:end_repeat], joint_home_config=q_start_i, repetitions=1, it=j)
rArmInterface.apollo_run_one_iteration(dt=dt, T=mj.T_FULL, u=u_ff[end_repeat:], repetitions=15, it=j)
# rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_ff[:-end_repeat], joint_home_config=q_start_i, repetitions=25, it=j)
