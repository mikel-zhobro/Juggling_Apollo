import numpy as np
import time

import __add_path__
from juggling_apollo.utils import plt
from juggling_apollo.settings import dt
from juggling_apollo.ILC import ILC
from juggling_apollo.DynamicSystem import ApolloDynSys, ApolloDynSysIdeal, ApolloDynSys2
from apollo_interface.Apollo_It import ApolloInterface
from kinematics.ApolloKinematics import ApolloArmKinematics
from utils import plot_A, save, print_info, plot_info

np.set_printoptions(precision=4, suppress=True)


end_repeat = 14   # repeat the last position value this many time
SAVING = False

print("juggling_apollo")
T_home = np.array([[0.0, -1.0, 0.0,  0.32],  # uppword orientation(cup is up)
                   [0.0,  0.0, 1.0,  0.81],
                   [-1.0, 0.0, 0.0, -0.49],
                   [0.0,  0.0, 0.0,  0.0 ]], dtype='float')

# 0. Create Apollo objects
# A) INTERFACE: create rArmInterface and go to home position
N_joints = 7
rArmInterface = ApolloInterface(r_arm=True)

# B) KINEMATICS: create rArmInterface and go to home position
rArmKinematics = ApolloArmKinematics(r_arm=True, noise=0.0)  ## noise noisifies the forward dynamics only
rArmKinematics_nn = ApolloArmKinematics(r_arm=True)  ## noise noisifies the forward dynamics only
q_start = rArmKinematics.IK_best(T_home)


N_delay = 120
N_1 = 300 + N_delay
T_FULL=N_1*dt-0.00001


amplitudes = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape(7,1)
q_traj_des = np.zeros((N_1,7,1))
q_traj_des[N_delay:100+N_delay,:,0] = -np.arange(100).reshape(-1,1)/100.0
q_traj_des[100+N_delay:200+N_delay,:] = q_traj_des[99+N_delay]
q_traj_des[200+N_delay:,:,0] = -np.arange(99, -1, -1).reshape(-1,1)/100.0
q_traj_des *= amplitudes


repeat = [1]* N_1
repeat[-1] = end_repeat+2
q_traj_des = np.repeat(q_traj_des, repeats=repeat, axis=0)
q_traj_des += q_start

if False:
  print(q_traj_des. shape)
  plot_A([q_traj_des], dt=dt, limits=rArmKinematics.limits)
  plt.legend("Desired Velocity")
  plt.show()
# Cartesian -> JointSpace                   <------------------------------------------------------------------------------------------ Mean Jerk Trajectory (CARTESIAN AND JOINT SPACE)
xyz_traj_des = rArmKinematics.seqFK(q_traj_des)[:, :3, -1] - T_home[:3, -1]


# B. Initialize ILC
def kf_params(n_m=0.02, epsilon=1e-5, n_d=0.06):
  kf_dpn_params = {
    'M': n_m*np.eye(N_1+end_repeat, dtype='float'),       # covariance of noise on the measurment
    'P0': n_d*np.eye(N_1+end_repeat, dtype='float'),      # initial disturbance covariance
    'd0': np.zeros((N_1+end_repeat, 1), dtype='float'),   # initial disturbance value
    'epsilon0': epsilon,                                  # initial variance of noise on the disturbance
    'epsilon_decrease_rate': 0.9                          # the decreasing factor of noise on the disturbance
  }
  return kf_dpn_params

n_ms = [0.02]*7
n_ds = [6e-4]*7
ep_s = [1e-6]*7
# ep_s = [5e-3] * 7     # works well for velocity disturbance
# ep_s = [1e-2] * 7     # works well for velocity disturbance
n_ms = [3e-5]*7
n_ds = [6e-3]*7
ep_s = [1e-4]*7
alpha = 16.0
my_ilcs = [
  # ILC(dt=dt, sys=ApolloDynSys(dt, alpha_=alpha), kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i]), x_0=[q_start[i, 0], 0.0])    # include the initial state in the dynamics of the system
  ILC(dt=dt, sys=ApolloDynSys2(dt, alpha_=alpha), kf_dpn_params=kf_params(n_ms[i], ep_s[i], n_ds[i]), x_0=[0.0, 0.0])                # make sure to make up for the initial state during learning
  for i in range(N_joints)]

# C. LEARN BY ITERATING
# Learn Throw
ILC_it = 13  # number of ILC iteration

# Data collection
q_traj_des_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
# a. System Trajectories
joints_q_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float');  joints_q_vec[:,:] = q_start
joints_vq_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
joints_aq_vec  = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
xyz_vec        = np.zeros([ILC_it, N_1+1+end_repeat, 3], dtype='float')
# b. ILC Trajectories
disturbanc_vec = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
joints_d_vec   = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
u_ff_vec       = np.zeros([ILC_it, N_1+1+end_repeat, N_joints], dtype='float')
# c. Measurments
joint_torque_vec     = np.zeros([ILC_it, N_1+1+end_repeat, N_joints, 1], dtype='float')
# d. Trajectory error norms
error_norms    = np.zeros([ILC_it, N_joints, 1], dtype='float')

# ILC Vectors Init
u_ff = [None] * N_joints
y_meas = np.zeros((N_1+end_repeat, N_joints), dtype='float')

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

# Extra Loop (In case we want to try out smth on different combination of joints)
every_N = 3
## CHOOOSE JOINTS THAT LEARN
jjoint = "all"
# learnable_joints = [0,1,2,3,5]
# learnable_joints = [0,1,2,3,4,5]
learnable_joints = [5]
non_learnable_joints = set(range(7)) - set(learnable_joints)
for i in non_learnable_joints:
  q_traj_des[:,i] = 0.0
for ilc in my_ilcs:
  ilc.resetILC()
y_meas = np.zeros((N_1+end_repeat, N_joints), dtype='float')
u_ff = [None] * 7
eps = 1e-3
UB = 1.0-eps

# Main Loop
## Cartesian error params
# rArmKinematics_nn:  kinematics without noise  (used to calculate measurments, plays the wrole of a localization system)
# rArmKinematics:     kinematics with noise (used for its (wrong)IK calculations)
damp            = 1e-12
mu              = 1e-2
CARTESIAN_ERROR = False
q_traj_des_i    = q_traj_des.copy()   # Changing (possibly wrong/noisy) desired trajectory to make up for kinematics errors

for j in range(ILC_it):
  # Learn feed-forward signal
  # u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=q_traj_des_i[:, i], y_meas=y_meas[:, i],             # initial state considered in the dynamics
  # u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=q_traj_des_i[:, i] - q_start[i], y_meas=y_meas[:, i] - q_start[i],             # substract the initial state from the desired joint traj
  u_ff = [ilc.learnWhole(u_ff_old=u_ff[i], y_des=q_traj_des_i[:, i] - q_start[i], y_meas=y_meas[:, i] - y_meas[0,i],             # substract the initial state from the desired joint traj
                          verbose=False,  # bool(i in learnable_joints and j%every_N==0 and False),
                        #  lb=-0.8,ub=0.8
                          )
          for i, ilc in enumerate(my_ilcs)]
  u_arr = np.array(u_ff, dtype='float').squeeze().T
  for i in non_learnable_joints:
    u_arr[:,i] = 0.0

  # CLIP
  u_arr = np.clip(u_arr,-UB,UB)

  # plot_A([u_arr])
  # plt.suptitle("Velocity inputs")
  # plt.show()

  # Main Simulation
  q_traj, q_v_traj, q_a_traj, dP_N_vec, u_vec = rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL+end_repeat*dt, u=u_arr, joint_home_config=q_start, repetitions=1, it=j)

  # For the next iteration
  if CARTESIAN_ERROR:
    xyz_traj_meas = rArmKinematics_nn.seqFK(q_traj)[:, :3, -1]                    # actual cartesian errors
    d_xyz = xyz_traj_des + T_home[:3, -1] - xyz_traj_meas                         # [N_1, 3] d_xyz = xyz_des - xyz_i
    for i in range(N_1):
      J_invj = np.linalg.pinv(rArmKinematics.J(q_traj[i+1])[:3,:])                # q_traj[1:]  # N_1x7
      q_traj_des_i[i] = q_traj_des_i[i] + mu* J_invj.dot(d_xyz[i].reshape(3, 1))
    q_start = q_traj_des_i[0]


  # System Output
  y_meas           = q_traj[1:]
  d_xyz            = xyz_traj_des + T_home[:3, -1] - rArmKinematics_nn.seqFK(q_traj)[:, :3, -1]         # measured cartesian error: calculated using the noise-less FK
  d_xyz_best       = xyz_traj_des + T_home[:3, -1] - rArmKinematics_nn.seqFK(q_traj_des)[:, :3, -1]
  d_xyz_achievable = xyz_traj_des + T_home[:3, -1] - rArmKinematics.seqFK(q_traj_des)[:, :3, -1]

  # Collect Data
  q_traj_des_vec[j]     = q_traj_des_i
  joints_q_vec[j]       = q_traj
  joints_vq_vec[j]      = q_v_traj
  joints_aq_vec[j]      = q_a_traj
  u_ff_vec[j, :-1]      = u_arr
  joint_torque_vec[j]   = dP_N_vec
  disturbanc_vec[j, 1:] = np.squeeze([ilc.d for ilc in my_ilcs]).T  # learned joint space disturbances
  joints_d_vec[j, 1:]   = np.squeeze(q_traj_des[1:]-y_meas)         # actual joint space error
  xyz_vec[j]            = rArmKinematics.seqFK(q_traj)[:, :3, -1]   # actual cartesian errors
  error_norms[j]        = np.linalg.norm(joints_d_vec[j, :], axis=0, keepdims=True).T

  if False and j%every_N==0: plot_info(dt, j, learnable_joints,
                                      joints_q_vec, q_traj_des,
                                      u_ff_vec[:,:-1], q_v_traj[:,1:],
                                      joint_torque_vec,
                                      disturbanc_vec, d_xyz, error_norms,
                                      v=True, p=True, dp=False, e_xyz=False, e=False, torque=False)
  print_info(j, learnable_joints, joints_d_vec, d_xyz)


if False:
  plot_info(dt, j, learnable_joints,
            joints_q_vec, q_traj_des,
            u_ff_vec[:,:-1], q_v_traj[:,1:],
            joint_torque_vec,
            disturbanc_vec, d_xyz, error_norms,
            v=True, p=True, dp=False, e_xyz=True, e=True, torque=True)

SAVING = True
if SAVING:
  # Saving Results
  filename = "examplesReal/dataReal/TrapezoidTest/joint_{}".format(learnable_joints) + time.strftime("%Y_%m_%d-%H_%M_%S.txt")

  save(filename,
        dt=dt, q_start=q_start, T_home=T_home,                                            # Home
        xyz_traj_des=xyz_traj_des, q_traj_des=q_traj_des,                                 # Desired Trajectories
        joints_q_vec=joints_q_vec, joints_vq_vec=joints_vq_vec,                           # Joint Informations
        joints_aq_vec=joints_aq_vec, joint_torque_vec=joint_torque_vec,                   #        =|=
        disturbanc_vec=disturbanc_vec, u_ff_vec=u_ff_vec,                                 # Learned Trajectories (uff and disturbance)
        xyz_vec=xyz_vec, joints_d_vec=joints_d_vec, error_norms=error_norms, d_xyz=d_xyz, # Progress Measurments
        learnable_joints=learnable_joints, alpha=alpha, n_ms=n_ms, n_ds=n_ds, ep_s=ep_s)  # ILC parameters

  with open('examplesReal/dataReal/TrapezoidTest/list_files.txt', 'a') as f:
    f.write(filename + "\n")



# Run Simulation with several repetition
# rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL, u=u_arr[:-end_repeat], joint_home_config=q_start, repetitions=25, it=j)
rArmInterface.apollo_run_one_iteration(dt=dt, T=T_FULL+end_repeat*dt, u=u_arr, joint_home_config=q_start, repetitions=5, it=j)
