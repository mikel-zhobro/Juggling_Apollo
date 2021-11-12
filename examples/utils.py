import __add_path__
import matplotlib.pyplot as plt
import numpy as np
import pickle

from juggling_apollo.utils import DotDict

colors = ["r", 'g', 'b', 'k', 'c', 'm', 'y']
line_types = ["-", "--", ":", '-.']

def plot_A(lines_list, indexes_list=list(range(7)), labels=None, dt=1, xlabel="", ylabel="", limits=None, fill_between=None):
  # assert len(lines_list) == len(labels), "Please use same number of lines and labels"
  N = len(lines_list)
  M = len(indexes_list)
  if M >= 3:
    a = M//3 + (1 if M%3 !=0 else 0)
    b = 3
  else:
    a = 1
    b = M
  timesteps = dt*np.arange(lines_list[0].shape[0])
  fig, axs = plt.subplots(a,b, figsize=(12,8))
  axs = np.array(axs)
  for iii, ix in enumerate(indexes_list):
    for i in range(N):
      axs.flatten()[iii].plot(timesteps, lines_list[i][:, ix].squeeze(), color=colors[iii], linestyle=line_types[i], label=r"$\theta_{}$ {}".format(ix, labels[i] if labels is not None else ""))
    if limits is not None:
      axs.flatten()[iii].axhspan(limits[iii].a, limits[iii].b, color=colors[iii], alpha=0.3, label='feasible set')
      axs.flatten()[iii].set_ylim([min(-np.pi, limits[iii].a), max(np.pi, limits[iii].b)])
    if fill_between is not None:
      axs.flatten()[iii].fill_between(timesteps, fill_between[0][:, ix].squeeze(), fill_between[1][:, ix].squeeze(), color=colors[iii], alpha=0.5)

    axs.flatten()[iii].legend(loc=1)
    axs.flatten()[iii].grid(True)
  fig.text(0.5, 0.04, xlabel, ha='center')
  fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')


def save(filename, **kwargs):
  with open(filename, 'w') as f:
    pickle.dump(kwargs, f)
    
def load(filename):
  with open(filename, 'rb') as f:
    dd = pickle.load(f)
  return DotDict(dd)

# USAGE
# filename = os.path.join("data", '{}_simdata_raw.dat'.format(time.strftime("%Y_%m_%d-%H_%M_%S")))
# aa = DotDict(mikel=np.eye(7), andy=12)
# save(filename, **aa)
# sd = load(filename)
# print(sd.mikel.shape)


def print_info(j, learnable_joints, joints_d_vec, d_xyz):
  print("ITERATION: " + str(j+1))
  print(          "j. -----------degree----------      L2-norm        L1-norm        e_end      <- unnormalized")
  for i in learnable_joints:
    print(str(i) + ". Trajectory_track_error_norm: {:13.8f}  {:13.8f} {:13.8f}".format(np.linalg.norm(180.0/np.pi*joints_d_vec[j, :, i]), 
                                                                                        np.linalg.norm(180.0/np.pi*joints_d_vec[j, :, i], ord=1),
                                                                                        np.abs(180.0/np.pi*joints_d_vec[j, -1, i])
                                                                                        ))
  ls = ['x', 'y', 'z']
  print(          "j. -----------meters----------      L2-norm        L1-norm        e_end      <- unnormalized")
  for i in range(3):
    print(ls[i] + ". Trajectory_track_error_norm: {:13.8f}  {:13.8f} {:13.8f}".format(np.linalg.norm(d_xyz[:, i]), 
                                                                                      np.linalg.norm(d_xyz[:, i], ord=1),
                                                                                      np.abs(d_xyz[-1, i])
                                                                                      ))
    
    
def plot_info(dt, j=0, learnable_joints=list(range(7)), 
          joints_q_vec=None, q_traj_des=None, 
          u_ff_vec=None, q_v_traj=None,
          joint_torque_vec=None,
          disturbanc_vec=None, d_xyz=None, error_norms=None,
          v=True, p=True, dp=False, e_xyz=False, e=False, torque=False):
  if joints_q_vec is not None and q_traj_des is not None and p:
    plot_A([q_traj_des, joints_q_vec[j], joints_q_vec[j-1], joints_q_vec[0]], learnable_joints, ["des", "it="+str(j), "it="+str(j-4), "it=0"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
    plt.suptitle("Angle Positions")
    plt.show(block=False)
  
  if u_ff_vec is not None and q_v_traj is not None and v:
    plot_A([u_ff_vec[j, :-1], q_v_traj[1:]], learnable_joints, fill_between=[np.max(u_ff_vec, axis=0)[1:], np.min(u_ff_vec, axis=0)[1:]],
            labels=["desired", "real"], dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{rad}{s}$]")
    plt.suptitle("Angle Velocities")
    plt.show(block=False)
  
  if disturbanc_vec is not None and dp:
    plot_A([disturbanc_vec[j, 1:], disturbanc_vec[j-2, 1:], disturbanc_vec[j-4, 1:], disturbanc_vec[1, 1:]], learnable_joints, ["d", "it-2", "it-4", "it=1"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
    plt.suptitle("Disturbance")
    plt.show(block=False)
    
  if d_xyz is not None and e_xyz:
    ls = ['x', 'y', 'z']
    fig, axs = plt.subplots(3,1, figsize=(12,8))
    for ii in range(3):
      axs[ii].plot(np.abs(d_xyz[:, ii]), c=colors[ii], label="d_"+ls[ii], linestyle=line_types[0])
      axs[ii].legend(loc=1)
    plt.suptitle("Cartesian Error Trajectories")
    plt.show(block=False)
    
  if error_norms is not None and e:
    plot_A([error_norms[:j+1]], learnable_joints, ["L2-norm"], xlabel=r"$IT$", ylabel=r"angle [$rad$]")
    plt.suptitle("Joint angle errors through iterations")
    plt.show(block=False)


  if joint_torque_vec is not None and torque:
    plot_A([joint_torque_vec[j]], learnable_joints, ["torque"], dt=dt, xlabel=r"$t$", ylabel=r"Newton")
    plt.suptitle("Torque trajectories for each joint")
    plt.show(block=False)


  plt.show()
