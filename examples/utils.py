import __add_path__
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pickle

from juggling_apollo.utils import DotDict


colors = ["r", 'g', 'b', 'k', 'c', 'm', 'y']
line_types = ["-", "--", ":", '-.']

def plot_A(lines_list, indexes_list=list(range(7)), labels=None, dt=1, xlabel="", ylabel="", limits=None, fill_between=None, index_labels=None):
  # assert len(lines_list) == len(labels), "Please use same number of lines and labels"
  N = len(lines_list)
  M = len(indexes_list)
  lines_list = np.asarray(lines_list)
  timesteps = dt*np.arange(lines_list[0].shape[0])
  fig, axs = plt.subplots(1,M, figsize=(18, 3))
  axs = np.array(axs)
  for iii, ax in enumerate(axs.flat):
    ix = indexes_list[iii]
    for i in range(N):
      l = ax.plot(timesteps, lines_list[i][:, ix].squeeze(),
                              # color=colors[iii%len(colors)],
                              linestyle=line_types[i%len(line_types)],
                              label=r"{} {}".format(r"$\theta_{}$".format(ix+1) if index_labels is None else index_labels[ix], labels[i] if labels is not None else ''))
    if limits is not None:
      ax.axhspan(limits[iii].a, limits[iii].b, color='gray', alpha=0.2)  # color=l[0].get_color()
    if fill_between is not None:
      ax.fill_between(timesteps, fill_between[0][:, ix].squeeze(), fill_between[1][:, ix].squeeze(), color='gray', alpha=0.2)

    ymin = np.min(lines_list); ymax = np.max(lines_list); ytmp = abs(ymin - ymax)
    ax.set_ylim([ymin-0.1*ytmp, ymax+0.1*ytmp])
    ax.grid(True)
    # ax.set_title('joint ' + str(ix+1))
    ax.legend(loc='upper left')

  # Put legend on last axis
  # handles, labels = ax.get_legend_handles_labels()
  # axs.flatten()[-1].legend(handles, labels, loc='upper center')
  if limits is not None:
    liness = [plt.Rectangle((0,0),1,1, color='gray', alpha=0.2)]
    labelss = ['feasible set']
    fig.legend(liness, labelss, loc ="lower center", mode=None)

  fig.text(0.5, 0.04, xlabel, ha='center')
  fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
  fig.tight_layout(rect=[0, 0.06, 1, 0.95])


def save(filename, **kwargs):
  with open(filename, 'w') as f:
    pickle.dump(kwargs, f)


def load(filename):
  with open(filename, 'rb') as f:
    dd = pickle.load(f)
  return DotDict(dd)


def print_info(j, learnable_joints, joints_d_vec, d_xyz):
  print("ITERATION: " + str(j+1))
  print(          "j. -----------degree----------      L2-norm        L1-norm        e_end      <- unnormalized")
  for i in learnable_joints:
    print(str(i) + ". Trajectory_track_error_norm: {:13.8f}  {:13.8f} {:13.8f}".format(np.linalg.norm(180.0/np.pi*joints_d_vec[j, :, i]),  # /joints_d_vec.shape[1],
                                                                                       np.linalg.norm(180.0/np.pi*joints_d_vec[j, :, i], ord=1),  # /joints_d_vec.shape[1],
                                                                                       float(np.abs(180.0/np.pi*joints_d_vec[j, -1, i]))
                                                                                        ))
  ls = ['x', 'y', 'z']
  print(          "j. -----------meters----------      L2-norm        L1-norm        e_end      <- unnormalized")
  for i in range(3):
    print(ls[i] + ". Trajectory_track_error_norm: {:13.8f}  {:13.8f} {:13.8f}".format(np.linalg.norm(d_xyz[:, i]),  # /d_xyz.shape[0],
                                                                                      np.linalg.norm(d_xyz[:, i], ord=1),  #/d_xyz.shape[0],
                                                                                      np.abs(d_xyz[-1, i])
                                                                                      ))


def plot_info(dt, learnable_joints=list(range(7)),
          joints_q_vec=None, q_traj_des=None,
          u_ff_vec=None, q_v_traj=None,
          joint_torque_vec=None,
          disturbanc_vec=None, d_xyz=None,
          joint_error_norms=None, cartesian_error_norms=None,
          v=True, p=True, dp=False, e_xyz=False, e=False, torque=False, N=None, M=1,
          fname=None):

  def save(typ):
    if fname is not None:
      fig = plt.gcf()
      fig.set_size_inches((25.5, 8), forward=False)
      plt.savefig(fname+ "_" +typ + '.pdf', bbox_inches='tight')

  if joints_q_vec is not None and q_traj_des is not None and p:
    line_list = [180./np.pi*q_traj_des] + list(180./np.pi*joints_q_vec[:N:M])
    label_list = ["des"] + ["it="+str(i*M) for i in np.arange(len(line_list)-1)]
    plot_A(line_list, learnable_joints, label_list, dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$grad$]")
    plt.suptitle("Angle Positions")
    save("angle_pos")

  if u_ff_vec is not None and q_v_traj is not None and v:
    line_list = [q_v_traj] + list(u_ff_vec[:N:M])
    label_list = ["performed"] + ["it="+str(i*M) for i in np.arange(len(line_list)-1)]
    plot_A(line_list, learnable_joints, fill_between=[np.max(u_ff_vec, axis=0), np.min(u_ff_vec, axis=0)],
           labels=label_list, dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{grad}{s}$]")
    plt.suptitle("Angle Velocities")
    save("angle_vel")

  if disturbanc_vec is not None and dp:
    line_list = list(disturbanc_vec[:N:M])
    label_list = ["it="+str(i*M) for i in range(len(line_list))]
    plot_A(line_list, learnable_joints, label_list, dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$rad$]")
    plt.suptitle("Disturbance")
    save("disturbance")

  if d_xyz is not None and e_xyz:
    ls = ['x', 'y', 'z']
    fig, axs = plt.subplots(3,1, figsize=(12,8))
    for ii in range(3):
      axs[ii].plot(np.abs(d_xyz[:, ii]), c=colors[ii], label="d_"+ls[ii], linestyle=line_types[0])
      axs[ii].legend(loc=1)
    plt.suptitle("Cartesian Error Trajectories")
    save("d_xyz")

  if joint_error_norms is not None and e:
    plot_A([joint_error_norms], learnable_joints, ["L2-norm"], xlabel=r"$IT$", ylabel=r"angle [$rad$]")
    plt.suptitle("Joint angle errors through iterations")
    save("joint_error")

  if cartesian_error_norms is not None and e_xyz:
    labels = ["x", "y", "z", "nx", "ny", "nz"]
    plot_A([cartesian_error_norms], list(range(6)), None, xlabel=r"$IT$", ylabel="L2 norm", index_labels=labels)
    plt.suptitle("Cartesian errors through iterations")
    save("cartesian_error")

  if joint_torque_vec is not None and torque:
    plot_A([joint_torque_vec[-1]], learnable_joints, ["torque"], dt=dt, xlabel=r"$t$", ylabel=r"Newton")
    plt.suptitle("Torque trajectories for each joint")
    save("joint_torque")

  if fname is None:
    plt.show()


def save_all(filename, special=None, **kwargs):
  "Backups the data for reproduction, creates plots and saves plots."
  directory="/home/apollo/Desktop/Investigation/{}/".format(time.strftime("%Y_%m_%d"))
  if not os.path.exists(directory):
      os.makedirs(directory)

  dir_exp =directory+"{}/".format(time.strftime("%H_%M_%S") + ("_"+special if special is not None else ""))
  if not os.path.exists(dir_exp):
      os.makedirs(dir_exp)



  ld = DotDict(kwargs)
  filename = dir_exp + filename +'.data'

  # Save backup file
  save(filename, **ld)
  # Write down name of backup file
  with open(directory + 'list_files_all.txt', 'a') as f:
    f.write(filename + "\n")
  # Save plots
  plot_info(1, ld.learnable_joints,
            joints_q_vec=ld.joints_q_vec, q_traj_des=ld.q_traj_des_vec[-1,1:],
            u_ff_vec=ld.u_ff_vec, q_v_traj=ld.joints_vq_vec[-1,],
            joint_torque_vec=ld.joint_torque_vec, cartesian_error_norms =ld.cartesian_error_norms,
            disturbanc_vec=ld.disturbanc_vec, d_xyz=ld.d_xyz_vec[-1], joint_error_norms=ld.joint_error_norms,
            v=True, p=True, dp=True, e_xyz=True, e=True, torque=True, M=1,fname=dir_exp)
