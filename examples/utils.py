import __add_path__
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pickle

from juggling_apollo.utils import DotDict, full_extent

colors = ["r", 'g', 'b', 'k', 'c', 'm', 'y']
line_types = ["-", "--", ":", '-.']

def plot_A(lines_list, indexes_list=list(range(7)), labels=None, dt=1, xlabel="", ylabel="", limits=None, fill_between=None, index_labels=None, scatter_times=list(), degree=True, rows=1):
  # assert len(lines_list) == len(labels), "Please use same number of lines and labels"
  A = 180./np.pi if degree else 1.
  # A = 1.
  N = len(lines_list)
  M = len(indexes_list)
  lines_list = A*np.asarray(lines_list)
  timesteps = dt*np.arange(lines_list[0].shape[0])
  fig, axs = plt.subplots(M/rows, rows, figsize=(6, 18), sharex=True, sharey=True)

  axs = np.array(axs)
  lines2 = []
  for iii, ax in enumerate(axs.flat):
    ix = indexes_list[iii]
    for i in range(N):
      lines2 += ax.plot(timesteps,  lines_list[i][:, ix].squeeze(),
                              linestyle="-" if i ==0 else '--',
                              linewidth=1, marker="x"if i ==0 else None, markersize = 2,
                              label= i*"_" + (r"$\theta_{}$".format(ix+1) if index_labels is None else index_labels[ix]))
    if limits is not None:
      ax.axhspan(A*limits[iii].a, A*limits[iii].b, color='gray', alpha=0.2)  # color=l[0].get_color()
    if fill_between is not None:
      ax.fill_between(timesteps, A*fill_between[0][:, ix].squeeze(), A*fill_between[1][:, ix].squeeze(), color='purple', alpha=0.2)

    ymin = np.min(lines_list[:,:,indexes_list]); ymax = np.max(lines_list[:,:,indexes_list]); ytmp = abs(ymin - ymax)
    
    [ax.axvline(pos, linestyle='--', color='k') for pos in scatter_times]

    ax.set_ylim([ymin-0.1*ytmp, ymax+0.1*ytmp])
    ax.grid(True)
    # ax.set_title('joint ' + str(ix+1))
    ax.legend(loc='upper left')

  # Add extra legend for iteration space information
  liness = []
  labelss = []
  if limits is not None:
    liness = [plt.Rectangle((0,0),1,1, color='gray', alpha=0.2)]
    labelss = ['feasible set']


  if labels is not None:
    liness = liness + lines2
    labelss = labelss + [ labels[i] for i in range(N)  ]
  fig.legend(liness, labelss, loc ="lower right",
              mode=None, borderaxespad=1, ncol=4) # , fontsize=10

  fig.text(0.5, 0.14, xlabel, ha='center')
  fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical')
  fig.tight_layout()
  return axs.flat

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
          disturbanc_vec=None, d_xyz_rpy_vec=None,
          joint_error_norms=None, cartesian_error_norms=None,
          v=True, p=True, dp=False, e_xyz=False, e=False, torque=False, N=None, M=1,
          fname=None, kinematics=None):

  def save(typ, axs):
    if fname is not None:
      fig = plt.gcf()
      # fig.set_size_inches((25.5, 8), forward=False)
      plt.savefig(fname+ "_" +typ + '.pdf', bbox_inches='tight')
      
      # for i, ax2 in enumerate(axs):
      #   extent = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
      #   fname2 = fname+"/joint_{}/".format(learnable_joints[i])
      #   if not os.path.exists(fname2):
      #       os.makedirs(fname2)
        
      #   fig.savefig(fname2 + typ + "_" + '.pdf', bbox_inches=extent)

  qlim = kinematics.limits if kinematics is not None else None
  qvlim = kinematics.vlimits if kinematics is not None else None

  if joints_q_vec is not None and q_traj_des is not None and p:
    N_ = N if N is not None else len(joints_q_vec)
    its = [0] + [N_-i for i in np.arange(3,0,-1)]
    line_list = [q_traj_des] + list(joints_q_vec[its])
    label_list = ["des"] + ["it="+str(it) for it in its]
    axs = plot_A(line_list, learnable_joints, label_list, dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$degree$]", limits=qlim)
    plt.suptitle("Angle Positions")
    save("angle_pos", axs)

  if u_ff_vec is not None and q_v_traj is not None and v:
    N_ = N if N is not None else len(u_ff_vec)

    its = [0] + [N_-i for i in np.arange(3,0,-1)]
    line_list = [q_v_traj] + list(u_ff_vec[its])
    label_list = ["performed"] + ["it="+str(it) for it in its]
    axs = plot_A(line_list, learnable_joints, fill_between=[np.max(u_ff_vec, axis=0), np.min(u_ff_vec, axis=0)],
           index_labels=[r"$\dot{\theta}_%d$" %(i+1) for i in learnable_joints], labels=label_list, limits=qvlim,
           dt=dt, xlabel=r"$t$ [s]", ylabel=r" angle velocity [$\frac{degree}{s}$]")
    plt.suptitle("Angle Velocities")
    save("angle_vel", axs)

  if disturbanc_vec is not None and dp:
    N_ = N if N is not None else len(disturbanc_vec)

    its = [0] + [N_-i for i in np.arange(3,0,-1)]
    line_list = list(disturbanc_vec[its])
    label_list = ["it="+str(it) for it in its]
    index_labels=[r"$d_%d$" %(i+1) for i in learnable_joints]
    axs = plot_A(line_list, learnable_joints, label_list, dt=dt, xlabel=r"$t$ [s]", ylabel=r"angle [$degee$]",
           index_labels=index_labels)
    plt.suptitle("Disturbance")
    save("disturbance", axs)

  if d_xyz_rpy_vec is not None and e_xyz:
    ls = ['x', 'y', 'z', 'roll', 'pitch' ,'yaw']
    axs = plot_A(d_xyz_rpy_vec[0:2], list(range(6)), ["it0", "it1"], dt=dt, xlabel=r"$t$ [s]", ylabel=r"d [$m$]",
           index_labels=ls, degree=False, rows=1)
    plt.suptitle("Cartesian Error Trajectories")
    save("d_xyz_rpy_vec", axs)

  if joint_error_norms is not None and e:
    index_labels=[r"$||\theta_%d - \theta^{des}_%d||_2$" %(i+1, i+1) for i in learnable_joints]
    axs = plot_A([joint_error_norms], learnable_joints, ["L2-norm"], xlabel=r"$IT$", ylabel=r"angle [$degree$]",
           index_labels=index_labels)
    plt.suptitle("Joint angle errors through iterations")
    save("joint_error", axs)

  if cartesian_error_norms is not None and e_xyz:
    labels = ["x", "y", "z", "nx", "ny", "nz"]
    axs = plot_A([cartesian_error_norms], list(range(6)), None, xlabel=r"$IT$", ylabel="L2 norm", index_labels=labels)
    plt.suptitle("Cartesian errors through iterations")
    save("cartesian_error", axs)

  if joint_torque_vec is not None and torque:
    axs = plot_A([joint_torque_vec[-1]], learnable_joints, ["torque"], dt=dt, xlabel=r"$t$", ylabel=r"Newton",
           index_labels=[r"$M_%d$" %(i+1) for i in learnable_joints])
    plt.suptitle("Torque trajectories for each joint")
    save("joint_torque", axs)

  if fname is None:
    plt.show()


def save_all(filename, kinematics=None, special=None, **kwargs):
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
            u_ff_vec=ld.joints_vq_vec, q_v_traj=ld.qv_traj_des,
            joint_torque_vec=ld.joint_torque_vec, cartesian_error_norms =ld.cartesian_error_norms,
            disturbanc_vec=ld.disturbanc_vec, d_xyz_rpy_vec=ld.d_xyz_rpy_vec, joint_error_norms=ld.joint_error_norms,
            v=True, p=True, dp=True, e_xyz=True, e=True, torque=True, M=1, fname=dir_exp, kinematics=kinematics)
