import matplotlib.pyplot as plt
import numpy as np

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
      axs.flatten()[iii].plot(timesteps, lines_list[i][:, ix].squeeze(), color=colors[ix], linestyle=line_types[i], label=r"$\theta_{}$ {}".format(ix, labels[i] if labels is not None else ""))
    if limits is not None:
      axs.flatten()[iii].axhspan(limits[iii].a, limits[iii].b, color=colors[iii], alpha=0.3, label='feasible set')
      axs.flatten()[iii].set_ylim([min(-np.pi, limits[iii].a), max(np.pi, limits[iii].b)])
    if fill_between is not None:
      axs.flatten()[iii].fill_between(timesteps, fill_between[0][:, ix].squeeze(), fill_between[1][:, ix].squeeze(), color=colors[iii], alpha=0.5)

    axs.flatten()[iii].legend(loc=1)
    axs.flatten()[iii].grid(True)
  fig.text(0.5, 0.04, xlabel, ha='center')
  fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
