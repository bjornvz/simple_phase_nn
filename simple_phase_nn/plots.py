import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import itertools
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score # NOTE: use pytorch r2_score instead
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from simple_phase_nn.utils import *
import matplotlib.gridspec as gridspec

sns.set_theme()
sns.color_palette("Paired")


def save_fig(fig, folder_path, file_name):
    """
    Saves fig to path as a png file.
    """
    folder = Path(folder_path)
    save_path = folder.joinpath(file_name)
    fig.savefig(save_path)





def plot_phase_transition(metrics, cf, file_name):
    """
    Plots the output node 0 against the true labels and the derivative of the (averaged per label) output per label.
    """
    fig = plt.figure(figsize=(4, 8))
    labels = np.array(metrics["labels"]).flatten()  # true labels, shape (2000,)
    out0 = np.array(metrics["out0"])  # output node 0

    unique_labels = np.unique(labels)
    sorted_label_indices = np.argsort(labels)
    
    sorted_out0 = out0[sorted_label_indices]

    sorted_labels = labels[sorted_label_indices]

    avg_sorted_out = [sorted_out0[sorted_labels == label].mean()
                for label in unique_labels]
    derivative = np.gradient(avg_sorted_out, unique_labels)
    peak_index = np.argmax(derivative)

    phase_indicator = np.array(metrics["phase_indicator"])

    plt.subplot(2, 1, 1)
    plt.plot(sorted_labels, sorted_out0, 'o', markersize=2, color='black')
    plt.plot(unique_labels, avg_sorted_out, color='green', label='avg out per label')
    plt.xlabel(rf"$\gamma$={cf.label_param}")  # label for the x-axis
    plt.ylabel(r"$\hat{\gamma}$")  # label for the y-axis
    plt.title("output vs true label")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(unique_labels, derivative, color='magenta', label='derivative of avg out')
    # plt.plot(unique_labels, phase_indicator[-1], color='blue', label='phase_indicator during training')
    plt.axvline(x=unique_labels[peak_index], color='black', linestyle='--', label=f'max, {cf.label_param}={unique_labels[peak_index]:.2f}',linewidth=2)
    plt.xlabel(rf'$\gamma$={cf.label_param}')  # label for the x-axis
    plt.ylabel(cf.phase_indicator_str)
    plt.title("phase indicator")
    plt.legend()

    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    plt.close()


def plot_history(metrics, cf, file_name):
    """"
    Plots losses and the bottleneck activations for each epoch.
    """
    metric1, metric2 = "train_loss", "train_l1"
    metric3, metric4 = "val_loss", "val_l1"
    metric5 = cf.goodness_str

    if cf.kernels is not None:
        z = np.array([ np.abs(np.mean(metrics[f'z_{i}'], axis=1)) for i in range(len(cf.kernels)) ]) # mean over the batch

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  # 4 rows, 2 columns

    # subplot 1: all the losses
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics[metric1], label='train', color="red")
    ax1.plot(metrics[metric3], label='val', color="blue")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(fr"$\mathbf{{loss}}$, train= {metrics[metric1][-1]:.2e}, val={metrics[metric3][-1]:.2e}", loc='left')


    # subplot 3: acc/r2
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics[f"val_"+metric5], label=metric5, color="blue")
    ax3.plot(metrics[f"train_"+metric5], label=f"train_{metric5}", color="red")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel(cf.goodness_str)
    ax3.set_title(fr"$\mathbf{{{cf.goodness_str}}}$, train= {metrics[f'train_{metric5}'][-1]:.2f}, val={metrics[f'val_{metric5}'][-1]:.2f}", loc='left')
    lower_limit = 0 
    ax3.set_ylim(lower_limit, 1.01)


    phase_indicator = np.array(metrics["phase_indicator"])
    unique_labels = np.array(metrics["unique_labels"])
    peak_index = np.argmax(phase_indicator[-1])


    # subplot 5: phase indicator spanning two rows
    ax5 = fig.add_subplot(gs[2:, 0])  # Span rows 2 and 3, both columns
    sns.set_style("dark")
    cax = ax5.imshow(phase_indicator.T, aspect='auto', cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(cax, ax=ax5, orientation='horizontal', pad=0.2)
    cbar.set_label(cf.phase_indicator_str)
    ax5.set_ylabel(cf.label_param)
    ax5.grid(False)
    ax5.set_xlabel("epoch")
    ax5.set_title(f"phase indicator: {cf.phase_indicator_str}")
    ax5.axhline(y=peak_index, color='white', linestyle='--', label=f'max last epoch: {cf.label_param}={unique_labels[peak_index]:.2f}', linewidth=2)
    yticks = np.linspace(0, len(unique_labels) - 1, num=5, dtype=int)
    ax5.set_yticks(yticks)
    formatted_labels = [f"{label:.2f}" for label in unique_labels[yticks]]
    ax5.set_yticklabels(formatted_labels)
    ax5.invert_yaxis()
    ax5.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    plt.close()
