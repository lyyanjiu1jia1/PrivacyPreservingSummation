import numpy as np
from matplotlib import pyplot as plt

YANG2020 = 'yang2020'
BRACA2016 = 'braca2016'
HE2017 = 'he2017'
RUAN2019 = 'ruan2019'


def create_traj_file_name(algorithm_name):
    file_name = r'../data/' + algorithm_name + '-traj.npy'
    return file_name


def create_traj_figure_name(algorithm_name):
    file_name = r'../figure/' + algorithm_name + '-traj.png'
    return file_name


def plot_accuracy():
    # plot properties
    linewidth = 1

    # yang2020
    traj = np.load(create_traj_file_name(YANG2020))
    plt.plot(np.linspace(0, len(traj) - 1, len(traj)), traj, 'b', linewidth=linewidth)

    # braca2016
    traj = np.load(create_traj_file_name(BRACA2016))
    plt.plot(np.linspace(0, len(traj) - 1, len(traj)), traj, 'g', linewidth=linewidth)

    # he2017
    traj = np.load(create_traj_file_name(HE2017))
    plt.plot(np.linspace(0, len(traj) - 1, len(traj)), traj, 'y', linewidth=linewidth)

    # ruan2019
    traj = np.load(create_traj_file_name(RUAN2019))
    plt.plot(np.linspace(0, len(traj) - 1, len(traj)), traj, 'r', linewidth=linewidth)

    # settings
    plt.ylim((0, 1.0))
    plt.grid(True)
    plt.legend(['SI-PPSP', 'Braca2016', 'He2017', 'Ruan2019'])
    plt.xlabel('iterations $k$')
    plt.ylabel('$\max_{i\in\mathrm{V}}\mid z_i(k) - ave[sum]\mid$')
    plt.title(r'Error Trajectories')
    plt.savefig(r'../figure/accuracy.png')


plot_accuracy()
