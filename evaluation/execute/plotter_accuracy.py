import copy

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


def split_traj(traj):
    cliff = -0.5
    split_point1 = None
    split_point2 = None
    idx = 0
    while not split_point1 or not split_point2:
        if not split_point1:
            if traj[idx] - traj[idx + 1] < cliff:
                split_point1 = idx
                idx += 100
        elif traj[idx] - traj[idx + 1] < cliff:
            split_point2 = idx
        idx += 1
    return traj[:split_point1 + 1], traj[split_point1 + 1:split_point2 + 1], traj[split_point2 + 1:]


def plot_accuracy():
    # plot properties
    linewidth = 1.5
    fig, subfigs = plt.subplots(2, 3)

    # yang2020
    traj = np.load(create_traj_file_name(YANG2020))
    traj1, traj2, traj3 = split_traj(traj)
    subfigs[0, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'b', linewidth=linewidth)
    subfigs[0, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'b', linewidth=linewidth)
    subfigs[0, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)), traj3,
                       'b', linewidth=linewidth)
    subfigs[1, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'b', linewidth=linewidth)
    subfigs[1, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'b',
                       linewidth=linewidth)
    subfigs[1, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)),
                       traj3,
                       'b', linewidth=linewidth)

    # braca2016
    traj = np.load(create_traj_file_name(BRACA2016))
    traj1, traj2, traj3 = split_traj(traj)
    subfigs[0, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'g', linewidth=linewidth)
    subfigs[0, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'g', linewidth=linewidth)
    subfigs[0, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)), traj3,
                       'g', linewidth=linewidth)
    subfigs[1, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'g', linewidth=linewidth)
    subfigs[1, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'g',
                       linewidth=linewidth)
    subfigs[1, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)),
                       traj3,
                       'g', linewidth=linewidth)

    # he2017
    traj = np.load(create_traj_file_name(HE2017))
    traj1, traj2, traj3 = split_traj(traj)
    subfigs[0, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'y', linewidth=linewidth)
    subfigs[0, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'y', linewidth=linewidth)
    subfigs[0, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)), traj3,
                       'y', linewidth=linewidth)
    subfigs[1, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'y', linewidth=linewidth)
    subfigs[1, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'y',
                       linewidth=linewidth)
    subfigs[1, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)),
                       traj3,
                       'y', linewidth=linewidth)

    # ruan2019
    traj = np.load(create_traj_file_name(RUAN2019))
    traj1, traj2, traj3 = split_traj(traj)
    subfigs[0, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'r', linewidth=linewidth)
    subfigs[0, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'r', linewidth=linewidth)
    subfigs[0, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)), traj3,
                       'r', linewidth=linewidth)
    subfigs[1, 0].plot(np.linspace(0, len(traj1) - 1, len(traj1)), traj1, 'r', linewidth=linewidth)
    subfigs[1, 1].plot(np.linspace(len(traj1), len(traj2) + len(traj1) - 1, len(traj2)), traj2, 'r',
                       linewidth=linewidth)
    subfigs[1, 2].plot(np.linspace(len(traj2) + len(traj1), len(traj2) + len(traj1) + len(traj3) - 1, len(traj3)),
                       traj3,
                       'r', linewidth=linewidth)

    # settings
    subfigs[0, 0].set_title('initial phase')
    subfigs[0, 1].set_title('node join')
    subfigs[0, 2].set_title('node leave')
    fig.suptitle('Error Trajectories')

    subfigs[0, 2].legend(['SI-PPSP', 'Braca2016', 'He2017', 'Ruan2019'])

    subfigs[0, 0].grid(True)
    subfigs[0, 1].grid(True)
    subfigs[0, 2].grid(True)
    subfigs[1, 0].grid(True)
    subfigs[1, 1].grid(True)
    subfigs[1, 2].grid(True)

    subfigs[0, 0].set_xlim((0, 500))
    subfigs[0, 1].set_xlim((500, 999))
    subfigs[0, 2].set_xlim((1000, 1500))
    subfigs[1, 0].set_xlim((0, 500))
    subfigs[1, 1].set_xlim((500, 999))
    subfigs[1, 2].set_xlim((1000, 1500))

    subfigs[0, 0].set_ylim((0, 5))
    subfigs[0, 1].set_ylim((0, 35))
    subfigs[0, 2].set_ylim((0, 35))
    subfigs[1, 0].set_ylim((0, 0.6))
    subfigs[1, 1].set_ylim((0, 0.6))
    subfigs[1, 2].set_ylim((0, 0.6))

    subfigs[1, 1].set_xlabel('iteration $k$')
    subfigs[0, 0].set_ylabel('$\max_{i\in\mathrm{V}}\mid z_i(k) - ave[sum]\mid$')
    subfigs[1, 0].set_ylabel('$\max_{i\in\mathrm{V}}\mid z_i(k) - ave[sum]\mid$')

    fig.savefig(r'../figure/accuracy.png')


plot_accuracy()
