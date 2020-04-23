import numpy as np

from evaluation.algorithm.base_algorithm import PrivacyPreservingSummation, INIT, JOIN
from evaluation.algorithm.braca2016 import Braca2016
from evaluation.algorithm.he2017 import He2017
from evaluation.algorithm.ruan2019 import Ruan2019
from evaluation.algorithm.yang2020 import Yang2020
from matplotlib import pyplot as plt


YANG2020 = 'yang2020'

n = 100
k_max = 500
init_mean = 1
init_std_dev = 1
eta = 20
sigma = 1
varrho = 0.9
epsilon = 0.005
key_size = 2048
a_scale = 10
universal_init_node_states = PrivacyPreservingSummation.init_node_states(dim=n,
                                                                         mean=init_mean,
                                                                         scale=init_std_dev)
universal_new_node_state = PrivacyPreservingSummation.init_node_states(dim=1,
                                                                       mean=init_mean * 100,
                                                                       scale=init_std_dev)

error_type = 'max'


def run_yang2020():
    yang2020 = Yang2020(n=n,
                        k_max=k_max,
                        eta=eta,
                        node_states=universal_init_node_states,
                        new_node_state=universal_new_node_state,
                        error_type=error_type)
    yang2020.run_effectiveness()
    print("new node state = {}".format(universal_new_node_state[0, 0]))
    return yang2020.get_trajectory(), yang2020.get_average()


def create_lin_space(start, length):
    return np.linspace(start, start + length - 1, length)


def plot_effectiveness(traj, average):
    # plot properties
    linewidth = 0.1
    color_offset = -4

    # yang2020
    for node_traj in traj:
        phase1 = node_traj[:k_max + color_offset]
        phase2 = node_traj[k_max + color_offset + 1:2 * k_max + color_offset]
        phase3 = node_traj[2 * k_max + color_offset + 1:]
        plt.plot(create_lin_space(0, len(phase1)), phase1, 'b', linewidth=linewidth)
        plt.plot(create_lin_space(len(phase1), len(phase2)), phase2, 'r', linewidth=linewidth)
        plt.plot(create_lin_space(len(phase1) + len(phase2), len(phase3)), phase3, 'g', linewidth=linewidth)
        plt.legend(['initial phase', 'node join', 'node leave'])

    # plot benchmark
    plt.plot(create_lin_space(0, len(traj[0])), [average[INIT] * n] * len(traj[0]), 'c--', linewidth=linewidth * 10)
    plt.plot(create_lin_space(0, len(traj[0])), [average[JOIN] * n] * len(traj[0]), 'm--', linewidth=linewidth * 10)
    plt.text(0.71 * len(traj[0]), average[INIT] * n * 0.8, r'average before join', fontsize=10)
    plt.text(0.733 * len(traj[0]), average[JOIN] * n * 0.9, r'average after join', fontsize=10)

    # settings
    plt.xlim((0, 3 * k_max))
    plt.ylim((-50, 330))
    plt.grid(True)
    plt.xlabel('iteration $k$')
    plt.ylabel('$y_i(k)$')
    plt.title(r'Trajectories of $y_i(k), i\in\mathrm{V}$')
    plt.savefig(r'../figure/effectiveness.png')


traj, average = run_yang2020()
plot_effectiveness(traj, average)
