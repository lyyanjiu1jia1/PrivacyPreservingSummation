import copy

import numpy as np
import time as tm


INIT = 'init'
JOIN = 'join'
LEAVE = 'leave'


class PrivacyPreservingSummation(object):
    def __init__(self, n, node_states, k_max, algorithm_name, new_node_state, error_type, save_data):
        self.n = n
        self.node_states = copy.deepcopy(node_states)
        self.init_node_states = copy.deepcopy(node_states)
        self.k_max = k_max
        self.k = 0
        self.trajectory = [self.node_states]
        self.algorithm_name = algorithm_name
        self.error_type = error_type
        self.save_data = save_data

        self.new_node_state = copy.deepcopy(new_node_state)

        self.average = {}
        self.sum = {}
        self._compute_average()

        self._gen_A()
        self._gen_B()

    @staticmethod
    def init_node_states(dim, scale=10, mean=500):
        node_states = scale * (np.random.random(size=(dim, 1)) - 0.5) + mean
        return node_states

    def run(self):
        beg_time = tm.time()

        while self.k < self.k_max:
            self._print_iteration()
            self._iterate()
            self._save_trajectory()

        # node join
        self._node_join()
        while self.k < 2 * self.k_max:
            self._print_iteration()
            self._iterate()
            self._save_trajectory()

        # node leave
        self._node_leave()
        self._save_trajectory()
        while self.k < 3 * self.k_max:
            self._print_iteration()
            self._iterate()
            self._save_trajectory()
        print(self.algorithm_name + " time used = {}".format(tm.time() - beg_time))

        if self.error_type == 'max':
            self._parse_trajectory_max()
        elif self.error_type == 'norm':
            self._parse_trajectory_norm()
        else:
            raise TypeError("Incompatible error type: {}".format(self.error_type))

        if self.save_data:
            self._save_file()
        print("final error = {}".format(self.trajectory[-1]))

    def _save_file(self):
        file_name = r'../data/' + self.algorithm_name + r'-traj.npy'
        np.save(file_name, self.trajectory)

    def _parse_trajectory_max(self):
        output_traj = []
        for k in range(len(self.trajectory)):
            phase, n = self._parse_phase(k)
            try:
                cur_traj = self.trajectory[k]
                cur_traj = np.max(np.abs(cur_traj - self.average[phase]))
            except:
                cur_traj = output_traj[-1]
            output_traj.append(cur_traj)
        self.trajectory = np.array(output_traj)

    def _parse_trajectory_norm(self):
        output_traj = []
        for k in range(len(self.trajectory)):
            phase, n = self._parse_phase(k)
            try:
                cur_traj = self.trajectory[k]
                cur_traj = np.linalg.norm(cur_traj - self.average[phase]) ** 2
            except:
                cur_traj = output_traj[-1]
            output_traj.append(cur_traj)
        self.trajectory = np.array(output_traj)

    def _iterate(self):
        pass

    def _save_trajectory(self):
        self.trajectory.append(self.node_states)

    @staticmethod
    def gen_gaussian_random_vector(dim, mean, std_dev):
        rand_vec = np.random.normal(size=(dim, 1), loc=mean, scale=std_dev)
        return rand_vec

    @staticmethod
    def gen_uniform_random_vector(dim, low, high):
        rand_vec = np.random.uniform(low=low,
                                     high=high,
                                     size=(dim, 1))
        return rand_vec

    def _gen_A(self):
        """
        Laplacian for ring graphs
        :return:
        """
        self.A = np.identity(self.n)
        for i in range(self.A.shape[0]):
            self.A[i, (i - 1) % self.n] = 1
            self.A[i, (i + 1) % self.n] = 1
        self.A *= 1 / 3

    def _gen_B(self):
        pass

    def _compute_average(self):
        self.average[INIT] = np.average(self.init_node_states)
        self.average[JOIN] = (np.sum(self.init_node_states) + self.new_node_state[0, 0])\
                             / (self.n + 1)
        self.average[LEAVE] = np.average(self.init_node_states)

    def _parse_phase(self, k):
        if (k - 2) // self.k_max <= 0:
            return INIT, self.n
        elif (k - 2) // self.k_max == 1:
            return JOIN, self.n + 1
        elif (k - 2) // self.k_max >= 2:
            return LEAVE, self.n

    def _node_join(self):
        # network topology switch
        self.n += 1
        self._gen_A()
        self._gen_B()

        # node states switch
        self.node_states = np.concatenate((self.node_states, self.new_node_state))

    def _node_leave(self):
        # network topology switch
        self.n -= 1
        self._gen_A()
        self._gen_B()

        # node states switch
        leaving_node_state = self.node_states[-1]
        compensation = leaving_node_state - self.new_node_state[0]
        self.node_states = np.delete(self.node_states, -1, axis=0)
        self.node_states[0] += 1 / 2 * compensation
        self.node_states[-1] += 1 / 2 * compensation

    def _next_node(self, i):
        return (i + 1) % self.n

    def _pre_node(self, i):
        return (i - 1) % self.n

    def _print_iteration(self):
        return
        if self.k_max <= 100 or self.k % 100 == 0:
            print("iteration k = {}".format(self.k))
