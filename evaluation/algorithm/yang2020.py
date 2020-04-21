import numpy as np

from evaluation.algorithm.base_algorithm import PrivacyPreservingSummation, JOIN, LEAVE


class Yang2020(PrivacyPreservingSummation):
    def __init__(self, n, node_states, k_max, eta, new_node_state, error_type, save_data=True):
        """

        :param n:
        :param node_states:
        :param k:
        :param eta: the variance of beta is eta / k
        """
        super(Yang2020, self).__init__(n, node_states, k_max, algorithm_name='yang2020',
                                       new_node_state=new_node_state, error_type=error_type, save_data=save_data)
        self.eta = eta

    def _parse_trajectory_max(self):
        output_traj = []
        for k in range(len(self.trajectory)):
            phase, n = self._parse_phase(k)
            try:
                cur_traj = sum(self.trajectory[max(k // self.k_max * self.k_max + 1, k - n):k])
                cur_traj = np.max(np.abs(cur_traj / n - self.average[phase]))
            except:
                cur_traj = output_traj[-1]
            output_traj.append(cur_traj)
        output_traj.pop()
        self.trajectory = np.array(output_traj)

    def _parse_trajectory_norm(self):
        output_traj = []
        for k in range(len(self.trajectory)):
            phase, n = self._parse_phase(k)
            try:
                cur_traj = sum(self.trajectory[max(k // self.k_max * self.k_max + 1, k - n):k])
                cur_traj = np.linalg.norm(cur_traj / n - self.average[phase]) ** 2
            except:
                cur_traj = output_traj[-1]
            output_traj.append(cur_traj)
        self.trajectory = np.array(output_traj)

    def _iterate(self):
        """
        x(k+1) = A * x(k) + B * beta(k)
        :return:
        """
        std_dev = self._std_dev()
        beta = self.gen_gaussian_random_vector(dim=self.n,
                                               mean=0,
                                               std_dev=std_dev)
        self.node_states = self.A.dot(self.node_states) + self.B.dot(beta)
        self.k += 1

    def _gen_A(self):
        A_submatrix = np.identity(self.n - 1)
        A_submatrix = np.concatenate((A_submatrix, np.zeros(shape=(self.n - 1, 1))), axis=1)
        A_first_row = np.concatenate((np.zeros(shape=(1, self.n - 1)), np.ones(shape=(1, 1))), axis=1)
        self.A = np.concatenate((A_first_row, A_submatrix))

    def _gen_B(self):
        self.B = np.subtract(np.identity(self.n), self.A)

    def _std_dev(self):
        std_dev = self.eta / (self.k + 1)
        return std_dev

    def _node_leave(self):
        std_dev = self._std_dev()
        beta = self.gen_gaussian_random_vector(dim=self.n,
                                               mean=0,
                                               std_dev=std_dev)
        beta[-1, 0] = self.new_node_state
        beta[-2, 0] = self.node_states[-2, 0]
        self.node_states = self.A.dot(self.node_states) + self.B.dot(beta)
        self.node_states = np.delete(self.node_states, -1, axis=0)
        self.k += 1

        self.n -= 1
        self._gen_A()
        self._gen_B()