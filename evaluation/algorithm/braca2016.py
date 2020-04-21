import numpy as np

from evaluation.algorithm.base_algorithm import PrivacyPreservingSummation


class Braca2016(PrivacyPreservingSummation):
    def __init__(self, n, node_states, k_max, noise_std_dev, new_node_state, error_type, save_data=True):
        self.noise_std_dev = noise_std_dev
        self.new_node_noise = self.gen_gaussian_random_vector(dim=1, mean=0, std_dev=self.noise_std_dev)

        super(Braca2016, self).__init__(n, node_states, k_max, algorithm_name='braca2016',
                                        new_node_state=new_node_state + self.new_node_noise,
                                        error_type=error_type, save_data=save_data)

    def _iterate(self):
        if self.k % self.k_max == 0:
            beta = self.gen_gaussian_random_vector(dim=self.n,
                                                   mean=0,
                                                   std_dev=self.noise_std_dev)
            self.node_states += beta

        self.node_states = self.A.dot(self.node_states)
        self.k += 1
