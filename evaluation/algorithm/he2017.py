import numpy as np

from evaluation.algorithm.base_algorithm import PrivacyPreservingSummation


class He2017(PrivacyPreservingSummation):
    def __init__(self, n, node_states, k_max, sigma, varrho, new_node_state, error_type, save_data=True):
        super(He2017, self).__init__(n, node_states, k_max, algorithm_name='he2017',
                                     new_node_state=new_node_state, error_type=error_type, save_data=save_data)
        self.sigma = sigma
        self.varrho = varrho
        self._gen_v()

    def _iterate(self):
        new_v = self.gen_uniform_random_vector(dim=self.n,
                                               low=-np.sqrt(3) * self.sigma,
                                               high=np.sqrt(3) * self.sigma)
        theta = self.varrho ** (self.k + 1) * new_v - self.varrho ** self.k * self.v

        self.node_states = self.A.dot(self.node_states + theta)

        self.k += 1
        self.v = new_v

    def _node_join(self):
        super(He2017, self)._node_join()
        additional_v = self.gen_uniform_random_vector(dim=1,
                                                      low=-np.sqrt(3) * self.sigma,
                                                      high=np.sqrt(3) * self.sigma)
        self.v = np.concatenate((self.v, additional_v))

    def _node_leave(self):
        super(He2017, self)._node_leave()
        self.v = np.delete(self.v, -1)

    def _gen_v(self):
        """
        the noise
        :return:
        """
        self.v = self.gen_uniform_random_vector(dim=self.n,
                                                low=-np.sqrt(3) * self.sigma,
                                                high=np.sqrt(3) * self.sigma)
