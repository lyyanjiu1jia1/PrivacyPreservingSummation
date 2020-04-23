import copy

import numpy as np

from evaluation.algorithm.base_algorithm import PrivacyPreservingSummation
# from phe import generate_paillier_keypair
from phe.dummy import generate_paillier_keypair


class Ruan2019(PrivacyPreservingSummation):
    def __init__(self, n, node_states, k_max, epsilon, key_size, new_node_state, error_type, a_scale, save_data=True):
        super(Ruan2019, self).__init__(n, node_states, k_max, algorithm_name='ruan2019',
                                       new_node_state=new_node_state, error_type=error_type, save_data=save_data)
        self.epsilon = epsilon
        self.key_size = key_size

        self.a_list = []            # [[pre_a, next_a], [pre_a, next_a], ...]
        for i in range(self.n):
            self.a_list.append(np.random.random(2) * a_scale)

        self.pk_list, self.sk_list = [], []
        for i in range(self.n):
            print("generating {}-th key".format(i + 1))
            pk, sk = generate_paillier_keypair(n_length=self.key_size)
            self.pk_list.append(pk)
            self.sk_list.append(sk)

        self.new_node_pk, self.new_node_sk = generate_paillier_keypair(n_length=self.key_size)
        self.new_node_a = np.random.random(2) * a_scale

    def _iterate(self):
        # compute delta edge by edge
        delta_matrix = [[] for _ in range(self.node_states.shape[0])]
        for i in range(self.n):
            j = self._next_node(i)

            # encrypt the own states
            ei_i = self.pk_list[i].encrypt(-self.node_states[i, 0])
            ej_j = self.pk_list[j].encrypt(-self.node_states[j, 0])

            # encrypt the other's state
            ei_j = self.pk_list[i].encrypt(self.node_states[j, 0])
            ej_i = self.pk_list[j].encrypt(self.node_states[i, 0])

            # add and mult by weights
            eij = self.a_list[j][0] * (ei_j + ei_i)     # for deltaij
            eji = self.a_list[i][1] * (ej_i + ej_j)     # for deltaji

            # decrypt and save
            delta_ij = self.a_list[i][1] * self.sk_list[i].decrypt(eij)
            delta_ji = self.a_list[j][0] * self.sk_list[j].decrypt(eji)
            delta_matrix[i].append(delta_ij)
            delta_matrix[j].append(delta_ji)

        # update
        temp_node_states = copy.deepcopy(np.zeros((self.node_states.shape[0], 1)))
        for i in range(self.node_states.shape[0]):
            temp_node_states[i, 0] = self.node_states[i, 0] + self.epsilon * sum(delta_matrix[i])
        self.node_states = temp_node_states

        self.k += 1

    def _node_join(self):
        super(Ruan2019, self)._node_join()

        self.pk_list.append(self.new_node_pk)
        self.sk_list.append(self.new_node_sk)
        self.a_list.append(self.new_node_a)

    def _node_leave(self):
        super(Ruan2019, self)._node_leave()

        self.pk_list.pop()
        self.sk_list.pop()
        self.a_list.pop()
