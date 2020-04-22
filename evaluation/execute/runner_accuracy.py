from evaluation.algorithm.base_algorithm import PrivacyPreservingSummation
from evaluation.algorithm.braca2016 import Braca2016
from evaluation.algorithm.he2017 import He2017
from evaluation.algorithm.ruan2019 import Ruan2019
from evaluation.algorithm.yang2020 import Yang2020

n = 100
k_max = 500
init_mean = 1
init_std_dev = 1
eta = 3
sigma = 3
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
    yang2020.run()


def run_braca2016():
    braca2016 = Braca2016(n=n,
                          k_max=k_max,
                          noise_std_dev=eta,
                          node_states=universal_init_node_states,
                          new_node_state=universal_new_node_state,
                          error_type=error_type)
    braca2016.run()


def run_he2017():
    he2017 = He2017(n=n,
                    k_max=k_max,
                    sigma=sigma,
                    varrho=varrho,
                    node_states=universal_init_node_states,
                    new_node_state=universal_new_node_state,
                    error_type=error_type)
    he2017.run()


def run_ruan2019():
    ruan2019 = Ruan2019(n=n,
                        k_max=k_max,
                        node_states=universal_init_node_states,
                        new_node_state=universal_new_node_state,
                        error_type=error_type,
                        epsilon=epsilon,
                        key_size=key_size,
                        a_scale=a_scale)
    ruan2019.run()


run_yang2020()
run_braca2016()
run_he2017()
run_ruan2019()
