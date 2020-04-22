import numpy as np

YANG2020 = 'yang2020'
BRACA2016 = 'braca2016'
HE2017 = 'he2017'
RUAN2019 = 'ruan2019'


def create_compl_file_name(algorithm_name):
    file_name = r'../data/' + algorithm_name + '-compl.npy'
    return file_name


def read_file(alg_name):
    file_name = create_compl_file_name(alg_name)
    time_cost = np.load(file_name)
    return time_cost


time_table = {}
alg_name_list = [YANG2020, BRACA2016, HE2017, RUAN2019]

for alg_name in alg_name_list:
    time_table[alg_name] = read_file(alg_name)[0]

print(time_table)
