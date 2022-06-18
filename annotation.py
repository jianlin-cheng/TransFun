import pandas as pd

import Constants
from preprocessing.utils import pickle_load, collect_test


# raw_file_list_03 = set()
# processed_file_list = []
# data = pickle_load(Constants.ROOT + "{}/{}/{}".format(0.3, 'cellular_component', 'train'))
# for i in data:
#     raw_file_list_03.update(data[i])
# print(len(raw_file_list_03))
#
#
# raw_file_list_05 = set()
# processed_file_list = []
# data = pickle_load(Constants.ROOT + "{}/{}/{}".format(0.5, 'cellular_component', 'train'))
# for i in data:
#     raw_file_list_05.update(data[i])
# print(len(raw_file_list_05))
#
#
# raw_file_list_09 = set()
# processed_file_list = []
# data = pickle_load(Constants.ROOT + "{}/{}/{}".format(0.9, 'cellular_component', 'train'))
# for i in data:
#     raw_file_list_09.update(data[i])
# print(len(raw_file_list_09))
#
#
# raw_file_list_95 = set()
# processed_file_list = []
# data = pickle_load(Constants.ROOT + "{}/{}/{}".format(0.95, 'cellular_component', 'train'))
# for i in data:
#     raw_file_list_95.update(data[i])
# print(len(raw_file_list_95))
#
# print(len(raw_file_list_03 - raw_file_list_05), len(raw_file_list_05 - raw_file_list_03))


def collect_test_clusters(cluster_path):
    # collect test and clusters
    total_test = collect_test()

    computed = pd.read_csv(cluster_path, names=['cluster'], header=None).to_dict()['cluster']
    computed = {i: set(computed[i].split('\t')) for i in computed}

    test_cluster = set()
    train_cluster_indicies = []
    for i in computed:
        if total_test.intersection(computed[i]):
            test_cluster.update(computed[i])
        else:
            train_cluster_indicies.append(i)

    return test_cluster, train_cluster_indicies


seqs = [0.3, 0.5, 0.9, 0.95]
for i in seqs:
    cluster_path = Constants.ROOT + "{}/mmseq/final_clusters.csv".format(i)

    test_cluster, train_cluster_indicies = collect_test_clusters(cluster_path)

    # print(len(train_cluster_indicies))
    print(len(test_cluster))