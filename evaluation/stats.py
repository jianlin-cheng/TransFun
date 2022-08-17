import pandas as pd

import Constants
from preprocessing.utils import pickle_load

annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
seq_id = [0.3, 0.5, 0.9, 0.95]
onts = ['molecular_function', 'cellular_component', 'biological_process']


def protein_stats_training():
    print("*****************--Statistics of proteins of training set--*************************")
    for ont in onts:
        print("*****************--{}--*************************".format(ont))
        for seq in seq_id:
            data = pickle_load(Constants.ROOT + '{}/{}/train'.format(seq, ont))
            print(seq, len(data))

        tmp = annot[annot[ont].notnull()][['Protein', ont]]
        print("Total # of proteins", tmp.shape[0])


def train_val_test_integrity():
    training_set, validation_set = set(), set()
    for ont in onts:
        for seq in seq_id:
            train = pickle_load(Constants.ROOT + '{}/{}/train'.format(seq, ont))
            training_set.update([protein for cluster in train.values() for protein in cluster])
            validation_set.update(pickle_load(Constants.ROOT + '{}/valid'.format(seq)))
    test_set_0 = set([i[0] for i in pickle_load(Constants.ROOT + 'eval/test_proteins_list')])
    test_set_1 = set([i[0] for i in pickle_load(Constants.ROOT + 'eval/test_proteins_not_found_list')])
    test_set_2 = set([i[0] for i in pickle_load(Constants.ROOT + 'eval/test_proteins_not_found_fasta')])
    return training_set, validation_set, test_set_0, test_set_1, test_set_2


##########################################################
# protein_stats_training()
training_set, validation_set, test_set_0, test_set_1, test_set_2 = train_val_test_integrity()
print(len(training_set), len(validation_set), len(test_set_0), len(test_set_1), len(test_set_2),
      len(training_set.intersection(validation_set)),
      len(training_set.intersection(test_set_0)),
      len(validation_set.intersection(test_set_0)))
##########################################################