import pandas as pd

import Constants
from preprocessing.utils import pickle_load


annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
seq_id = [0.3, 0.5, 0.9, 0.95]
onts = ['molecular_function', 'cellular_component', 'biological_process']

print("*****************--Statistics of proteins--*************************")
for ont in onts:
    print("*****************--Ontology {}--*************************".format(ont))
    for seq in seq_id:
        data = pickle_load(Constants.ROOT + '{}/{}/train'.format(seq, ont))
        print(seq, len(data))

    tmp = annot[annot[ont].notnull()][['Protein', ont]]
    print("Total # of proteins", tmp.shape[0])