import os
import subprocess
import pandas as pd
import torch
import esm
import torch.nn.functional as F

import Constants
from preprocessing.utils import pickle_save, pickle_load


# Script to test esm
def test_esm():
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", "K A <mask> I S Q"),
    ]

    for i, j in data:
        print(len(j))

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    print(token_representations.shape)

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))

    for i in sequence_representations:
        print(len(i))

    # # Look at the unsupervised self-attention map contact predictions
    # import matplotlib.pyplot as plt
    # for (_, seq), attention_contacts in zip(data, results["contacts"]):
    #     plt.matshow(attention_contacts[: len(seq), : len(seq)])
    #     plt.title(seq)
    #     plt.show()


# Generate ESM embeddings in bulk
# In this function, I create embedding for each fasta sequence in the fasta file
# Extract file is taken from the github directory
def generate_bulk_embedding(fasta_file, output_dir, path_to_extract_file):
    subprocess.call('python extract.py esm1b_t33_650M_UR50S {} {} --repr_layers 0 32 33 '
                    '--include mean per_tok --truncate'.format("{}".format(fasta_file),
                                                    "{}".format(output_dir)),
                    shell=True, cwd="{}".format(path_to_extract_file))


# generate_bulk_embedding(Constants.ROOT + "uniprot/{}.fasta".format("filtered"),
#                        "/data/pycharm/TransFunData/data/esm1",
#                       "/data/pycharm/TransFun/preprocessing")


# Generate data for each group
def generate_data():

    def get_stats(data):
        go_terms = {}
        for i in data:
            for j in i.split(","):
                if j in go_terms:
                    go_terms[j] = go_terms[j] + 1
                else:
                    go_terms[j] = 1
        return go_terms


    categories = [('molecular_function', 'GO-terms (molecular_function)'),
                  ('biological_process', 'GO-terms (biological_process)'),
                  ('cellular_component', 'GO-terms (cellular_component)')]
    x_id = '### PDB-chain'

    train_set = pickle_load(Constants.ROOT + "final_train")
    valid_set = pickle_load(Constants.ROOT + "final_valid")
    test_set = pickle_load(Constants.ROOT + "final_test")

    for i in categories:
        print("Generating for {}".format(i[0]))

        if not os.path.isdir(Constants.ROOT + i[0]):
            os.mkdir(Constants.ROOT + i[0])

        df = pd.read_csv("/data/pycharm/TransFunData/data/final_annot.tsv", skiprows=12, delimiter="\t")
        df = df[df[i[1]].notna()][[x_id, i[1]]]

        train_df = df[df[x_id].isin(train_set)]
        train_df.to_pickle(Constants.ROOT + i[0] + "/train.pickle")
        stats = get_stats(train_df[i[1]].to_list())
        pickle_save(stats, Constants.ROOT + i[0] + "/train_stats")
        print(len(stats))

        valid_df = df[df[x_id].isin(valid_set)]
        valid_df.to_pickle(Constants.ROOT + i[0] + "/valid.pickle")
        stats = get_stats(valid_df[i[1]].to_list())
        pickle_save(stats, Constants.ROOT + i[0] + "/valid_stats")
        print(len(stats))

        test_df = df[df[x_id].isin(test_set)]
        test_df.to_pickle(Constants.ROOT + i[0] + "/test.pickle")
        stats = get_stats(test_df[i[1]].to_list())
        pickle_save(stats, Constants.ROOT + i[0] + "/test_stats")
        print(len(stats))
# generate_data()


# Generate labels for data
def generate_labels(_type='GO-terms (molecular_function)', _name='molecular_function'):
    # ['GO-terms (molecular_function)', 'GO-terms (biological_process)', 'GO-terms (cellular_component)']

    # if not os.path.isfile('/data/pycharm/TransFunData/data/{}.pickle'.format(_name)):

    file = '/data/pycharm/TransFunData/data/nrPDB-GO_2021.01.23_annot.tsv'
    data = pd.read_csv(file, sep='\t', skiprows=12)
    data = data[["### PDB-chain", _type]]
    data = data[data[_type].notna()]

    classes = data[_type].to_list()
    classes = set([one_word for class_list in classes for one_word in class_list.split(',')])
    class_keys = list(range(0, len(classes)))

    classes = dict(zip(classes, class_keys))

    data_to_one_hot = {}
    for index, row in data.iterrows():
        tmp = row[_type].split(',')
        x = torch.tensor([classes[i] for i in tmp])
        x = F.one_hot(x, num_classes=len(classes))
        x = x.sum(dim=0).float()
        assert len(tmp) == x.sum(dim=0).float()
        data_to_one_hot[row['### PDB-chain']] = x.to(dtype=torch.int)
    pickle_save(data_to_one_hot, '/data/pycharm/TransFunData/data/{}'.format(_name))

# generate_labels()

#
# x = pd.read_csv("/data/pycharm/TransFun/nrPDB-GO_2019.06.18_annot.tsv", sep='\t', skiprows=12)
# go_terms = set()
# mf = x['GO-terms (cellular_component)'].to_list()
# for i in mf:
#     if isinstance(i, str):
#         go_terms.update(i.split(','))
# print(len(go_terms))
#
# xx = set(pickle_load(Constants.ROOT + "cellular_component/train_stats"))
# print(len(xx))
#
# print(len(xx.intersection(go_terms)))

# bp = x['GO-terms (biological_process)']
# for i in mf:
#     go_terms.update(i.split(','))
# cc = x['GO-terms (cellular_component)']
# for i in mf:
#     go_terms.update(i.split(','))
