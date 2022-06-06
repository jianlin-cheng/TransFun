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