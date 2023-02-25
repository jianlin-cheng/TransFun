import os
import subprocess

import obonet
import pandas as pd
from torch import optim
import torch
from Bio import SeqIO
from torch_geometric.loader import DataLoader
import Constants
import params111
from Dataset.Dataset import load_dataset
from models.gnn import GCN3
from parser import get_parser

from preprocessing.utils import get_sequence_from_pdb, create_seqrecord, fasta_to_dictionary, load_ckp, pickle_load, \
    pickle_save

pdbs = ["Cre11g467752", "Cre12g505800", "Cre01g041100", "Cre11g467730", "Cre12g523650"]
cut_sequences = []
sequences = []


# Extract sequence from pdbs
def extract_sequences():
    for i in pdbs:
        seq = get_sequence_from_pdb(Constants.ROOT + "inference/struct/AF-{}-F1-model_v2.pdb.gz".format(i), "A")
        sequences.append(create_seqrecord(id="jian|{}|{}".format(i, i), description=i, seq=str(seq)))
        for j in range(int(len(seq) / 1021) + 1):
            tmp = i + "_____" + str(j)
            id = "jian|{}|{}".format(tmp, i)
            cut_seq = str(seq[j * 1021:(j * 1021) + 1021])
            cut_sequences.append(create_seqrecord(id=id, description=i, seq=str(cut_seq)))
    SeqIO.write(cut_sequences, Constants.ROOT + "/inference/{}.fasta".format("jian_cut"), "fasta")
    SeqIO.write(sequences, Constants.ROOT + "/inference/{}.fasta".format("jian"), "fasta")


# extract_sequences()
# exit()

# generate embeddings
def generate_bulk_embedding(fasta_file, output_dir, path_to_extract_file):
    subprocess.call('python extract.py esm1b_t33_650M_UR50S {} {} --repr_layers 0 32 33 '
                    '--include mean per_tok --truncate'.format("{}".format(fasta_file),
                                                               "{}".format(output_dir)),
                    shell=True, cwd="{}".format(path_to_extract_file))


# generate_bulk_embedding(Constants.ROOT + "/inference/{}.fasta".format("jian"),
#                         Constants.ROOT + "/inference/esm",
#                         "/data_bp/pycharm/TransFun/preprocessing")


def merge_pts():
    unique_files = {i.split("_____")[0] for i in pdbs}
    levels = {0, 1}
    embeddings = [0, 32, 33]

    for i in unique_files:
        fasta = fasta_to_dictionary(Constants.ROOT + "inference/jian.fasta")
        tmp = []
        for j in levels:
            os_path = Constants.ROOT + 'inference/esm/{}_____{}.pt'.format(i, j)
            if os.path.isfile(os_path):
                tmp.append(torch.load(os_path))

        data = {'representations': {}, 'mean_representations': {}}
        for index in tmp:
            splits = index['label'].split("|")
            data['label'] = splits[0] + "|" + splits[1].split("_____")[0] + "|" + splits[2]

            for rep in embeddings:
                assert torch.equal(index['mean_representations'][rep], torch.mean(index['representations'][rep], dim=0))

                if rep in data['representations']:
                    data['representations'][rep] = torch.cat(
                        (data['representations'][rep], index['representations'][rep]))
                else:
                    data['representations'][rep] = index['representations'][rep]

        assert len(fasta[i][3]) == data['representations'][33].shape[0]

        for rep in embeddings:
            data['mean_representations'][rep] = torch.mean(data['representations'][rep], dim=0)

        print("saving {}".format(i))
        torch.save(data, Constants.ROOT + "inference/esm/{}.pt".format(i))


# merge_pts()

ontologies = ["molecular_function", "cellular_component", "biological_process"]
# seq_ids = [0.5]
#
# args = get_parser()
# if args.cuda:
#     device = 'cuda'
#
# results = {}
#
# for ontology in ontologies:
#     for seq in seq_ids:
#
#         results[ontology] = {}
#         print("*****-Evaluating Ontology:{} at {}% sequence identity-*****".format(ontology[1], seq * 100))
#
#         args.ont = ontology
#         args.seq = seq
#
#         if args.ont == 'molecular_function':
#             ont_kwargs = params1.mol_kwargs
#         elif args.ont == 'cellular_component':
#             ont_kwargs = params1.cc_kwargs
#         elif args.ont == 'biological_process':
#             ont_kwargs = params1.bio_kwargs
#
#         # model
#         model = GCN3(**ont_kwargs)
#         model.to(device)
#
#         # optimizer
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=ont_kwargs['wd'])
#
#         # define checkpoint saved path
#         ckp_dir = Constants.ROOT + '{}/{}/model_checkpoint/{}/'.format(args.seq, args.ont, ont_kwargs['edge_type'])
#         ckp_pth = ckp_dir + "current_checkpoint.pt"
#
#         # load the saved checkpoint
#         if os.path.exists(ckp_pth):
#             print("Loading model checkpoint @ {}".format(ckp_pth))
#             model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
#         else:
#             print("Model not found. Skipping...")
#             continue
#
#         model.eval()
#
#         kwargs = {
#             'seq_id': 0.5,
#             'ont': 'molecular_function',
#             'session': 'selected',
#             'prot_ids': pdbs
#         }
#         dataset = load_dataset(root=Constants.ROOT, **kwargs)
#
#         num_class = pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}']
#
#         print("start_epoch = ", current_epoch)
#         print("valid_loss_min = ", min_val_loss)
#         print("valid_loss_min = {:.6f}".format(min_val_loss))
#         print("{} proteins available for evaluation".format(len(dataset)))
#         print("Number of classes: {}".format(len(num_class)))
#
#         test_dataloader = DataLoader(dataset,
#                                      batch_size=32,
#                                      drop_last=False,
#                                      shuffle=False)
#
#         probabilities = []
#         proteins = []
#
#         for data_bp in test_dataloader:
#             with torch.no_grad():
#                 proteins.extend(data_bp['atoms'].protein)
#                 probabilities.extend(model(data_bp.to(device)).tolist())
#
#         assert len(proteins) == len(probabilities)
#
#         for protein, score in zip(proteins, probabilities):
#             results[ontology][protein] = {}
#             assert len(score) == len(num_class)
#             for goterm, goscore in zip(num_class, score):
#                 results[ontology][protein][goterm] = goscore
#
#         print(results)
    #pickle_save(results, Constants.ROOT + "Jian/results")
# exit()

filtered = {}
results = pickle_load(Constants.ROOT + "Jian/results")
thresholds = {"cellular_component": 0.42, "molecular_function": 0.33, "biological_process": 0.4}
for ontology in ontologies:
    filtered[ontology] = {}
    for protein in pdbs:
        tmp = {k: v for k, v in results[ontology][protein].items() if v > thresholds[ontology]}

        filtered[ontology][protein] = tmp
print(filtered)

# with open(Constants.ROOT + "Jian/output.txt", 'w') as f:
#     for ontology in filtered:
#         f.write(f"{ontology}\n")
#         for protein in filtered[ontology]:
#             f.write(f"{protein}\n")
#             for go_term in filtered[ontology][protein]:
#                 f.write("{}: {}, ".format(go_term, filtered[ontology][protein][go_term]))
#             f.write(f"\n")

go_graph = obonet.read_obo(open(Constants.ROOT + "obo/go-basic.obo", 'r'))
id_to_name = {id_: data.get('name') for id_, data in go_graph.nodes(data=True)}

formatted = []
for ont in filtered:
    for protein in filtered[ont]:
        for go_term, probability in filtered[ont][protein].items():
            tmp = [protein, go_term, probability, id_to_name[go_term], ont]
            formatted.append(tmp)

df = pd.DataFrame(data=formatted, columns=['Protein', 'Go term', 'Probability', 'GO name', 'Ontology'])
df = df.sort_values(['Protein', 'Ontology'], ascending=[True, False])
df.to_csv("formatted.csv", index=False, sep="\t")