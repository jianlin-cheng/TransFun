import argparse
import os

import networkx as nx
import obonet
import torch
from Bio import SeqIO
from torch import optim
from torch_geometric.loader import DataLoader
import Constants
import params
from Dataset.Dataset import load_dataset
from models.gnn import GCN
from preprocessing.utils import load_ckp, get_sequence_from_pdb, create_seqrecord, get_proteins_from_fasta, \
    generate_bulk_embedding, pickle_load

parser = argparse.ArgumentParser(description=" Predict protein functions with TransFun ", epilog=" Thank you !!!")
parser.add_argument('--data-path', type=str, default="data", help="Path to data files")
parser.add_argument('--ontology', type=str, default="cellular_component", help="Path to data files")
parser.add_argument('--no-cuda', default=False, help='Disables CUDA training.')
parser.add_argument('--batch-size', default=10, help='Batch size.')
parser.add_argument('--input-type', choices=['fasta', 'pdb'], default="fasta",
                    help='Input Data: fasta file or PDB files')
parser.add_argument('--fasta-path', default="sequence.fasta", help='Path to Fasta')
parser.add_argument('--pdb-path', default="alphafold", help='Path to directory of PDBs')
parser.add_argument('--cut-off', type=float, default=0.0, help="Cut of to report function")
parser.add_argument('--output', type=str, default="output", help="File to save output")
parser.add_argument('--add-ancestors', default=False, help="Add ancestor terms to prediction")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = 'cuda'

if args.ontology == 'molecular_function':
    ont_kwargs = params.mol_kwargs
elif args.ontology == 'cellular_component':
    ont_kwargs = params.cc_kwargs
elif args.ontology == 'biological_process':
    ont_kwargs = params.bio_kwargs

FUNC_DICT = {
    'cellular_component': 'GO:0005575',
    'molecular_function': 'GO:0003674',
    'biological_process': 'GO:0008150'
}

print("Predicting proteins")

def create_fasta(proteins):
    fasta = []
    for protein in proteins:
        alpha_fold_seq = get_sequence_from_pdb(args.data_path + "/alphafold/{}.pdb.gz".format(protein), "A")
        fasta.append(create_seqrecord(id=protein, seq=alpha_fold_seq))
    SeqIO.write(fasta, "{}/sequence.fasta".format(args.data_path), "fasta")


def write_to_file(data, output):
    with open('{}.txt'.format(output), 'w') as fp:
        fp.write('\n'.join('%s %s %s' % x for x in data))


if args.input_type == 'fasta':
    if not args.fasta_path is None:
        proteins = set(get_proteins_from_fasta("{}/{}".format(args.data_path, args.fasta_path)))
        pdbs = set([i.split(".")[0] for i in os.listdir("{}/{}".format(args.data_path, args.pdb))])
        proteins = pdbs.intersection(proteins)
elif args.input_type == 'pdb':
    if not args.pdb_path is None:
        pdb_path = "{}/{}".format(args.data_path, args.pdb_path)
        if os.path.exists(pdb_path):
            proteins = os.listdir(pdb_path)
            proteins = [protein.split('.')[0] for protein in proteins if protein.endswith(".pdb.gz")]
            create_fasta(proteins)
        else:
            print("PDB directory not found")
            exit()

if len(proteins) > 0:
    print("Predicting for {} proteins".format(len(proteins)))

print("Generating Embeddings")
os.makedirs("{}/esm".format(args.data_path), exist_ok=True)
if len([]) > 1022:
    pass
else:
    generate_bulk_embedding("./preprocessing/extract.py", "{}/{}".format(args.data_path, args.fasta),
                            "{}/esm".format(args.data_path))

kwargs = {
    'seq_id': Constants.Final_thresholds[args.ontology],
    'ont': args.ontology,
    'session': 'selected',
    'prot_ids': proteins
}

dataset = load_dataset(root=args.data_path, **kwargs)

test_dataloader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             drop_last=False,
                             shuffle=False)

# model
model = GCN(**ont_kwargs)
model.to(device)

optimizer = optim.Adam(model.parameters())

ckp_pth = "{}/{}".format(args.data_path, args.ontology)

# load the saved checkpoint
if os.path.exists(ckp_pth):
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
else:
    print("Model not found. Skipping...")
    exit()

model.eval()

scores = []
proteins = []

for data in test_dataloader:
    with torch.no_grad():
        proteins.extend(data['atoms'].protein)
        scores.extend(model(data.to(device)).tolist())

assert len(proteins) == len(scores)

goterms = pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ontology}']
go_graph = obonet.read_obo(open("{}/go-basic.obo".format(args.data_path), 'r'))
go_set = nx.ancestors(go_graph, FUNC_DICT[args.ontology])

results = []
for protein, score in zip(proteins, scores):
    protein_terms = []
    tmp = set()
    for go_term, _score in zip(goterms, score):
        if _score > args.cut_off:
            results.append((protein, go_term, _score))
            protein_terms.append((go_term, _score))
            tmp.add(go_term)

    if args.add_ancestors:
        for i, j in protein_terms:
            ansc = nx.ancestors(go_graph, i).intersection(go_set)
            # remove those predicted already
            ansc = ansc.difference(tmp)
            for _term in ansc:
                results.append((protein, _term, j))
                tmp.add(_term)

write_to_file(results, "{}/{}".format(args.data_path, args.output))
