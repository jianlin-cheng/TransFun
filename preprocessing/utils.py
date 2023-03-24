import math
import os, subprocess
import shutil

import pandas as pd
import torch
from Bio import SeqIO
import pickle

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from biopandas.pdb import PandasPdb
from collections import deque, Counter
import csv

from sklearn.metrics import roc_curve, auc
from torchviz import make_dot

import Constants
from Constants import INVALID_ACIDS, amino_acids


def extract_id(header):
    return header.split('|')[1]


def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num


def read_dictionary(file):
    reader = csv.reader(open(file, 'r'), delimiter='\t')
    d = {}
    for row in reader:
        k, v = row[0], row[1]
        d[k] = v
    return d


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


# Count the number of protein sequences in a fasta file with biopython -- slower.
def count_proteins_biopython(fasta_file):
    num = len(list(SeqIO.parse(fasta_file, "fasta")))
    return num


def get_proteins_from_fasta(fasta_file):
    proteins = list(SeqIO.parse(fasta_file, "fasta"))
    # proteins = [i.id.split("|")[1] for i in proteins]
    proteins = [i.id for i in proteins]
    return proteins


def fasta_to_dictionary(fasta_file, identifier='protein_id'):
    if identifier == 'protein_id':
        loc = 1
    elif identifier == 'protein_name':
        loc = 2
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        if "|" in seq_record.id:
            data[seq_record.id.split("|")[loc]] = (
            seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
        else:
            data[seq_record.id] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def cafa_fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data[seq_record.description.split(" ")[0]] = (
        seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def alpha_seq_fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        _protein = seq_record.id.split(":")[1].split("-")[1]
        data[_protein] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def download_msa_database(url, name):
    database_path = "./msa/hh_suite_database/{}".format(name)
    if not os.path.isdir(database_path):
        os.mkdir(database_path)
        # download database, note downloading a ~1Gb file can take a minute
        database_file = "{}/{}.tar.gz".format(database_path, name)
        subprocess.call('wget -O {} {}'.format(database_file, url), shell=True)
        # unzip the database
        subprocess.call('tar xzvf {}.tar.gz'.format(name), shell=True, cwd="{}".format(database_path))


def search_database(file, database):
    base_path = "./msa/{}"
    output_path = base_path.format("outputs/{}.hhr".format(file))
    input_path = base_path.format("inputs/{}.fasta".format(file))
    oa3m_path = base_path.format("oa3ms/{}.03m".format(file))
    database_path = base_path.format("hh_suite_database/{}/{}".format(database, database))
    if not os.path.isfile(oa3m_path):
        subprocess.call(
            'hhblits -i {} -o {} -oa3m {} -d {} -cpu 4 -n 1'.format(input_path, output_path, oa3m_path, database_path),
            shell=True)


# Just used to group msas to keep track of generation
def partition_files(group):
    from glob import glob
    dirs = glob("/data_bp/fasta_files/{}/*/".format(group), recursive=False)
    for i in enumerate(dirs):
        prt = i[1].split('/')[4]
        if int(i[0]) % 100 == 0:
            current = "/data_bp/fasta_files/{}/{}".format(group, (int(i[0]) // 100))
            if not os.path.isdir(current):
                os.mkdir(current)
        old = "/data_bp/fasta_files/{}/{}".format(group, prt)
        new = current + "/{}".format(prt)
        if old != current:
            os.rename(old, new)


# Just used to group msas to keep track of generation
def fasta_for_msas(proteins, fasta_file):
    root_dir = '/data_bp/uniprot/'
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    num_protein = 0
    for record in input_seq_iterator:
        if num_protein % 200 == 0:
            parent_dir = root_dir + str(int(num_protein / 200))
            print(parent_dir)
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)
        protein = extract_id(record.id)
        if protein in proteins:
            protein_dir = parent_dir + '/' + protein
            if not os.path.exists(protein_dir):
                os.mkdir(protein_dir)
            SeqIO.write(record, protein_dir + "/{}.fasta".format(protein), "fasta")


# Files to generate esm embedding for.
def fasta_for_esm(proteins, fasta_file):
    protein_path = Constants.ROOT + "uniprot/{}.fasta".format("filtered")
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")

    filtered_seqs = [record for record in input_seq_iterator if extract_id(record.id) in proteins]

    if not os.path.exists(protein_path):
        SeqIO.write(filtered_seqs, protein_path, "fasta")


def get_sequence_from_pdb(pdb_file, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(pdb_file)

    pdb_df = pdb_to_pandas.df['ATOM']

    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & ((pdb_df['chain_id'])[0] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    residues = pdb_df['residue_name'].to_list()
    residues = ''.join([amino_acids[i] for i in residues if i != "UNK"])
    return residues


def is_ok(seq, MINLEN=49, MAXLEN=1022):
    """
           Checks if sequence is of good quality
           :param MAXLEN:
           :param MINLEN:
           :param seq:
           :return: None
           """
    if len(seq) < MINLEN or len(seq) >= MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def is_cafa_target(org):
    return org in Constants.CAFA_TARGETS


def is_exp_code(code):
    return code in Constants.exp_evidence_codes


def read_test_set(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split("\t")[0] for line in lines]
    return lines


def read_test_set_x(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split("\t") for line in lines]
    return lines


def read_test(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    return lines


def collect_test():
    cafa3 = pickle_load(Constants.ROOT + "test/test_proteins_list")
    cafa3 = set([i[0] for i in cafa3])

    new_test = set()
    for ts in Constants.TEST_GROUPS:
        # tmp = read_test_set(Constants.ROOT + "test/195-200/{}".format(ts))
        # total_test.update(set([i[0] for i in tmp]))
        tmp = read_test_set(Constants.ROOT + "test/205-now/{}".format(ts))
        new_test.update(set([i[0] for i in tmp]))

    return cafa3, new_test


def test_annotation():
    # Add annotations for test set
    data = {}
    for ts in Constants.TEST_GROUPS:
        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/195-200/{}".format(ts))
        for i in tmp:
            if i[0] in data:
                data[i[0]][ts].add(i[1])
            else:
                data[i[0]] = {'LK_bpo': set(), 'LK_mfo': set(), 'LK_cco': set(), 'NK_bpo': set(), 'NK_mfo': set(),
                              'NK_cco': set()}
                data[i[0]][ts].add(i[1])

        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/205-now/{}".format(ts))
        for i in tmp:
            if i[0] in data:
                data[i[0]][ts].add(i[1])
            else:
                data[i[0]] = {'LK_bpo': set(), 'LK_mfo': set(), 'LK_cco': set(), 'NK_bpo': set(), 'NK_mfo': set(),
                              'NK_cco': set()}
                data[i[0]][ts].add(i[1])

    return data


# GO terms for test set.
def get_test_classes():
    data = set()
    for ts in Constants.TEST_GROUPS:
        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/195-200/{}".format(ts))
        for i in tmp:
            data.add(i[1])

        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/205-now/{}".format(ts))
        for i in tmp:
            data.add(i[1])

    return data


def create_cluster(seq_identity=None):
    def get_position(row, pos, column, split):
        primary = row[column].split(split)[pos]
        return primary

    computed = pd.read_pickle(Constants.ROOT + 'uniprot/set1/swissprot.pkl')
    computed['primary_accession'] = computed.apply(lambda row: get_position(row, 0, 'accessions', ';'), axis=1)
    annotated = pickle_load(Constants.ROOT + "uniprot/anotated")

    def max_go_terms(row):
        members = row['cluster'].split('\t')
        largest = 0
        max = 0
        for index, value in enumerate(members):
            x = computed.loc[computed['primary_accession'] == value]['prop_annotations'].values  # .tolist()
            if len(x) > 0:
                if len(x[0]) > largest:
                    largest = len(x[0])
                    max = index
        return members[max]

    if seq_identity is not None:
        src = "/data_bp/pycharm/TransFunData/data_bp/uniprot/set1/mm2seq_{}/max_term".format(seq_identity)
        if os.path.isfile(src):
            cluster = pd.read_pickle(src)
        else:
            cluster = pd.read_csv("/data_bp/pycharm/TransFunData/data_bp/uniprot/set1/mm2seq_{}/final_clusters.tsv"
                                  .format(seq_identity), names=['cluster'], header=None)

            cluster['rep'] = cluster.apply(lambda row: get_position(row, 0, 'cluster', '\t'), axis=1)
            cluster['max'] = cluster.apply(lambda row: max_go_terms(row), axis=1)
            cluster.to_pickle("/data_bp/pycharm/TransFunData/data_bp/uniprot/set1/mm2seq_{}/max_term".format(seq_identity))

        cluster = cluster['max'].to_list()
        computed = computed[computed['primary_accession'].isin(cluster)]

    return computed


def class_distribution_counter(**kwargs):
    """
        Count the number of proteins for each GO term in training set.
    """
    data = pickle_load(Constants.ROOT + "{}/{}/{}".format(kwargs['seq_id'], kwargs['ont'], kwargs['session']))

    all_proteins = []
    for i in data:
        all_proteins.extend(data[i])

    annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
    annot = annot.where(pd.notnull(annot), None)
    annot = annot[annot['Protein'].isin(all_proteins)]
    annot = pd.Series(annot[kwargs['ont']].values, index=annot['Protein']).to_dict()

    terms = []
    for i in annot:
        terms.extend(annot[i].split(","))

    counter = Counter(terms)

    # for i in counter.most_common():
    #     print(i)
    # print("# of ontologies is {}".format(len(counter)))

    return counter


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data_bp to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer, device):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device(device))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def draw_architecture(model, data_batch):
    '''
    Draw the network architecture.
    '''
    output = model(data_batch)
    make_dot(output, params=dict(model.named_parameters())).render("rnn_lstm_torchviz", format="png")


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def generate_bulk_embedding(path_to_extract_file, fasta_file, output_dir):
    subprocess.call('python {} esm1b_t33_650M_UR50S {} {} --repr_layers 0 32 33 '
                    '--include mean per_tok --truncate'.format(path_to_extract_file,
                                                               "{}".format(fasta_file),
                                                               "{}".format(output_dir)),
                    shell=True)