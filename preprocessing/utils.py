import math
import os, subprocess
import pandas as pd
from Bio import SeqIO
import pickle
from biopandas.pdb import PandasPdb
from collections import deque, Counter
import csv
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



# Count the number of protein sequences in a fasta file with biopython -- slower.
def count_proteins_biopython(fasta_file):
    num = len(list(SeqIO.parse(fasta_file, "fasta")))
    return num


def fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data[seq_record.id.split("|")[1]] = str(seq_record.seq)
    return data


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)



def download_pdb_files(file):
    chunksize = 100
    with pd.read_csv(file, chunksize=chunksize, sep='\t', skiprows=1) as reader:
        for chunk in reader:
            print(chunk.head(10))
            pdb_ids = chunk['PDB'].tolist()
            print(pdb_ids)

    pdb_ids = x['PDB'].tolist()
    for i in pdb_ids:
        if not os.path.isfile('data/pdb/GO/{}.pdb'.format(i)):
            # note downloading a ~1Gb file can take a minute
            print('Downloading pdb file for {}'.format(i))
            subprocess.call(
                'wget -O data/pdb/GO/{}.pdb https://files.rcsb.org/download/{}.pdb1'.format(i, i),
                shell=True)


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
    dirs = glob("/data/fasta_files/{}/*/".format(group), recursive=False)
    for i in enumerate(dirs):
        prt = i[1].split('/')[4]
        if int(i[0]) % 100 == 0:
            current = "/data/fasta_files/{}/{}".format(group, (int(i[0]) // 100))
            if not os.path.isdir(current):
                os.mkdir(current)
        old = "/data/fasta_files/{}/{}".format(group, prt)
        new = current + "/{}".format(prt)
        if old != current:
            os.rename(old, new)


# Just used to group msas to keep track of generation
def fasta_for_msas(proteins, fasta_file):

    root_dir = '/data/uniprot/'
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    num_protein = 0
    for record in input_seq_iterator:
        if num_protein%200 == 0:
            parent_dir = root_dir + str(int(num_protein/200))
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


def is_ok(seq, MINLEN=49, MAXLEN=1001):
    """
           Checks if sequence is of good quality
           :param MAXLEN:
           :param MINLEN:
           :param seq:
           :return: None
           """
    if len(seq) < MINLEN or len(seq) > MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def is_cafa_target(org):
    return org in Constants.CAFA_TARGETS


def is_exp_code(code):
    return code in Constants.exp_evidence_codes


class Ontology(object):

    def __init__(self, filename=Constants.ROOT + 'obo/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set