import pickle
import re

from Bio import SeqIO
import requests
import xml.etree.ElementTree as ET

# Count the number of protein sequences in a fasta file.
def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    print(num)

# Count the number of protein sequences in a fasta file with biopython -- slower.
def count_proteins_biopython(fasta_file):
    num = len(list(SeqIO.parse(fasta_file, "fasta")))
    print(num)


def get_proteins_from_uniprot_fasta(fasta_file, use_biopython=False):
    names = []
    if use_biopython:
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            seq_name = seq_record.name
            offset = len(">sp|")
            start = seq_name.find(">sp|") + offset
            end = seq_name.find("|", offset)
            names.append(seq_name[start:end])
    else:
        for line in open(fasta_file):
            if line.startswith(">"):
                offset = len(">sp|")
                start = line.find(">sp|") + offset
                end = line.find("|", offset)
                names.append(line[start:end])
                print(names)
                exit()
    return names


def search_sequence(fasta_file, prt_seq):
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        print(seq_record)
        exit()


def extract_seq(pdb_file_name):
    '''
        Extract fasta sequence from pdb file.
    :param: pdb_file: pdb file name,
    :return:
    '''
    with open(pdb_file_name, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            print('>' + record.id)
            print(str(record.seq))

def seqres_vs_uniprot():
    pass


pdb_mapping_url = 'https://data.rcsb.org/rest/v1/core/uniprot/{}/1'
uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'

def get_uniprot_protein_name(uniport_id):
    uinprot_response = requests.get(uniprot_url.format(uniport_id)).text
    return ET.fromstring(uinprot_response).find('.//{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName').text

def map_pdb_to_uniprot(pdb_id):
    pdb_mapping_response = requests.get(pdb_mapping_url.format(pdb_id)).json()
    try:
        pdb_mapping_response = pdb_mapping_response[0]
        uniprot_id = pdb_mapping_response['rcsb_uniprot_accession'][0]
        # uniprot_name = get_uniprot_protein_name(uniprot_id)
    except KeyError as ke:
        pdb_id, uniprot_id, uniprot_name = 'None', 'None', 'None'

    return {
        'pdb_id': pdb_id,
        'uniprot_id': uniprot_id,
        #'uniprot_name': uniprot_name
    }


def get_proteins_from_seqres_fasta(fasta_file, use_biopython=False):
    names = []
    if use_biopython:
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            seq_name = seq_record.name
            offset = len(">")
            start = seq_name.find(">") + offset
            end = seq_name.find("_", offset)
            names.append(map_pdb_to_uniprot(seq_name[start:end])['uniprot_id'])
    else:
        for line in open(fasta_file):
            if line.startswith(">"):
                offset = len(">")
                start = line.find(">") + offset
                end = line.find("_", offset)
                tmp = map_pdb_to_uniprot(line[start:end])
                names.append(tmp['uniprot_id'])
    return names


def compare_fasta_sequences():
    '''
        This script was used to compare various sequences. 1. from the pdb api, uniprot sequence and pdb file.
    :return:
    '''
    uniprot_fasta_file = '../TransFunData/data/uniprot_sprot.fasta'
    pdb_fasta = ''
    pdb_api_fasta = ''
    seqres_fasta_file = '../TransFunData/data/pdb_seqres.txt'

    mylist = get_proteins_from_seqres_fasta(seqres_fasta_file, use_biopython=False)


    # import pickle
    # with open('parrot.pkl', 'wb') as f:
    #     pickle.dump(mylist, f)
    print(get_proteins_from_uniprot_fasta(uniprot_fasta_file, use_biopython=False))
    # print(map_pdb_to_uniprot(pdb_id))
    pass


def filter():
    uniprot_fasta_file = '../TransFunData/data/uniprot_sprot.fasta'
    pdb_fasta = ''
    pdb_api_fasta = ''
    seqres_fasta_file = '../TransFunData/data/pdb_seqres.txt'



uniprot_fasta_file = 'data/uniprot_sprot.fasta'
seqres_fasta_file = 'data/pdb_seqres.txt'
pdb_file = 'data/101m.pdb'
# count_proteins(fasta_file)
# count_proteins_biopython(fasta_file)
# search_sequence(fasta_file, "")
# extract_seq(pdb_file)


# compare_fasta_sequences()



import os, subprocess
import pandas as pd

def download_pdb_files(file):

    chunksize = 100

    with pd.read_csv(file, chunksize=chunksize, sep='\t',  skiprows=1) as reader:
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

# download_pdb_files('data/pdb_chain_go.csv')
# download_pdb_files('data/pdb_chain_enzyme.csv')

def compare_pdb_seqres(input_pdb, input_seqres):
    from Bio import SeqIO
    with open(input_pdb, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            print('>' + record.id)
            print(record.seq)


# compare_pdb_seqres("../TransFunData/data/101m.pdb", "")

def selected():
    df = pd.read_csv("../TransFunData/data/nrPDB-GO_2021.01.23_annot.tsv", sep='\t', skiprows=12)
    chains = set(df['### PDB-chain'].to_list())
    print(len(chains))
    for i in chains:
        file_path = "/data/fasta_files/{}.fasta".format(i)
        dir_path = "/data/fasta_files/{}".format(i)
        new_path = "/data/fasta_files/{}/{}.fasta".format(i, i)
        if os.path.isfile(file_path):
            os.mkdir(dir_path)
            os.rename(file_path, new_path)
    # print(len(chains))
# selected()


# def check_msa_equality():
#     sx = set()
#     file = open("preprocessing/msa/outputs/mgnify_hits.a3m")
#     lines = [line.strip('\n') for line in file.readlines() if line.strip()]
#     file.close()
#     filtere1 = []
#     empty = ""
#     for i in lines:
#         if i.startswith(">"):
#             if empty !="":
#                 filtere1.append(empty)
#             empty = ""
#         else:
#             empty = empty+i
#
#
#     file = open("preprocessing/msa/outputs/small_bfd_hits.a3m")
#     lines = [line.strip('\n') for line in file.readlines() if line.strip()]
#     file.close()
#     filtere2 = []
#     empty = ""
#     for i in lines:
#         if i.startswith(">"):
#             if empty != "":
#                 filtere2.append(empty)
#             empty = ""
#         else:
#             empty = empty + i
#
#
#     file = open("preprocessing/msa/outputs/uniref90_hits.a3m")
#     lines = [line.strip('\n') for line in file.readlines() if line.strip()]
#     file.close()
#     filtere3 = []
#     empty = ""
#     for i in lines:
#         if i.startswith(">"):
#             if empty != "":
#                 filtere3.append(empty)
#             empty = ""
#         else:
#             empty = empty + i
#
#     sx.update(filtere1, filtere2, filtere3)
#
#
#     with open('mine.pickle', 'wb') as handle:
#         pickle.dump(sx, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# # check_msa_equality()
#
# with open('mine.pickle', 'rb') as handle:
#     a = pickle.load(handle)
#
# with open('alp.pickle', 'rb') as handle:
#     b = pickle.load(handle)
#
# bo = []
# for i in a:
#     pattern = '[a-z]+'
#     if re.search(pattern, i):
#         croped = re.sub("[a-z]+", "", i)
#         bo.append(croped)
#     else:
#         bo.append(i)
#
#
# a = set(bo)
# b = set(b)
# x = a.difference(b)
# y = b.difference(a)
# print(len(x), len(y))
#
# print(y)

def partition_files(group):
    from glob import glob
    dirs = glob("/data/fasta_files/{}/*/".format(group), recursive=False)
    for i in enumerate(dirs):
        prt = i[1].split('/')[4]
        if int(i[0])%100 == 0:
            current = "/data/fasta_files/{}/{}".format(group, (int(i[0])//100))
            if not os.path.isdir(current):
                os.mkdir(current)
        old = "/data/fasta_files/{}/{}".format(group, prt)
        new = current+"/{}".format(prt)
        if old != current:
            os.rename(old, new)


partition_files(8)
