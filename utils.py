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


pdb_id = '102l'
pdb_mapping_url = 'https://data.rcsb.org/rest/v1/core/uniprot/{}/1'
uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'

def get_uniprot_protein_name(uniport_id):
    uinprot_response = requests.get(uniprot_url.format(uniport_id)).text
    return ET.fromstring(uinprot_response).find('.//{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName').text

def map_pdb_to_uniprot(pdb_id):
    pdb_mapping_response = requests.get(pdb_mapping_url.format(pdb_id)).json()
    print(pdb_mapping_response)
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
    uniprot_fasta_file = 'data/uniprot_sprot.fasta'
    pdb_fasta = ''
    pdb_api_fasta = ''
    seqres_fasta_file = 'data/pdb_seqres.txt'

    mylist = get_proteins_from_seqres_fasta(seqres_fasta_file, use_biopython=False)

    import pickle
    with open('parrot.pkl', 'wb') as f:
        pickle.dump(mylist, f)
    print(get_proteins_from_uniprot_fasta(uniprot_fasta_file, use_biopython=False)[:10])
    print(map_pdb_to_uniprot(pdb_id))
    pass


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
            exit()


    exit()
    pdb_ids = x['PDB'].tolist()
    for i in pdb_ids:
        if not os.path.isfile('data/pdb/GO/{}.pdb'.format(i)):
            # note downloading a ~1Gb file can take a minute
            print('Downloading pdb file for {}'.format(i))
            subprocess.call(
                'wget -O data/pdb/GO/{}.pdb https://files.rcsb.org/download/{}.pdb1'.format(i, i),
                shell=True)

download_pdb_files('data/pdb_chain_go.csv')
# download_pdb_files('data/pdb_chain_enzyme.csv')

