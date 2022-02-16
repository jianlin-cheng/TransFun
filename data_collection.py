import os, subprocess
import pandas as pd

# text url file directory
files_to_download = [('GO identifier(s)', 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_go.tsv.gz', 'pdb_chain_go.csv', 'data'),
                     ('EC number(s)', 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_enzyme.tsv.gz', 'pdb_chain_enzyme.csv', 'data'),
                     ('Alpha fold mmcif Swiss-Prot', 'https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_cif_v2.tar', 'swissprot_cif_v2', 'data/alphafold'),
                     ('Uniprot fasta file', '')]


def download_files(files_to_download):
    for i in files_to_download:
        if not os.path.isfile('{}/{}'.format(i[3], i[2])):
            # note downloading a ~1Gb file can take a minute
            print('Downloading {}'.format(i[0]))
            subprocess.call(
                'wget -O {}/{}.gz {}'.format(i[3], i[2], i[1]),
                shell=True)
            subprocess.call('gunzip {}/{}.gz'.format(i[3], i[2]), shell=True)



def download_pdb_files(file):
    x = pd.read_csv(file, sep='\t',  skiprows=1)
    pdb_ids = x['PDB'].tolist()
    for i in pdb_ids:
        if not os.path.isfile('data/pdb/{}.pdb'.format(i)):
            # note downloading a ~1Gb file can take a minute
            print('Downloading pdb file for {}'.format(i))
            subprocess.call(
                'wget -O data/pdb/{}.pdb https://files.rcsb.org/download/{}.pdb1'.format(i, i),
                shell=True)


# def create

# download_pdb_files('data/pdb_chain_go.csv')
# download_pdb_files('data/pdb_chain_enzyme.csv')
download_files(files_to_download[2:])

