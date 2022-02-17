import os, subprocess
import pandas as pd


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
        subprocess.call('hhblits -i {} -o {} -oa3m {} -d {} -cpu 4 -n 1'.format(input_path, output_path, oa3m_path, database_path), shell=True)



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
# partition_files(8)