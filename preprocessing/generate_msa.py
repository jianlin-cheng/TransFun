# Tutorial to understand msa
import os
import subprocess

import ray
from biotransformers import BioTransformers


def download_msa_database(url, name):
    database_path = "./msa/hh_suite_database/{}".format(name)
    if not os.path.isdir(database_path):
        os.mkdir(database_path)
        # download database, note downloading a ~1Gb file can take a minute
        database_file = "{}/{}.tar.gz".format(database_path, name)
        subprocess.call('wget -O {} {}'.format(database_file, url), shell=True)
        # unzip the database
        subprocess.call('tar xzvf {}.tar.gz'.format(name), shell=True, cwd="{}".format(database_path))
'''
    Download some database. 
'''
dbCAN = ("http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/dbCAN-fam-V9.tar.gz", "dbCAN-fam-V9")
# download_msa_database(dbCAN[0], dbCAN[1])


def search_database(file, database):
    base_path = "./msa/{}"
    output_path = base_path.format("outputs/{}.hhr".format(file))
    input_path = base_path.format("inputs/{}.fasta".format(file))
    oa3m_path = base_path.format("oa3ms/{}.03m".format(file))
    database_path = base_path.format("hh_suite_database/{}/{}".format(database, database))
    if not os.path.isfile(oa3m_path):
        subprocess.call('hhblits -i {} -o {} -oa3m {} -d {} -cpu 4 -n 1'.format(input_path, output_path, oa3m_path, database_path), shell=True)


test_file = "query"
database = "pfamA_35.0"
search_database(test_file, database)


# BioTransformers.list_backend()
#
# sequences = [
#         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
#         "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
#     ]
#
# ray.init()
# bio_trans = BioTransformers(backend="protbert_bfd", num_gpus=1)
# embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls', 'mean'), batch_size=2)
#
# print(embeddings['cls'])
# print(embeddings['mean'])