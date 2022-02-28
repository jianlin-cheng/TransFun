# Tutorial to understand msa
import os
import subprocess

import ray
from biotransformers import BioTransformers



'''
    Download some database. 
'''
# dbCAN = ("http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/dbCAN-fam-V9.tar.gz", "dbCAN-fam-V9")
# download_msa_database(dbCAN[0], dbCAN[1])
# pdb70 = ("http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pdb70_from_mmcif_latest.tar.gz", "pdb70_from_mmcif_latest")
# download_msa_database(pdb70[0], pdb70[1])



test_file = "query"
database = "pfamA_35.0"
# search_database(test_file, database)

# exit()
BioTransformers.list_backend()
msa_folder = "./msa/oa3ms"
bio_trans = BioTransformers("esm_msa1_t12_100M_UR50S", num_gpus=1)
msa_embeddings = bio_trans.compute_embeddings(sequences=msa_folder, pool_mode=("cls", "mean"), n_seqs_msa=128)


proteins = [i[0][0] for i in msa_embeddings['sequence']]
seq_length = msa_embeddings['lengths']
shape = msa_embeddings['cls']
print("Example with esm")
print(proteins)
print(seq_length)
print(shape.shape)
# for i in msa_embeddings:
#     print(msa_embeddings[i])



sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

ray.init()
bio_trans = BioTransformers(backend="protbert_bfd", num_gpus=1)
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls', 'mean'), batch_size=2)
print("Example with esm")
print(embeddings['cls'].shape)
# print(embeddings['mean'].shape)

exit()