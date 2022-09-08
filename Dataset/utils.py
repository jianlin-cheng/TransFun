import math
import os
import re
import subprocess
from pathlib import Path
import pickle

import numpy as np
import torch
from biopandas.pdb import PandasPdb
import torch.nn.functional as F
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

import Constants
from Constants import residues, amino_acids


def get_input_data():
    # test data
    input = ["1a0b", "1a0c", "1a0d", "1a0e", "1a0f", "1a0g", "1a0h", "1a0i", "1a0j", "1a0l"]
    raw = [s + ".pdb" for s in input]
    processed = [s + ".pt" for s in input]

    with open('../Dataset/raw.pickle', 'wb') as handle:
        pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../Dataset/proceesed.pickle', 'wb') as handle:
        pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)


patterns = {
    'pdb': r'pdb[0-9]*$',
    'pdb.gz': r'pdb[0-9]*\.gz$',
    'mmcif': r'(mm)?cif$',
    'sdf': r'sdf[0-9]*$',
    'xyz': r'xyz[0-9]*$',
    'xyz-gdb': r'xyz[0-9]*$',
    'silent': r'out$',
    'sharded': r'@[0-9]+',
}

_regexes = {k: re.compile(v) for k, v in patterns.items()}


def is_type(f, filetype):
    if filetype in _regexes:
        return _regexes[filetype].search(str(f))
    else:
        return re.compile(filetype + r'$').search(str(f))


def find_files(path, suffix, relative=None, type="Path"):
    """
    Find all files in path with given suffix. =

    :param path: Directory in which to find files.
    :type path: Union[str, Path]
    :param suffix: Suffix determining file type to search for.
    :type suffix: str
    :param relative: Flag to indicate whether to return absolute or relative path.

    :return: list of paths to all files with suffix sorted by their names.
    :rtype: list[str]
    """
    if not relative:
        find_cmd = r"find {:} -regex '.*\.{:}' | sort".format(path, suffix)
    else:
        find_cmd = r"cd {:}; find . -regex '.*\.{:}' | cut -d '/' -f 2- | sort" \
            .format(path, suffix)
    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.getcwd(), shell=True)
    (stdout, stderr) = out.communicate()
    name_list = stdout.decode().split()
    name_list.sort()
    if type == "Path":
        return sorted([Path(x) for x in name_list])
    elif type == "Name":
        return sorted([Path(x).name for x in name_list])


def process_pdbpandas(raw_path, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(raw_path)

    pdb_df = pdb_to_pandas.df['ATOM']
    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & (pdb_df['chain_id'] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    _residues = pdb_df['residue_name'].to_list()
    _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    sequence_features = [[residues[residue] for residue in _residues]]

    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')

    # sequences + padding
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))
    # sequence_features = F.one_hot(sequence_features, num_classes=len(residues) + 1).to(dtype=torch.int64)

    node_coords = torch.tensor(pdb_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    return node_coords, sequence_features, ''.join(_residues)

    return residues


def generate_Identity_Matrix(shape, sequence):

    node_coords = torch.from_numpy(np.zeros(shape=(shape[0], 3)))
    _residues = sequence[3]

    # _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    sequence_features = [[residues[residue] for residue in list(_residues) if residue not in Constants.INVALID_ACIDS]]
    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')
    # sequences + padding
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))
    # sequence_features = F.one_hot(sequence_features, num_classes=len(residues) + 1).to(dtype=torch.int64)
    return node_coords, sequence_features, str(_residues)


def get_cbrt(a):
    return a**(1./3.)


def get_knn(**kwargs):
    mode = kwargs["mode"]
    seq_length = kwargs["sequence_length"]
    if mode == "sqrt":
        x = int(math.sqrt(seq_length))
        if x % 2 == 0:
            return x + 1
        return x
    elif mode == "cbrt":
        x = int(get_cbrt(seq_length))
        if x % 2 == 0:
            return x + 1
        return x
    else:
        return seq_length