from pathlib import Path
import torch
from Bio import SeqIO
from biopandas.pdb import PandasPdb
from torch.utils.data import Dataset
from utils import find_files


class PDBDataset(Dataset):
    """
    Creates a dataset from a list of PDB files.
    :param file_list: path to LMDB file containing dataset
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    """

    def __init__(self, file_list, transform=None, store_file_path=True):
        """constructor
        """
        self._file_list = [Path(x).absolute() for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._store_file_path = store_file_path

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        file_path = self._file_list[index]
        print(file_path)

        # if file_exist thus if list of paths is passed
        pdb_to_pandas = PandasPdb().read_pdb(str(file_path))
        item = {
            'pdb_to_pandas': pdb_to_pandas,
            'protein_id': file_path.name,
            'label': 'label',
            'chain_id': 'A',
            'metric': 'knn'
        }
        if self._store_file_path:
            item['file_path'] = str(file_path)
        if self._transform:
            item = self._transform(item)
        return item


def load_dataset(path, filetype, transform=None):
    """
    Load files in file_list into corresponding dataset object. All files should be of type filetype.

    :param file_list: List containing paths to files. Assumes one structure per file.
    :type file_list: list[Union[str, Path]]
    :param filetype: Type of dataset. Allowable types are 'mmcif', 'pdb'.
    :type filetype: str
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional

    :return: Pytorch Dataset containing data
    :rtype: torch.utils.data.Dataset
    """

    file_list = find_files(path, filetype)

    if (filetype == 'pdb') or (filetype == 'pdb.gz'):
        dataset = PDBDataset(file_list, transform=transform)
    elif filetype == 'mmcif':
        raise RuntimeError('mmcif not implemented yet')
    else:
        raise RuntimeError(f'Unrecognized filetype {filetype}.')
    return dataset