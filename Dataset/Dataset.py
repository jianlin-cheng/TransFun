import pickle
from pathlib import Path
import torch
import os.path as osp
from Bio import SeqIO
from biopandas.pdb import PandasPdb
# from torch.utils.data import Dataset
from torch_geometric.data import Dataset, download_url
from utils import find_files

#
# class PDBDataset(Dataset):
#     """
#     Creates a dataset from a list of PDB files.
#     :param file_list: path to LMDB file containing dataset
#     :type file_list: list[Union[str, Path]]
#     :param transform: transformation function for data augmentation, defaults to None
#     :type transform: function, optional
#     """
#
#     def __init__(self, file_list, transform=None, store_file_path=True):
#         """constructor
#         """
#         self._file_list = [Path(x).absolute() for x in file_list]
#         self._num_examples = len(self._file_list)
#         self._transform = transform
#         self._store_file_path = store_file_path
#
#     def __len__(self) -> int:
#         return self._num_examples
#
#     def __getitem__(self, index: int):
#         if not 0 <= index < self._num_examples:
#             raise IndexError(index)
#
#         file_path = self._file_list[index]
#         print(file_path)
#
#         # if file_exist thus if list of paths is passed
#         pdb_to_pandas = PandasPdb().read_pdb(str(file_path))
#         item = {
#             'pdb_to_pandas': pdb_to_pandas,
#             'protein_id': file_path.name,
#             'label': 'label',
#             'chain_id': 'A',
#             'metric': 'knn'
#         }
#         if self._store_file_path:
#             item['file_path'] = str(file_path)
#         if self._transform:
#             item = self._transform(item)
#         return item



class PDBDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, download_url=""):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.url = download_url
        with open('filename.pickle', 'rb') as handle:
            self.raw_file_list = pickle.load(handle)
        with open('filename.pickle', 'rb') as handle:
            self.processed_file_list = pickle.load(handle)

    @property
    def raw_file_names(self):
        return self.raw_file_names

    @property
    def processed_file_names(self):
        return self.processed_file_list

    def download(self):
        path = download_url(self.url, self.raw_dir)

    def process(self):
        pass
        # idx = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        #     idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

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

dataset = load_dataset()
print(dataset)