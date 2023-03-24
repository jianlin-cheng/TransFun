import math
import os
import pickle
import subprocess
import torch
import os.path as osp
from torch_geometric.data import Dataset, download_url, HeteroData

import Constants
from Dataset.distanceTransform import myDistanceTransform
from Dataset.myKnn import myKNNGraph
from Dataset.myRadiusGraph import myRadiusGraph
from Dataset.utils import find_files, process_pdbpandas, get_knn, generate_Identity_Matrix
import torch_geometric.transforms as T
from torch_geometric.data import Data
from Dataset.AdjacencyTransform import AdjacencyFeatures
from preprocessing.utils import pickle_load, pickle_save, get_sequence_from_pdb, fasta_to_dictionary, collect_test, \
    read_test_set, read_test, cafa_fasta_to_dictionary
import pandas as pd
import random


class PDBDataset(Dataset):
    """
        Creates a dataset from a list of PDB files.
        :param file_list: path to LMDB file containing dataset
        :type file_list: list[Union[str, Path]]
        :param transform: transformation function for data_bp augmentation, defaults to None
        :type transform: function, optional
        """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, **kwargs):

        self.root = root
        self.seq_id = kwargs.get('seq_id', None)
        self.ont = kwargs.get('ont', None)
        self.session = kwargs.get('session', None)
        self.prot_ids = kwargs.get('prot_ids', [])
        self.test_file = kwargs.get('test_file', None)
        self.pdb_pth = kwargs.get('pdb_path', self.root + "/alphafold/")

        self.raw_file_list = []
        self.processed_file_list = []

        if self.session == "selected":
            self.data = self.prot_ids
            for i in self.data:
                self.raw_file_list.append('{}'.format(i))
                self.processed_file_list.append('{}.pt'.format(i))
        else:
            if self.session == "train":
                self.data = pickle_load(self.root + "/{}/{}/{}".format(self.seq_id, self.ont, self.session))
                for i in self.data:
                    for j in self.data[i]:
                        self.raw_file_list.append('AF-{}-F1-model_v2.pdb.gz'.format(j))
                        self.processed_file_list.append('{}.pt'.format(j))
            elif self.session == "validation":
                self.data = list(pickle_load(self.root + "{}/{}".format(self.seq_id, self.session)))
                for i in self.data:
                    self.raw_file_list.append('AF-{}-F1-model_v2.pdb.gz'.format(i))
                    self.processed_file_list.append('{}.pt'.format(i))
            elif self.session == "test":
                self.data = self.get_test(self.test_file)
                for i in self.data:
                    self.raw_file_list.append('AF-{}-F1-model_v2.pdb.gz'.format(i))
                    self.processed_file_list.append('{}.pt'.format(i))

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self.pdb_pth

    @property
    def processed_dir(self) -> str:
        return self.root + "/processed/"

    @property
    def raw_file_names(self):
        return self.raw_file_list

    @property
    def processed_file_names(self):
        return self.processed_file_list

    def download(self):
        rem_files = set(self.raw_file_list) - set(find_files(self.raw_dir, suffix="pdb.gz", type="Name"))
        for file in rem_files:
            src = "/data_bp/pycharm/TransFunData/data_bp/alphafold/AF-{}-F1-model_v2.pdb.gz"
            des = self.root + "/raw/{}".format(file)
            if os.path.isfile(src.format(file)):
                pass
                # subprocess.call('cp {} {}'.format(src.format("pdb", file), des), shell=True)
            else:
                pass
                # download

    def process(self):
        rem_files = set(self.processed_file_list) - set(find_files(self.processed_dir, suffix="pt", type="Name"))
        print("{} unprocessed proteins out of {}".format(len(rem_files), len(self.processed_file_list)))
        chain_id = 'A'

        for file in rem_files:
            protein = file.split(".")[0]
            print("Processing protein {}".format(protein))

            raw_path = self.raw_dir + '{}.pdb.gz'.format(protein)

            labels = {
                'molecular_function': [],
                'biological_process': [],
                'cellular_component': []
            }

            emb = torch.load(self.root + "/esm/{}.pt".format(protein))
            embedding_features_per_residue = emb['representations'][33]
            embedding_features_per_sequence = emb['mean_representations'][33].view(1, -1)

            if raw_path:
                node_coords, sequence_features, sequence_letters = process_pdbpandas(raw_path, chain_id)
            # else: node_coords, sequence_features, sequence_letters = generate_Identity_Matrix(
            # embedding_features_per_residue.shape, self.fasta[protein])

            assert embedding_features_per_residue.shape[0] == node_coords.shape[0]
            assert embedding_features_per_residue.shape[1] == embedding_features_per_sequence.shape[1]

            node_size = node_coords.shape[0]
            names = torch.arange(0, node_size, dtype=torch.int8)

            data = HeteroData()
            data['atoms'].pos = node_coords

            data['atoms'].molecular_function = torch.IntTensor(labels['molecular_function'])
            data['atoms'].biological_process = torch.IntTensor(labels['biological_process'])
            data['atoms'].cellular_component = torch.IntTensor(labels['cellular_component'])

            data['atoms'].sequence_features = sequence_features
            data['atoms'].embedding_features_per_residue = embedding_features_per_residue
            data['atoms'].names = names
            data['atoms'].sequence_letters = sequence_letters
            data['atoms'].embedding_features_per_sequence = embedding_features_per_sequence
            data['atoms'].protein = protein

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                _transforms = []
                for i in self.pre_transform:
                    if i[0] == "KNN":
                        kwargs = {'mode': i[1], 'sequence_length': node_size}
                        knn = get_knn(**kwargs)
                        _transforms.append(myKNNGraph(i[1], k=knn, force_undirected=True, ))
                    if i[0] == "DIST":
                        _transforms.append(myRadiusGraph(i[1], r=i[2], loop=False))
                _transforms.append(myDistanceTransform(edge_types=self.pre_transform, norm=True))
                _transforms.append(AdjacencyFeatures(edge_types=self.pre_transform))

                pre_transform = T.Compose(_transforms)
                data = pre_transform(data)

            torch.save(data, osp.join(self.root + "/processed/", f'{protein}.pt'))

    def len(self):
        return len(self.data)

    def get(self, idx):
        if self.session == "train":
            rep = random.sample(self.data[idx], 1)[0]
            return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))
        elif self.session == "validation" or self.session == "selected" or self.session == "test":
            rep = self.data[idx]
            return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))

    def get_test(self, test_file):
        # all_test = set(self.all_test.keys())
        # all_test = set(self.all_test)
        # x = list(set(read_test_set("{}supplementary_data/cafa3/benchmark20171115/groundtruth/{}".format(self.root, test_file))))
        # onlystructs_filter = pickle_load("/home/fbqc9/PycharmProjects/TransFun/evaluation/available_structures")
        # onlystructs_filter = set([i[0].split(".")[0] for i in onlystructs_filter if i[1] == True])
        # x = [i for i in x if i in onlystructs_filter]

        data = pd.read_csv(Constants.ROOT + "timebased/test_data", sep="\t")
        data = data.loc[data['ONTOLOGY'] == self.ont]
        missing = set(pickle_load(Constants.ROOT + "timebased/missing_proteins"))
        data = list(set(data['ACC'].to_list()).difference(missing))

        # x = list(pickle_load(test_file)[self.ont])
        return data


def load_dataset(root=None, **kwargs):
    """
    Load files in file_list into corresponding dataset object. All files should be of type filetype.

    :param root: path to root
    :type file_list: list[Union[str, Path]]
    :param raw_path: path to raw path
    :type file_list: list[Union[str, Path]]

    :return: Pytorch Dataset containing data_bp
    :rtype: torch.utils.data_bp.Dataset
    """

    if root == None:
        raise ValueError('Root path is empty, specify root directory')

    # Group; name; operation/cutoff; Description
    pre_transform = [("KNN", "sqrt", "sqrt", "K nearest neighbour with sqrt for neighbours"),
                     ("KNN", "cbrt", "cbrt", "K nearest neighbour with sqrt for neighbours"),
                     ("DIST", "dist_3", 3, "Distance of 2angs"),
                     ("DIST", "dist_4", 4, "Distance of 2angs"),
                     ("DIST", "dist_6", 6, "Distance of 2angs"),
                     ("DIST", "dist_10", 10, "Distance of 2angs"),
                     ("DIST", "dist_12", 12, "Distance of 2angs")]
    # PDB URL has 1 attached to it
    dataset = PDBDataset(root, pre_transform=pre_transform, **kwargs)
    return dataset


# create raw and processed list.
def generate_dataset(_group="molecular_function"):
    # load sequences as dictionary
    if _group == "molecular_function":
        x = pickle_load('/data_bp/pycharm/TransFunData/data_bp/molecular_function/{}'.format(_group))
        raw = list(set([i for i in x.keys()]))
    elif _group == "cellular_component":
        pass
    elif _group == "biological_process":
        pass
    if raw:
        pickle_save(raw, '/data_bp/pycharm/TransFunData/data_bp/molecular_function/{}'
                    .format("molecular_function_raw_list"))
