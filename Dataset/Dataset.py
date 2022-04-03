import math
import os
import pickle
import subprocess
import torch
import os.path as osp
from torch_geometric.data import Dataset, download_url

from Dataset.utils import find_files, process_pdbpandas, get_knn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from Dataset.AdjacencyTransform import AdjacencyFeatures
from preprocessing.utils import pickle_load, pickle_save, get_sequence_from_pdb, fasta_to_dictionary
import pandas as pd
import random


class PDBDataset(Dataset):
    """
        Creates a dataset from a list of PDB files.
        :param file_list: path to LMDB file containing dataset
        :type file_list: list[Union[str, Path]]
        :param transform: transformation function for data augmentation, defaults to None
        :type transform: function, optional
        """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, **kwargs):

        self.root = root

        self.seq_id = kwargs['seq_id']
        self.ont = kwargs['ont']
        self.session = kwargs['session']

        self.cluster = dict()
        self.raw_file_list = None
        self.processed_file_list = None

        self.file_list = pd.read_pickle(self.root + "{}/{}/{}.pickle".format(self.seq_id, self.ont, self.session))

        self.annot = pd.read_csv(self.root + '{}/annot.tsv'.format(self.seq_id), delimiter='\t')

        self.annot = self.annot.where(pd.notnull(self.annot), None)

        self.annot = pd.Series(self.annot[self.ont].
                               values, index=self.annot['Protein']).to_dict()

        if self.ont == 'all':
            self.terms = list(
                set(pickle_load(self.root + '{}/term2name'.format(self.seq_id))['GO-terms-molecular_function']).union(
                set(pickle_load(self.root + '{}/term2name'.format(self.seq_id))['GO-terms-biological_process'])).union(
                set(pickle_load(self.root + '{}/term2name'.format(self.seq_id))['GO-terms-cellular_component'])))
        else:
            self.terms = pickle_load(self.root + '{}/term2name'.format(self.seq_id))['GO-terms-' + self.ont]

        self.fasta = fasta_to_dictionary(self.root + 'uniprot/filtered.fasta')

        self.index_table = pickle_load(self.root + '/{}/{}/train'.format(self.seq_id, self.ont))

        label_set = set()
        for i in self.annot:
            if self.annot[i] is not None:
                label_set.update(self.annot[i].split(','))

        assert len(self.terms) == len(label_set)

        pickle_save(list(label_set), self.root + '{}/{}/terms'.format(self.seq_id, self.ont))

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return "/data/pycharm/TransFunData/data/alphafold/"
        # return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return self.root + "/{}/{}/processed/".format(self.seq_id, self.ont)

    @property
    def raw_file_names(self):
        cluster = pd.read_csv(self.root + "/{}/mmseq/final_clusters.csv".format(self.seq_id),
                              names=['cluster'], header=None)

        for line_number, (index, row) in enumerate(cluster.iterrows()):
            line = row[0].split('\t')

            self.cluster[line[0]] = [i for i in line if self.annot[i] is not None]

        self.raw_file_list = ['AF-{}-F1-model_v2.pdb.gz'.format(clust) for rep in self.file_list for clust in
                              self.cluster[rep]]

        return list(set(self.raw_file_list))

    @property
    def processed_file_names(self):
        self.processed_file_list = [clust + ".pt" for rep in self.file_list for clust in
                                    self.cluster[rep]]

        return list(set(self.processed_file_list))

    def download(self):
        rem_files = set(self.raw_file_list) - set(find_files(self.raw_dir, suffix="pdb.gz", type="Name"))
        for file in rem_files:
            src = "/data/pycharm/TransFunData/data/alphafold/AF-{}-F1-model_v2.pdb.gz"
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
            raw_path = self.raw_dir + '/AF-{}-F1-model_v2.pdb.gz'.format(protein)

            # if protein in self.annot:
            #     print('here')
            #     if self.annot[protein] is None:
            #         label = [0] * len(self.terms)
            #         assert sum(label) == 0
            #     else:
            label = []
            for term in self.terms:
                if term in self.annot[protein]:
                    label.append(1)
                else:
                    label.append(0)

            assert sum(label) == len(self.annot[protein].split(','))

            label = torch.tensor(label, dtype=torch.float32).view(1, -1)

            emb = torch.load(self.root + "/esm1/{}.pt".format(protein))
            embedding_features_per_residue = emb['representations'][33]
            embedding_features_per_sequence = emb['mean_representations'][33].view(1, -1)

            # print(label.shape)
            # print(embedding_features_per_sequence.shape)
            # print(embedding_features_per_residue.shape)

            node_coords, sequence_features, sequence_letters = process_pdbpandas(raw_path, chain_id)

            assert self.fasta[protein] == sequence_letters

            node_size = node_coords.shape[0]
            names = torch.arange(0, node_size, dtype=torch.int8)

            data = Data(pos=node_coords,
                        y=label,
                        sequence_features=sequence_features,
                        embedding_features_per_residue=embedding_features_per_residue,
                        names=names,
                        sequence_letters=sequence_letters,
                        embedding_features_per_sequence=embedding_features_per_sequence,
                        protein=protein)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                _transforms = []
                if "KNN" in self.pre_transform:
                    kwargs = {'mode': 'sqrt', 'sequence_length': node_size}
                    knn = get_knn(**kwargs)
                    _transforms.append(T.KNNGraph(k=knn, force_undirected=True, ))
                if "Distance" in self.pre_transform:
                    _transforms.append(T.RadiusGraph(r=12, loop=False))
                if "Edge_weight" in self.pre_transform:
                    _transforms.append(T.Distance(norm=True))
                if "Adjacent_features":
                    _transforms.append(AdjacencyFeatures())

                pre_transform = T.Compose(_transforms[0:1])
                data = pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'{protein}.pt'))
            # pickle_save(index_table, '/data/pycharm/TransFunData/data/molecular_function/index_table')
            # idx += 1

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        rep = random.choice(self.cluster[self.index_table[idx]])
        return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))


def load_dataset(root=None, **kwargs):
    """
    Load files in file_list into corresponding dataset object. All files should be of type filetype.

    :param root: path to root
    :type file_list: list[Union[str, Path]]
    :param raw_path: path to raw path
    :type file_list: list[Union[str, Path]]

    :return: Pytorch Dataset containing data
    :rtype: torch.utils.data.Dataset
    """

    # PDB URL has 1 attached to it
    dataset = PDBDataset(root, pre_transform=("KNN", "Edge_weight"), **kwargs)
    return dataset


# create raw and processed list.
def generate_dataset(_group="molecular_function"):
    # load sequences as dictionary
    if _group == "molecular_function":
        x = pickle_load('/data/pycharm/TransFunData/data/molecular_function/{}'.format(_group))
        raw = list(set([i for i in x.keys()]))
    elif _group == "cellular_component":
        pass
    elif _group == "biological_process":
        pass
    if raw:
        pickle_save(raw, '/data/pycharm/TransFunData/data/molecular_function/{}'
                    .format("molecular_function_raw_list"))
