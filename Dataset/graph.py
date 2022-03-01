import itertools
from typing import Tuple, List
import Bio
from torch_geometric.transforms import KNNGraph
import numpy as np
import pandas as pd
import torch
from itertools import groupby
import torch.nn.functional as F


residues = {
            "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10,
            "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19, "X": 20,
}


def one_of_k_encoding_unk(feat, allowable_set):
    """Convert input to 1-hot encoding given a set of (or sets of) allowable values.
     Additionally, map inputs not in the allowable set to the last element."""
    if feat not in allowable_set:
        feat = allowable_set[-1]
    return list(map(lambda s: feat == s, allowable_set))


def distance_metric(node1, node2, metric='Euclidean_norm'):
    if metric == 'Euclidean':
        return np.linalg.norm(node1 - node2)
    elif metric == 'Euclidean_norm':
        return 1.0 / (np.linalg.norm(node1 - node2) + 1e-5)


def prot_df_to_graph_feats_knn(df: pd.DataFrame, knn: int, sequence: Bio.Seq.Seq):#,  allowable_feats: List[List]):
    r"""Convert protein in DataFrame representation to a graph compatible with DGL, where each node is an atom.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param allowable_feats: List of lists containing all possible values of node type, to be converted into 1-hot node features.
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`one_of_k_encoding_unk`).
    :param knn: Maximum number of nearest neighbors (i.e. edges) to allow for a given node.
    :type knn: int

    :return: tuple containing
        - knn_graph (dgl.DGLGraph): edges_per_node-nearest neighbor graph for the structure DataFrame given.

        - node_coords (torch.FloatTensor): Cartesian coordinates of each node.

        - atoms_types (torch.FloatTensor): Atom type for each node, one-hot encoded by values in ``allowable_feats``.

        - normalized_chain_ids (torch.FloatTensor): Normalized chain ID for each node.
    :rtype: Tuple
    """
    # Aggregate one-hot encodings of each atom's type to serve as the primary source of node features
    # atom_type_feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[0]) for feat in df['atom_name']]
    # atoms_types = torch.FloatTensor(atom_type_feat_vecs)
    # assert not torch.isnan(atoms_types).any(), 'Atom types must be valid float values, not NaN'

    sequence_features = torch.tensor([residues[residue] for residue in sequence])
    sequence_features = F.one_hot(sequence_features, num_classes=len(residues)).to(dtype=torch.float)

    # Organize atom coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    # Define edges - KNN argument determines whether an atom-atom edge gets created in the resulting graph
    knn_graph = KNNGraph()
    knn_graph = dgl.knn_graph(node_coords, knn)

    # Remove self-loops in graph
    graph = dgl.remove_self_loop(knn_graph)  # By removing self-loops w/ k=11, we are effectively left w/ edges for k=10

    graph.ndata['sequence_features'] = sequence_features

    graph.edata['distance'] = torch.FloatTensor([distance_metric(i, j) for i, j in zip(graph.edges()[0], graph.edges()[1])])

    distance = []
    adjacent_edges = []
    for i, j in zip(graph.edges()[0], graph.edges()[1]):
        distance.append(distance_metric(i, j))
        assert i != j
        if abs(i - j) == 1:
            adjacent_edges.append(1)
        else:
            adjacent_edges.append(0)

    graph.edata['adjacent_edges'] = torch.FloatTensor(adjacent_edges)

    return graph #, node_coords, atoms_types


def prot_df_to_graph_feats_distance(df: pd.DataFrame, threshold: int, sequence: Bio.Seq.Seq):
    r"""Convert protein in DataFrame representation to a graph compatible with DGL, where each node is an atom.

        :param df: Protein structure in dataframe format.
        :type df: pandas.DataFrame
        :param allowable_feats: List of lists containing all possible values of node type, to be converted into 1-hot node features.
            Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`one_of_k_encoding_unk`).
        :param threshold: maximum distance to consider
        :type knn: int

        :return: tuple containing
            -
        :rtype: Tuple
        """

    threshold = threshold
    sequence_features = torch.tensor([residues[residue] for residue in sequence])
    sequence_features = F.one_hot(sequence_features, num_classes=len(residues)).to(dtype=torch.float)

    df['node'] = np.arange(df.shape[0])
    node_coords = [tuple(x) for x in df[['node', 'x_coord', 'y_coord', 'z_coord']].to_numpy()]

    # pairwise = list(itertools.combinations(node_coords, 2))
    # Intentionally using product so that the distance threshold does not filter out nodes that are far away.
    # Then  I will remove nodes to remove the self edges
    pairwise = list(itertools.product(node_coords, node_coords))

    # Compute the distance for each node.
    src_ids = []
    dst_ids = []
    distance = []
    adjacent_edges = []
    for i, j in pairwise:
        dist = distance_metric(np.array(i[1:]), np.array(j[1:]), metric='Euclidean')
        dist1 = distance_metric(np.array(i[1:]), np.array(j[1:]))
        if dist < threshold:
            src_ids.append(i[0])
            dst_ids.append(j[0])
            # filter out self distance
            if i[0] != j[0]:
                distance.append(dist)
                if abs(i[0] - j[0]) == 1:
                    adjacent_edges.append(1)
                else:
                    adjacent_edges.append(0)

    graph = dgl.graph((src_ids, dst_ids))

    # Remove self-loops in graph
    graph = dgl.remove_self_loop(graph)  # By removing self-loops w/ k=11, we are effectively left w/ edges for k=10

    graph.ndata['sequence_features'] = sequence_features
    graph.edata['distance'] = torch.FloatTensor(distance)
    graph.edata['adjacent_edges'] = torch.FloatTensor(adjacent_edges)

    return graph # , node_coords, atoms_types