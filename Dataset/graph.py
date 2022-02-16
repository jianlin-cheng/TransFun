from typing import Tuple, List
import Bio
import dgl
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


def prot_df_to_dgl_graph_feats(df: pd.DataFrame, allowable_feats: List[List], knn: int, sequence: Bio.Seq.Seq):
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
    atom_type_feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[0]) for feat in df['atom_name']]
    atoms_types = torch.FloatTensor(atom_type_feat_vecs)
    assert not torch.isnan(atoms_types).any(), 'Atom types must be valid float values, not NaN'

    # Gather chain IDs to serve as an additional node feature
    chain_ids = torch.FloatTensor([c for c, (k, g) in enumerate(groupby(df['chain_id'].values.tolist()), 0) for _ in g])
    max_chain_id = max(max(chain_ids.tolist()), 1.0)  # With max(), account for monomers having only a single chain
    normalized_chain_ids = (chain_ids / max_chain_id).reshape(-1, 1)
    assert not torch.isnan(normalized_chain_ids).any(), 'Chain IDs must be valid float values, not NaN'

    sequence_features = torch.tensor([residues[residue] for residue in sequence])
    sequence_features = F.one_hot(sequence_features, num_classes=len(residues)).to(dtype=torch.float)

    # Organize atom coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    # Define edges - KNN argument determines whether an atom-atom edge gets created in the resulting graph
    knn_graph = dgl.knn_graph(node_coords, knn)


    return knn_graph, node_coords, atoms_types, normalized_chain_ids# , sequence_features