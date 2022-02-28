import math

import torch
from biopandas.pdb import PandasPdb
from typing import Tuple
import dgl
import numpy as np
from graph import prot_df_to_dgl_graph_feats_knn, prot_df_to_dgl_graph_feats_distance

PROT_ATOM_NAMES = ['CA', 'N', 'C', 'O', 'CB', 'OG', 'CG', 'CD1', 'CD2', 'CE1', 'CE2',
                   'CZ', 'OD1', 'ND2', 'CG1', 'CG2', 'CD', 'CE', 'NZ', 'OD2',
                   'OE1', 'NE2', 'OE2', 'OH', 'NE', 'NH1', 'NH2', 'OG1', 'SD',
                   'ND1', 'SG', 'NE1', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT', 'UNX']  # 'UNX' represents unknown atom type
ALLOWABLE_FEATS = [PROT_ATOM_NAMES[:1], ]


class GraphTransform(object):
    def __init__(self):
        super(GraphTransform, self).__init__()

    def __call__(self, item):
        pandas_pdb = item['pdb_to_pandas']
        chain_id = item['chain_id']
        edge_metric = item['metric']
        data = convert_dfs_to_dgl_graph(pandas_pdb, chain_id, edge_metric)

        return {"graph": data, "label": item["label"], "Protein": item['protein_id']}


def get_knn(**kwargs):
    mode = kwargs["mode"]
    seq_length = kwargs["sequence_length"]
    if mode == "sqrt":
        x = int(math.sqrt(seq_length))
        if x % 2 == 0:
            return x + 1
        return x


def convert_dfs_to_dgl_graph(input_pdb: PandasPdb, chain_id, metric='knn') -> \
        Tuple[dgl.DGLGraph, np.ndarray]:
    r""" Transform a given set of predicted and true atom DataFrames into a corresponding DGL graph.

    Parameters
    ----------
    pdb: PandasPdb
    knn: int

    Returns
    -------
    :class:`typing.Tuple[:class:`dgl.DGLGraph`, :class:`np.ndarray`]`
        Index 1. Graph structure, feature tensors for each node and edge.
        - ``ndata['f']``: feature tensors of the nodes
        Index 2. Sequence Encoding.
    """

    pdb_df = input_pdb.df['ATOM']
    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & (pdb_df['chain_id'] == chain_id)]

    pdb_df = pdb_df.drop_duplicates()
    residue_name = pdb_df['residue_name'].to_list()

    one_letter = input_pdb.amino3to1()
    one_letter = one_letter[(one_letter['chain_id'] == chain_id)]
    sequence = one_letter['residue_name'].to_list()

    assert(len(residue_name) == len(sequence))

    if metric == "knn":
        kwargs = {'mode': 'sqrt', 'sequence_length': len(sequence), 'metric': metric}
        knn = get_knn(**kwargs)
        # Construct KNN graph
        graph = prot_df_to_dgl_graph_feats_knn(pdb_df, knn, sequence)

    elif metric == "distance":
        threshold = 0.6
        graph = prot_df_to_dgl_graph_feats_distance(pdb_df, threshold, sequence)



    # graph.ndata['atom_type'] = atoms_types  # [num_nodes, num_node_feats=38]
    # Include normalized scalar feature indicating each atom's chain ID
    # graph.ndata['chain_id'] = normalized_chain_ids  # [num_nodes, num_node_feats=1]
    # Cartesian coordinates for each atom
    # graph.ndata['node_cordinates'] = node_coords  # [num_nodes, 3]

    return graph


# def voxel_transform(item, grid_config, rot_mat=None, center_fn=vox.get_center, random_seed=None, structure_keys=['atoms']):
#     """Transform for converting dataframes to voxelized grids compatible with 3D CNN, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
#     Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` argument.
#
#     :param item: Dataset item to transform
#     :type item: dict
#     :param grid_config: Config parameters for grid. Should contain the following keys:
#          `element_mapping`, dictionary mapping from element to 1-hot index;
#          `radius`, radius of grid to generate in Angstroms (half of side length);
#          `resolution`, voxel size in Angstroms;
#          `num_directions`, number of directions for data augmentation (required if ``rot_mat``=None);
#          `num_rolls`, number of rolls, or rotations, for data augmentation (required if ``rot_mat``=None);
#     :type grid_config: :class:`dotdict <atom3d.util.voxelize.dotdict>`
#     :param rot_mat: Rotation matrix (3x3) to apply to structure coordinates. If None (default), apply randomly sampled rotation according to parameters specified by ``grid_config.num_directions`` and ``grid_config.num_rolls``
#     :type rot_mat: np.array
#     :param center_fn: Arbitrary function for calculating the center of the voxelized grid (x,y,z coordinates) from a structure dataframe, defaults to vox.get_center
#     :type center_fn: f(df -> array), optional
#     :param random_seed: random seed for grid rotation, defaults to None
#     :type random_seed: int, optional
#     :return: Transformed Dataset item
#     :rtype: dict
#     """
#
#     for key in structure_keys:
#         df = item[key]
#         center = center_fn(df)
#
#         if rot_mat is None:
#             rot_mat = vox.gen_rot_matrix(grid_config, random_seed=random_seed)
#         grid = vox.get_grid(
#             df, center, config=grid_config, rot_mat=rot_mat)
#         item[key] = grid
#     return item
