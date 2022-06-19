import torch

from torch_geometric.transforms import BaseTransform


class AdjacencyFeatures(BaseTransform):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __init__(self, edge_types, cat=True):
        self.cat = cat
        self.edge_types = edge_types

    def __call__(self, data):
        for edge_type in self.edge_types:
            adjacent_edges = []
            (row, col), pseudo = data['atoms', edge_type[1], 'atoms'].edge_index, \
                                 data['atoms', edge_type[1], 'atoms'].get('edge_attr', None)

            for i, j in zip(row, col):
                assert i != j
                if abs(i - j) == 1:
                    adjacent_edges.append(1)
                else:
                    adjacent_edges.append(0)

            adjacent_edges = torch.FloatTensor(adjacent_edges).view(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data['atoms', edge_type[1], 'atoms'].edge_attr = torch.cat([pseudo, adjacent_edges.type_as(pseudo)],
                                                                           dim=-1)
            else:
                data['atoms', edge_type[1], 'atoms'].edge_attr = adjacent_edges

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} '
