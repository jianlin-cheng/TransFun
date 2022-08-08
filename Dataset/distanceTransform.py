import torch

from torch_geometric.transforms import Distance


class myDistanceTransform(Distance):
    r"""

    """

    def __init__(self, edge_types, norm=True, max_value=None, cat=True):
        super().__init__(norm, max_value, cat)
        self.edge_types = edge_types

    def __call__(self, data):
        for i in self.edge_types:
            (row, col), pos, pseudo = data['atoms', i[1], 'atoms'].edge_index, \
                                      data['atoms'].pos, \
                                      data['atoms', i[1], 'atoms'].get('edge_attr', None)

            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

            if self.norm and dist.numel() > 0:
                dist = dist / (dist.max() if self.max is None else self.max)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data['atoms', i[1], 'atoms'].edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data['atoms', i[1], 'atoms'].edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
