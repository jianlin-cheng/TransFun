from typing import Optional

import torch_geometric
from torch_geometric.transforms import RadiusGraph


class myRadiusGraph(RadiusGraph):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance.
    """
    def __init__(
        self,
        name: str,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
    ):
        super().__init__(r, loop, max_num_neighbors, flow)
        self.name = name

    def __call__(self, data):
        data['atoms', self.name, 'atoms'].edge_attr = None
        batch = data.batch if 'batch' in data else None
        data['atoms', self.name, 'atoms'].edge_index = torch_geometric.nn.radius_graph(
                                                        data['atoms'].pos,
                                                        self.r, batch, self.loop,
                                                        self.max_num_neighbors,
                                                        self.flow)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'
