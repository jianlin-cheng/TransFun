import torch_geometric
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected


class myKNNGraph(KNNGraph):
    r"""Creates a k-NN graph based on node positions :obj:`pos`.
    """
    def __init__(self, name: str, k=6, loop=False, force_undirected=False,
                 flow='source_to_target'):

        super().__init__(k, loop, force_undirected, flow)
        self.name = name

    def __call__(self, data):
        data['atoms', self.name, 'atoms'].edge_attr = None
        batch = data.batch if 'batch' in data else None
        edge_index = torch_geometric.nn.knn_graph(data['atoms'].pos,
                                                  self.k, batch,
                                                  loop=self.loop,
                                                  flow=self.flow)

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data['atoms', self.name, 'atoms'].edge_index = edge_index
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'
