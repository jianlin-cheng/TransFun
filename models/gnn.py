import dgl
# from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch import GlobalAttentionPooling
from torch import nn
from dgl.nn import GraphConv
import torch.nn.functional as F


class GCN(nn.Module):
    """A two layer Graph Conv net with Global Attention Pooling over the
    nodes.
    Args:
        in_feats: int, dim of input node features
        h_feats: int, dim of hidden layers
        num_classes: int, number of output units
    """

    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        # the gate layer that maps node feature to outputs
        self.gate_nn = nn.Linear(h_feats, num_classes)
        self.gap = GlobalAttentionPooling(self.gate_nn)
        # the output layer making predictions
        self.output = nn.Linear(h_feats, num_classes)

    def _conv_forward(self, g):
        """forward pass through the GraphConv layers"""
        in_feat = g.ndata["h"]
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        return h

    def forward(self, g):
        h = self._conv_forward(g)
        h = self.gap(g, h)
        return self.output(h)

    def attention_scores(self, g):
        """Calculate attention scores"""
        h = self._conv_forward(g)
        with g.local_scope():
            gate = self.gap.gate_nn(h)
            g.ndata["gate"] = gate
            gate = dgl.softmax_nodes(g, "gate")
            g.ndata.pop("gate")
            return gate
