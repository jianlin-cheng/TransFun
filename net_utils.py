import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool, global_mean_pool, global_max_pool


class GCN(nn.Module):
    def __init__(self, input_features, out_channels, relu=True):
        super(GCN, self).__init__()
        self.conv = GCNConv(input_features, out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        edge_index = x[1]
        x = self.conv(x[0], edge_index)
        if self.relu is not None:
            x = self.relu(x)
        return (x, edge_index)


class GCN_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(GCN_BatchNorm, self).__init__()

        self.conv = GCNConv(in_channels, out_channels, bias=False)
        self.bn = BatchNorm(out_channels, momentum=0.1)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        edge_index = x[1]
        x = self.conv(x[0], edge_index)
        if self.relu is not None:
            x = self.relu(x)
        x = self.bn(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True, bnorm=True):
        super(FC, self).__init__()
        _bias = False if bnorm else True
        self.fc = nn.Linear(in_features, out_features, bias=_bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        #self.bn = BatchNorm(out_features, momentum=0.1) if bnorm else None
        self.bn = nn.BatchNorm1d(out_features, momentum=0.1) if bnorm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BNormRelu(nn.Module):
    def __init__(self, in_features, relu=True, bnorm=True):
        super(BNormRelu, self).__init__()
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = BatchNorm(in_features, momentum=0.1) if bnorm else None
        # self.bn = nn.BatchNorm1d(out_features, momentum=0.1) if bnorm else None

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def get_pool(pool_type='max'):
    if pool_type == 'mean':
        return global_mean_pool
    elif pool_type == 'add':
        return global_add_pool
    elif pool_type == 'max':
        return global_max_pool
