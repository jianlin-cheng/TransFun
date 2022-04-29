import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn import Linear, Softmax, Sigmoid
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
                 hidden_channels_3, num_classes):
        super(GCN, self).__init__()
        print(input_features)
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(input_features, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, hidden_channels_3)
        self.lin = Linear(hidden_channels_3, num_classes)
        self.bn1 = BatchNorm(hidden_channels_1)
        self.bn2 = BatchNorm(hidden_channels_2)
        self.bn3 = BatchNorm(hidden_channels_3)
        self.softmax = Softmax(dim=0)
        self.sigm = Sigmoid()

        self.fc1 = torch.nn.Linear(1280, hidden_channels_3)

    def forward(self, data):
        # 1. Obtain node embeddings
        x, edge_index = data.embedding_features_per_residue, data.edge_index
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))

        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]
        # print(data.embedding_features_per_sequence.shape)
        y = self.fc1(data.embedding_features_per_sequence)
        # print(y.shape)
        x += y

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        x = self.sigm(x)

        return x
