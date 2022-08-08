import torch
import torch.nn.functional as F
from torch import cat
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn import Linear, Softmax, Sigmoid, LSTM
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
                 hidden_channels_3, num_classes):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(input_features, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, hidden_channels_3)

        self.bn1 = BatchNorm(hidden_channels_1)
        self.bn2 = BatchNorm(hidden_channels_2)
        self.bn3 = BatchNorm(hidden_channels_3)
        self.softmax = Softmax(dim=0)
        self.sigm = Sigmoid()

        self.lin0 = Linear(hidden_channels_3, 183)
        self.lin1 = Linear(hidden_channels_3, 182)

        self.fc = torch.nn.Linear(1280, hidden_channels_3)

    def forward_1(self, data):
        # 1. Obtain node embeddings
        x, edge_index = data.embedding_features_per_residue, data.edge_index

        # features = data.sequence_features

        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]
        # print(data.embedding_features_per_sequence.shape)

        y = self.fc(data.embedding_features_per_sequence)
        # print(y.shape)
        x += y

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin0(x)
        x = self.sigm(x)

        return x


    def forward_2(self, data):
        # 1. Obtain node embeddings
        x, edge_index = data.embedding_features_per_residue, data.edge_index

        # features = data.sequence_features
        #
        #
        #
        #
        # exit()
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]
        # print(data.embedding_features_per_sequence.shape)
        y = self.fc(data.embedding_features_per_sequence)
        # print(y.shape)
        x += y

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        x = self.sigm(x)

        return x

    def forward(self, data):

        x_1 = self.forward_1(data)
        x_2 = self.forward_2(data)
        x_3 = self.forward_2(data)

        x = cat([x_1, x_2, x_3], dim=1)

        return x
