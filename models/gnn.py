import torch
import torch.nn.functional as F
from torch import nn, cat, squeeze, flatten
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, global_add_pool
from torch.nn import Linear, Softmax, Sigmoid, ELU, LeakyReLU
from torch_geometric.nn import global_mean_pool


# class GCN(torch.nn.Module):
#     def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
#                  hidden_channels_3, num_classes):
#         # hidden_channels_1 = 1000
#         # hidden_channels_2 = 800
#         # hidden_channels_3 = 600
#         super(GCN, self).__init__()
#         # torch.manual_seed(12345)
#
#         # num_classes = 378
#
#         self.conv1 = GCNConv(input_features, 1000)
#         self.bn1 = BatchNorm(1000)
#         self.lin1 = Linear(1280, 1000)
#
#         self.conv2 = GCNConv(1000, 800)
#         self.bn2 = BatchNorm(800)
#         self.lin2 = Linear(1000, 800)
#
#         self.conv3 = GCNConv(800, 600)
#         self.bn3 = BatchNorm(600)
#         self.lin3 = Linear(800, 600)
#
#         self.final = Linear(800, num_classes)
#         self.sigm = Sigmoid()
#
#         # self.m = nn.LeakyReLU(0.1)
#
#     def forward_once(self, per_res, per_seq, seq, edge_index, batch):
#
#         output = F.relu(self.bn1(self.conv1(per_res, edge_index)))
#         # output = output + self.lin1(per_res)
#         output = F.relu(self.bn1(output))
#
#         output = F.relu(self.bn2(self.conv2(output, edge_index)))
#         # output = output + self.lin2(self.lin1(per_res))
#         output = F.relu(self.bn2(output))
#
#         # output = F.relu(self.bn3(self.conv3(output, edge_index)))
#         # # output = output + self.lin3(self.lin2(self.lin1(per_res)))
#         # output = F.relu(self.bn3(output))
#
#         output = global_mean_pool(output, batch)
#
#         output = output + self.lin2(self.lin1(per_seq))
#         #output = F.relu(self.bn3(output))
#
#         output = self.final(output)
#
#         return output
#
#
#     def forward(self, data):
#         # 1. Obtain node embeddings
#
#         # print(data.embedding_features_per_sequence.shape)
#         # print(data.sequence_features.shape)
#         # print(data.embedding_features_per_residue.shape)
#         # print(data.pos.shape)
#
#         per_res, per_seq, seq, edge_index, batch = data.embedding_features_per_residue, \
#                                             data.embedding_features_per_sequence, \
#                                             data.sequence_features, \
#                                             data.edge_index, \
#                                             data.batch
#
#         out_1 = self.forward_once(per_res, per_seq, seq, edge_index, batch)
#         # out_2 = self.forward_once(per_res, per_seq, seq, edge_index, batch)
#
#         # output = self.final(out_1 + out_2)
#
#         output = self.sigm(out_1)
#         return output


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
                 hidden_channels_3, num_classes):
        super(GCN, self).__init__()

        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_features, hidden_channels_1, )
        self.conv2 = GCNConv(input_features + hidden_channels_1, hidden_channels_2, )
        self.conv3 = GCNConv(input_features + hidden_channels_2, hidden_channels_3, )

        self.lin = Linear(hidden_channels_3, num_classes)
        self.bn1 = BatchNorm(hidden_channels_1)
        self.bn2 = BatchNorm(hidden_channels_2)
        self.bn3 = BatchNorm(hidden_channels_3)

        self.sigm = Sigmoid()

        self.bn4 = BatchNorm(hidden_channels_1 + hidden_channels_2 + 1280 + 1280)

        self.fc1 = torch.nn.Linear(input_features, hidden_channels_1 + hidden_channels_2 + 1280 + 1280)

        self.fc2 = torch.nn.Linear(hidden_channels_1 + hidden_channels_2 + 1280 + 1280, 3000)
        self.bn5 = BatchNorm(3000)

        self.fc3 = torch.nn.Linear(6000, num_classes)
        self.bn6 = BatchNorm(num_classes)

    def forward_once(self, data):

        x_res, x_emb_seq, x_raw_seq, edge_index, edge_weight = data.embedding_features_per_residue, \
                                                  data.embedding_features_per_sequence, \
                                                  data.sequence_features, \
                                                  data.edge_index, \
                                                  data.edge_attr[:, 0:1]

        xx = self.bn1(F.relu(self.conv1(x_res, edge_index)))
        xx = F.dropout(xx, p=0.5, training=self.training)
        xx = torch.cat((xx, x_res), 1)

        xxx = self.bn2(F.relu(self.conv2(xx, edge_index)))
        xxx = F.dropout(xxx, p=0.5, training=self.training)
        xxx = torch.cat((xxx, x_res), 1)

        # xxxx = self.bn3(F.relu(self.conv3(xxx, edge_index)))
        # xxxx = F.dropout(xxxx, p=0.2, training=self.training)

        xxxxx = torch.cat((xx, xxx), 1)

        # 2. Readout layer
        x = global_add_pool(xxxxx, data.batch)

        y = self.fc1(data.embedding_features_per_sequence)
        x += y

        # # print(data.embedding_features_per_sequence.shape)
        # # y = self.fc1(data.embedding_features_per_sequence)
        # # print(y.shape)
        # # x += y

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.bn4(x)

        x = self.fc2(x)
        x = self.bn5(x)

        # x = self.lin(x)
        #  print(x.shape)
        return x

    def forward(self, data):

        x_1 = self.forward_once(data)
        x_2 = self.forward_once(data)
        x = torch.cat((x_1, x_2), 1)
        x = self.fc3(x)
        x = self.bn6(x)
        x = self.sigm(x)

        return x


