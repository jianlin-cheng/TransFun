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
        # hidden_channels_1 = 1000
        # hidden_channels_2 = 800
        # hidden_channels_3 = 600
        super(GCN, self).__init__()

        hid_1 = 1000
        hid_2 = 1000
        hid_3 = 1000
        _input_features = 1280

        #num_classes = 378
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(input_features, hid_1, )
        self.conv2 = GCNConv(_input_features + hid_1, hid_2, )
        self.conv3 = GCNConv(_input_features + hid_2, hid_3, )

        self.lin = Linear(hidden_channels_3, num_classes)
        self.bn1 = BatchNorm(hid_1)
        self.bn2 = BatchNorm(hid_2)
        self.bn3 = BatchNorm(hid_3)

        self.sigm = Sigmoid()

        self.bn4 = BatchNorm(hid_1 + hid_2 + 1280 + 1280)

        self.fc1 = torch.nn.Linear(_input_features, hid_1 + hid_2 + 1280 + 1280)

        self.fc2 = torch.nn.Linear(hid_1 + hid_2 + 1280 + 1280, 3000)
        self.bn5 = BatchNorm(3000)

        self.fc3 = torch.nn.Linear(6000, num_classes)
        self.bn6 = BatchNorm(num_classes)

    #     self.c1 = nn.Conv1d(1024, 1000, 2)
    #     self.p1 = nn.AvgPool1d(2)
    #     self.c2 = nn.Conv1d(1000, 800, 1)
    #     self.p2 = nn.AvgPool1d(2)
    #     self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)
    #     self.out = nn.Linear(800, 600)
    #
    # def OneD_Conv(self, seq):
    #     print(seq.shape)
    #     x = seq#.transpose(1, 2)
    #
    #     c = self.c1(x)
    #     p = self.p1(c)
    #     c = self.c2(p)
    #     p = self.p2(c)
    #
    #
    #     print(p.shape)
    #
    #     exit()
    #     x = flatten(x)
    #     x = self.fcc(x)
    #     print(x.shape)
    #     return x

    def forward_once(self, data):
        # 1. Obtain node embeddings

        # print(data.embedding_features_per_sequence.shape)
        # print(data.sequence_features.shape)
        # print(data.embedding_features_per_residue.shape)
        # print(data.pos.shape)

        x_res, x_emb_seq, x_raw_seq, edge_index, edge_weight = data.embedding_features_per_residue, \
                                                  data.embedding_features_per_sequence, \
                                                  data.sequence_features, \
                                                  data.edge_index, \
                                                data.edge_attr[:, 0:1]

        # x = torch.eye(x.shape[0], m=x.shape[1]).to('cuda')
        #
        xx = self.bn1(F.relu(self.conv1(x_res, edge_index)))
        xx = F.dropout(xx, p=0.5, training=self.training)
        xx = torch.cat((xx, x_res), 1)

        xxx = self.bn2(F.relu(self.conv2(xx, edge_index)))
        xxx = F.dropout(xxx, p=0.5, training=self.training)
        xxx = torch.cat((xxx, x_res), 1)

        # xxxx = self.bn3(F.relu(self.conv3(xxx, edge_index)))
        # xxxx = F.dropout(xxxx, p=0.2, training=self.training)

        xxxxx = torch.cat((xx, xxx), 1)

        #
        # # 2. Readout layer
        x = global_add_pool(xxxxx, data.batch)  # [batch_size, hidden_channels]

        y = self.fc1(data.embedding_features_per_sequence)

        x += y

        # # print(data.embedding_features_per_sequence.shape)
        # # y = self.fc1(data.embedding_features_per_sequence)
        # # print(y.shape)
        # # x += y
        #
        # # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.bn4(x)

        x = self.fc2(x)
        x = self.bn5(x)



        # x = self.lin(x)
        #  print(x.shape)

        # x = self.OneD_Conv(x_raw_seq)


        return x


    def forward(self, data):

        x_1 = self.forward_once(data)
        x_2 = self.forward_once(data)

        x_12= torch.cat((x_1, x_2), 1)

        x = self.fc3(x_12)
        x = self.bn6(x)

        x = self.sigm(x)

        return x


