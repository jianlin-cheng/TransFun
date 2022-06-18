import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn import Linear, Softmax, Sigmoid, LSTM, Module, ReLU
from torch_geometric.nn import global_mean_pool
from torch.autograd import Variable


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
                 hidden_channels_3, num_classes):
        super(GCN, self).__init__()
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
        x, edge_index, edge_weight = data.embedding_features_per_residue, \
                                     data.edge_index, \
                                     data.edge_attr[:, 0:1]
        x = self.bn1(F.relu(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = self.bn2(F.relu(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight))

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

# class GCN(torch.nn.Module):
#     def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
#                  hidden_channels_3, num_classes):
#         # hidden_channels_1 = 1000
#         # hidden_channels_2 = 800
#         # hidden_channels_3 = 600
#         super(GCN, self).__init__()
#         # torch.manual_seed(12345)
#
#         self.batch = 10
#         self.layers = 5
#
#         self.lstm = LSTM(21, 600, self.layers, batch_first=True, bidirectional=True, dropout=0.3)
#
#         self.final = Linear(600*2, num_classes)
#
#         # self.softmax = Softmax(dim=0)
#         self.sigm = Sigmoid()
#
#     def forward_once(self, per_res, per_seq, seq, edge_index, batch):
#         # h0 = torch.randn(2 * self.layers, self.batch, 600).to('cpu')
#         # c0 = torch.randn(2 * self.layers, self.batch, 600).to('cpu')
#         # output, (hn, cn) = self.lstm(seq, (h0, c0))
#         output, (hn, cn) = self.lstm(seq)
#         hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
#         output = self.final(hn)
#         output = self.sigm(output)
#         return output
#
#     def forward(self, data):
#         per_res, per_seq, seq, edge_index, batch = data.embedding_features_per_residue, \
#                                                    data.embedding_features_per_sequence, \
#                                                    data.sequence_features, \
#                                                    data.edge_index, \
#                                                    data.batch
#         output = self.forward_once(per_res, per_seq, seq, edge_index, batch)
#         # lstm embedding
#         return output


# class MyLSTM(Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
#         super(MyLSTM, self).__init__()
#         self.num_classes = num_classes  # number of classes
#         self.num_layers = num_layers  # number of layers
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # hidden state
#         self.seq_length = seq_length  # sequence length
#
#         self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)  # lstm
#         self.fc_1 = Linear(hidden_size, 800)  # fully connected 1
#         self.fc = Linear(800, num_classes)  # fully connected last layer
#
#         self.relu = ReLU()
#         self.sigm = Sigmoid()
#
#     def forward(self, data):
#         per_res, per_seq, seq, edge_index, batch = data.embedding_features_per_residue, \
#                                                    data.embedding_features_per_sequence, \
#                                                    data.sequence_features, \
#                                                    data.edge_index, \
#                                                    data.batch
#
#         x = seq
#         h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
#         c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
#
#         # h_0 = torch.randn(2 * self.layers, self.batch, 600).to('cpu')
#         # c_0 = torch.randn(2 * self.layers, self.batch, 600).to('cpu')
#
#         # Propagate input through LSTM
#         output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
#         hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
#         out = self.relu(hn)
#         out = self.fc_1(out)  # first Dense
#         out = self.relu(out)  # relu
#         out = self.fc(out)  # Final Output
#         out = self.sigm(out)
#         return out


class myGCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels_1, hidden_channels_2,
                 hidden_channels_3, num_classes):
        super(myGCN, self).__init__()
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
        x, edge_index, edge_weight = data.embedding_features_per_residue, \
                                     data.edge_index, \
                                     data.edge_attr[:, 0:1]
        x = self.bn1(F.relu(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = self.bn2(F.relu(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight))

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



# x = myGCN()
# print(x)