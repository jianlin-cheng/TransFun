import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, BatchNorm, GCNConv
from torch.nn import Sigmoid, Linear

import net_utils


class GCN(torch.nn.Module):
    def __init__(self, input_features, **kwargs):
        super(GCN, self).__init__()

        input_features_size = kwargs['input_features_size']
        hidden_channels_1 = kwargs['hidden1']
        hidden_channels_2 = kwargs['hidden2']
        hidden_channels_3 = kwargs['hidden3']
        num_classes = kwargs['num_classes']
        fc2_out = kwargs['fc2_out']

        self.edge_type = kwargs['edge_type']

        self.conv1 = net_utils.GCN_BatchNorm(input_features, hidden_channels_1, )
        self.conv2 = net_utils.GCN_BatchNorm(input_features_size + hidden_channels_1, hidden_channels_2, )
        self.conv3 = net_utils.GCN_BatchNorm(input_features_size + hidden_channels_2, hidden_channels_3, )

        self.fc1 = net_utils.FC(input_features_size, hidden_channels_1 + \
                                hidden_channels_2 + input_features_size * 2, \
                                relu=False, bnorm=True)

        self.fc2 = net_utils.FC(hidden_channels_1 + hidden_channels_2 + \
                                input_features_size * 2, fc2_out, relu=False, bnorm=True)

        self.fc3 = net_utils.FC(fc2_out*2, num_classes, relu=False, bnorm=True)

        self.drp_out = nn.Dropout(p=0.4)

        #################################################
        self.bn4 = BatchNorm(hidden_channels_1 + hidden_channels_2 + input_features_size*2)

        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, edge_atr, x_batch = data['atoms'].embedding_features_per_residue, \
                                                          data['atoms'].embedding_features_per_sequence, \
                                                          data[self.edge_type].edge_index, \
                                                          data[self.edge_type].edge_attr, \
                                                          data['atoms'].batch

        # x_res, x_emb_seq, x_raw_seq, edge_index, x_batch = data.embedding_features_per_residue, \
        #                                                    data.embedding_features_per_sequence, \
        #                                                    data.sequence_features, \
        #                                                    data.edge_index, \
        #                                                    data.batch

        x = self.conv1((x_res, edge_index))
        x = self.drp_out(x)
        x_1 = torch.cat((x, x_res), 1)

        x = self.conv2((x_1, edge_index))
        x = self.drp_out(x)
        x_2 = torch.cat((x, x_res), 1)

        # x = self.conv3((x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = torch.cat((x, x_res), 1)

        x = torch.cat((x_1, x_2), 1)

        # 2. Readout layer
        # [batch_size, hidden_channels]
        x = net_utils.get_pool(pool_type='add')(x, x_batch)

        y = self.fc1(x_emb_seq)
        x += y

        # x = self.bn4(x)
        x = self.fc2(x)
        x = self.drp_out(x)

        return x

    def forward(self, data):
        x_1 = self.forward_once(data)
        x_2 = self.forward_once(data)

        x = torch.cat((x_1, x_2), 1)
        x = self.fc3(x)
        x = self.drp_out(x)

        x = self.sig(x)

        return x
