import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, BatchNorm, GCNConv
from torch.nn import Sigmoid, Linear
from models.egnn_clean import egnn_clean as eg
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
        self.num_layers = kwargs['layers']

        self.conv1 = net_utils.GCN_BatchNorm(input_features, hidden_channels_1, )
        self.conv2 = net_utils.GCN_BatchNorm(input_features_size + hidden_channels_1, hidden_channels_2, )
        self.conv3 = net_utils.GCN_BatchNorm(input_features_size + hidden_channels_2, hidden_channels_3, )

        self.fc1 = net_utils.FC(input_features_size, hidden_channels_1 + \
                                hidden_channels_2 + input_features_size * 2, \
                                relu=False, bnorm=True)

        self.fc2 = net_utils.FC(hidden_channels_1 + hidden_channels_2 + \
                                input_features_size * 2, fc2_out, relu=False, bnorm=True)

        self.fc3 = net_utils.FC(fc2_out * self.num_layers, num_classes, relu=False, bnorm=True)

        self.drp_out = nn.Dropout(p=0.4)

        #################################################
        self.bn4 = BatchNorm(hidden_channels_1 + hidden_channels_2 + input_features_size * 2)

        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, edge_atr, x_batch = data['atoms'].embedding_features_per_residue, \
                                                          data['atoms'].embedding_features_per_sequence, \
                                                          data[self.edge_type].edge_index, \
                                                          data[self.edge_type].edge_attr, \
                                                          data['atoms'].batch

        x = self.conv1((x_res, edge_index))
        # x = self.drp_out(x)
        x_1 = torch.cat((x, x_res), 1)

        x = self.conv2((x_1, edge_index))
        # x = self.drp_out(x)
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
        # x = self.drp_out(x)

        return x

    def forward(self, data):
        passes = []

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)
        x = self.fc3(x)
        # x = self.drp_out(x)

        x = self.sig(x)

        return x


class GCN2(torch.nn.Module):
    def __init__(self, input_features, **kwargs):
        super(GCN2, self).__init__()

        input_features_size = kwargs['input_features_size']
        hidden_channels_1 = kwargs['hidden1']
        hidden_channels_2 = kwargs['hidden2']
        hidden_channels_3 = kwargs['hidden3']
        num_classes = kwargs['num_classes']
        fc2_out = kwargs['fc2_out']

        self.edge_type = kwargs['edge_type']
        self.num_layers = kwargs['layers']

        self.conv1 = net_utils.GCN_BatchNorm(input_features, hidden_channels_1, )
        self.conv2 = net_utils.GCN_BatchNorm(hidden_channels_1, hidden_channels_2, )
        self.conv3 = net_utils.GCN_BatchNorm(hidden_channels_2, hidden_channels_3, )

        self.fc1 = net_utils.FC(input_features_size, hidden_channels_3, \
                                relu=False, bnorm=True)

        self.fc2 = net_utils.FC(hidden_channels_1 + hidden_channels_2 + \
                                input_features_size * 2, fc2_out, relu=False, bnorm=True)

        self.fc3 = net_utils.FC(fc2_out * self.num_layers, num_classes, relu=False, bnorm=True)

        #################################################
        self.bn4 = BatchNorm(hidden_channels_1 + hidden_channels_2 + input_features_size * 2)

        self.lin = Linear(hidden_channels_3, num_classes)
        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, edge_atr, x_batch = data['atoms'].embedding_features_per_residue, \
                                                          data['atoms'].embedding_features_per_sequence, \
                                                          data[self.edge_type].edge_index, \
                                                          data[self.edge_type].edge_attr, \
                                                          data['atoms'].batch

        x = self.conv1((x_res, edge_index))
        x = self.conv2((x, edge_index))
        x = self.conv3((x, edge_index))

        x = net_utils.get_pool(pool_type='mean')(x, x_batch)

        y = self.fc1(x_emb_seq)
        x += y

        x = self.lin(x)
        x = self.sig(x)

        return x

    def forward(self, data):
        passes = []

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)

        return x

class GCN3(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GCN3, self).__init__()

        input_features_size = kwargs['input_features_size']
        hidden_channels = kwargs['hidden']
        edge_features = kwargs['edge_features']
        num_classes = kwargs['num_classes']
        num_egnn_layers = kwargs['egnn_layers']

        self.edge_type = kwargs['edge_type']
        self.num_layers = kwargs['layers']
        self.device = kwargs['device']

        self.egnn_1 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=num_classes,
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_2 = eg.EGNN(in_node_nf=num_classes,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_3 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_4 = eg.EGNN(in_node_nf=int(num_classes / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 4),
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.fc1 = net_utils.FC(num_classes + int(num_classes / 2) * 2 + int(num_classes / 4), num_classes + 50,
                                relu=True, bnorm=True)
        # self.fc2 = net_utils.FC(800, 600, relu=True, bnorm=True)
        self.final = net_utils.FC(num_classes + 50, num_classes, relu=False, bnorm=False)

        self.bnrelu1 = net_utils.BNormRelu(num_classes)
        self.bnrelu2 = net_utils.BNormRelu(int(num_classes / 2))
        self.bnrelu3 = net_utils.BNormRelu(int(num_classes / 4))

        # self.final = nn.Linear(600, num_classes)
        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, edge_atr, x_batch, x_pos = data['atoms'].embedding_features_per_residue, \
                                                                 data['atoms'].embedding_features_per_sequence, \
                                                                 data[self.edge_type].edge_index, \
                                                                 data[self.edge_type].edge_attr, \
                                                                 data['atoms'].batch, \
                                                                 data['atoms'].pos
        ppi_shape = x_emb_seq.shape
        edge_index_2 = list(zip(*list(itertools.combinations(range(ppi_shape[0]), 2))))
        edge_index_2 = [torch.LongTensor(edge_index_2[0]).to(self.device), torch.LongTensor(edge_index_2[1]). \
            to(self.device)]

        output_res, pre_pos_res = self.egnn_1(h=x_res,
                                              x=x_pos.float(),
                                              edges=edge_index,
                                              edge_attr=None, )

        output_res_2, pre_pos_res_2 = self.egnn_2(h=output_res,
                                                  x=x_pos.float(),
                                                  edges=edge_index,
                                                  edge_attr=None, )

        output_res_3, pre_pos_res_3 = self.egnn_4(h=output_res_2,
                                                  x=net_utils.get_pool(pool_type='mean')(x_pos.float(), x_batch),
                                                  edges=edge_index_2,
                                                  edge_attr=None)

        output_seq, pre_pos_seq = self.egnn_3(h=x_emb_seq,
                                              x=net_utils.get_pool(pool_type='mean')(x_pos.float(), x_batch),
                                              edges=edge_index_2,
                                              edge_attr=None)

        output_res = net_utils.get_pool(pool_type='mean')(output_res, x_batch)
        output_res = self.bnrelu1(output_res)

        output_res_2 = net_utils.get_pool(pool_type='mean')(output_res_2, x_batch)
        output_res_2 = self.bnrelu2(output_res_2)

        output_res_3 = net_utils.get_pool(pool_type='mean')(output_res_3, x_batch)
        output_res_3 = self.bnrelu3(output_res_3)

        output_seq = self.bnrelu2(output_seq)

        output = torch.cat([output_res, output_seq, output_res_2, output_res_3], 1)


        return output

    def forward(self, data):
        passes = []

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)

        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.final(x)
        x = self.sig(x)

        return x


class GCN4(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GCN4, self).__init__()

        input_features_size = kwargs['input_features_size']
        hidden_channels = kwargs['hidden']
        edge_features = kwargs['edge_features']
        num_classes = kwargs['num_classes']
        num_egnn_layers = kwargs['egnn_layers']

        self.edge_type = kwargs['edge_type']
        self.num_layers = kwargs['layers']
        self.device = kwargs['device']

        self.egnn_1 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_2 = eg.EGNN(in_node_nf=int(num_classes / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 4),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_3 = eg.EGNN(in_node_nf=int(num_classes / 4),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 6),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=True,
                              tanh=True)

        ################################################################
        ################################################################
        ################################################################

        self.egnn_4 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_5 = eg.EGNN(in_node_nf=int(num_classes / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 4),
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        self.egnn_6 = eg.EGNN(in_node_nf=int(num_classes / 4),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 6),
                              in_edge_nf=0,
                              attention=True,
                              normalize=True,
                              tanh=True)

        ################################################################
        ################################################################
        ################################################################

        self.fc1 = net_utils.FC((int(num_classes / 2) + int(num_classes / 4) + int(num_classes / 6)) * 2,
                                800, relu=True, bnorm=True)

        self.final = net_utils.FC(800, num_classes, relu=True, bnorm=True)

        self.bnrelu1 = net_utils.BNormRelu(int(num_classes / 2))
        self.bnrelu2 = net_utils.BNormRelu(int(num_classes / 4))
        self.bnrelu3 = net_utils.BNormRelu(int(num_classes / 6))

        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, edge_atr, x_batch, x_pos = data['atoms'].embedding_features_per_residue, \
                                                                 data['atoms'].embedding_features_per_sequence, \
                                                                 data[self.edge_type].edge_index, \
                                                                 data[self.edge_type].edge_attr[:, 0:1], \
                                                                 data['atoms'].batch, \
                                                                 data['atoms'].pos

        ppi_shape = x_emb_seq.shape
        edge_index_2 = list(zip(*list(itertools.combinations(range(ppi_shape[0]), 2))))
        edge_index_2 = [torch.LongTensor(edge_index_2[0]).to(self.device),
                        torch.LongTensor(edge_index_2[1]).to(self.device)]

        output_res_1, pre_pos_res_1 = self.egnn_1(h=x_res,
                                                  x=x_pos,
                                                  edges=edge_index,
                                                  edge_attr=edge_atr)

        output_res_2, pre_pos_res_2 = self.egnn_2(h=output_res_1,
                                                  x=pre_pos_res_1,
                                                  edges=edge_index,
                                                  edge_attr=edge_atr)

        output_res_3, pre_pos_res_3 = self.egnn_3(h=output_res_2,
                                                  x=pre_pos_res_2,
                                                  edges=edge_index,
                                                  edge_attr=edge_atr)

        output_res_4, pre_pos_res_4 = self.egnn_4(h=x_emb_seq,
                                                  x=net_utils.get_pool(pool_type='mean')(x_pos, x_batch),
                                                  edges=edge_index_2,
                                                  edge_attr=None)

        output_res_5, pre_pos_res_5 = self.egnn_5(h=output_res_4,
                                                  x=pre_pos_res_4,
                                                  edges=edge_index_2,
                                                  edge_attr=None)

        output_res_6, pre_pos_res_6 = self.egnn_6(h=output_res_5,
                                                  x=pre_pos_res_5,
                                                  edges=edge_index_2,
                                                  edge_attr=None)

        output_res_1 = net_utils.get_pool(pool_type='mean')(output_res_1, x_batch)
        output_res_1 = self.bnrelu1(output_res_1)

        output_res_2 = net_utils.get_pool(pool_type='mean')(output_res_2, x_batch)
        output_res_2 = self.bnrelu2(output_res_2)

        output_res_3 = net_utils.get_pool(pool_type='mean')(output_res_3, x_batch)
        output_res_3 = self.bnrelu3(output_res_3)

        output_res_4 = self.bnrelu1(output_res_4)
        output_res_5 = self.bnrelu2(output_res_5)
        output_res_6 = self.bnrelu3(output_res_6)

        output = torch.cat([output_res_1, output_res_4, output_res_2, output_res_5, output_res_3, output_res_6], 1)

        return output

    def forward(self, data):
        passes = []

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)

        x = self.fc1(x)
        x = self.final(x)
        x = self.sig(x)

        return x
