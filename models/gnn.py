import itertools

import torch
from torch.nn import Sigmoid
from models.egnn_clean import egnn_clean as eg
import net_utils


class GCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GCN, self).__init__()

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
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_2 = eg.EGNN(in_node_nf=num_classes,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_3 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_4 = eg.EGNN(in_node_nf=int(num_classes / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 4),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.fc1 = net_utils.FC(num_classes + int(num_classes / 2) * 2 + int(num_classes / 4),
                                num_classes + 50, relu=False, bnorm=True)
        self.final = net_utils.FC(num_classes + 50, num_classes, relu=False, bnorm=False)

        self.bnrelu1 = net_utils.BNormRelu(num_classes)
        self.bnrelu2 = net_utils.BNormRelu(int(num_classes / 2))
        self.bnrelu3 = net_utils.BNormRelu(int(num_classes / 4))
        self.sig = Sigmoid()

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, x_batch, x_pos = data['atoms'].embedding_features_per_residue, \
            data['atoms'].embedding_features_per_sequence, \
            data[self.edge_type].edge_index, \
            data['atoms'].batch, \
            data['atoms'].pos

        ppi_shape = x_emb_seq.shape[0]

        if ppi_shape > 1:
            edge_index_2 = list(zip(*list(itertools.combinations(range(ppi_shape), 2))))
            edge_index_2 = [torch.LongTensor(edge_index_2[0]).to(self.device),
                            torch.LongTensor(edge_index_2[1]).to(self.device)]
        else:
            edge_index_2 = tuple(range(ppi_shape))
            edge_index_2 = [torch.LongTensor(edge_index_2).to(self.device),
                            torch.LongTensor(edge_index_2).to(self.device)]

        output_res, pre_pos_res = self.egnn_1(h=x_res,
                                              x=x_pos.float(),
                                              edges=edge_index,
                                              edge_attr=None)

        output_res_2, pre_pos_res_2 = self.egnn_2(h=output_res,
                                                  x=pre_pos_res.float(),
                                                  edges=edge_index,
                                                  edge_attr=None)

        output_seq, pre_pos_seq = self.egnn_3(h=x_emb_seq,
                                              x=net_utils.get_pool(pool_type='mean')(x_pos.float(), x_batch),
                                              edges=edge_index_2,
                                              edge_attr=None)

        output_res_4, pre_pos_seq_4 = self.egnn_4(h=output_res_2,
                                                  x=pre_pos_res_2.float(),
                                                  edges=edge_index,
                                                  edge_attr=None)

        output_res = net_utils.get_pool(pool_type='mean')(output_res, x_batch)
        output_res = self.bnrelu1(output_res)

        output_res_2 = net_utils.get_pool(pool_type='mean')(output_res_2, x_batch)
        output_res_2 = self.bnrelu2(output_res_2)

        output_seq = self.bnrelu2(output_seq)

        output_res_4 = net_utils.get_pool(pool_type='mean')(output_res_4, x_batch)
        output_res_4 = self.bnrelu3(output_res_4)

        output = torch.cat([output_res, output_seq, output_res_2, output_res_4], 1)

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
