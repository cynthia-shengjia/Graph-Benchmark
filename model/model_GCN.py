import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None, cat_node_feat_mf=False, data_name=None):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels, normalize=False))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))

                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))

                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args

        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x