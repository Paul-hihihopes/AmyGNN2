import torch
from torch_geometric.nn import GraphConv,global_sort_pool
from torch_geometric.nn.pool import SAGPooling
from torch.nn import Linear
import torch.nn.functional as F


class AMYGNN(torch.nn.Module):
    def __init__(self,input_channels, hidden_channels):
        super(AMYGNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels,eps = 1e-06,momentum = 0.01)
        self.sag1 = SAGPooling(hidden_channels, ratio = 1e-05)

        self.conv2 = GraphConv(hidden_channels,64)
        self.bn2 = torch.nn.BatchNorm1d(64,eps = 1e-06,momentum = 0.01)
        self.sag2 = SAGPooling(64, ratio = 1e-05)

        self.conv3 = GraphConv(64,32)
        self.bn3 = torch.nn.BatchNorm1d(32,eps = 1e-06,momentum = 0.01)
        self.sag3 = SAGPooling(32, ratio = 1e-05)

        self.lin = Linear(64,32)
        self.lin1 = Linear(32,16)
        self.lin2 = Linear(16,2)

    def forward(self, x, edge_index,edge_weight, batch):
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        x = self.bn1(x)
        # y = self.sag1(x,edge_index,edge_weight,batch = batch)
        # x = y[0]
        # edge_index = y[1]
        # edge_weight = y[2]
        # batch = y[3]


        x = self.conv2(x, edge_index,edge_weight)
        x = x.relu()
        x = self.bn2(x)
        # y = self.sag2(x,edge_index,edge_weight,batch = batch)
        # x = y[0]
        # edge_index = y[1]
        # edge_weight = y[2]
        # batch = y[3]


        x = self.conv3(x, edge_index,edge_weight)
        x = x.relu()
        x = self.bn3(x)
        y = self.sag3(x,edge_index,edge_weight,batch = batch)
        x = y[0]

        edge_index = y[1]
        edge_weight = y[2]
        batch = y[3]

        x = global_sort_pool(x,batch,2)
        x = F.dropout(x, p = 0.4, training=self.training)
        x = self.lin(x)
        x = x.relu()
        x = self.lin1(x)
        x = self.lin2(x)
        return F.log_softmax(x,dim= 1)