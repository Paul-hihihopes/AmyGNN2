import torch
from torch_geometric.nn import GraphConv,global_sort_pool
from torch_geometric.nn.pool import SAGPooling
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F


class AMYGNN(torch.nn.Module):
    def __init__(self,input_channels, hidden_channels):
        super(AMYGNN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.rcl = ReconstructionLayer1(0,2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels,eps = 1e-06,momentum = 0.01)

        self.sag1 = SAGPooling(hidden_channels, ratio = 1e-05)

        self.conv2 = GraphConv(hidden_channels,64)
        self.rcl2 = ReconstructionLayer1(0, 1)
        self.bn2 = torch.nn.BatchNorm1d(64,eps = 1e-06,momentum = 0.01)
        self.sag2 = SAGPooling(64, ratio = 1e-05)

        self.conv3 = GraphConv(64,32)
        self.rcl3 = ReconstructionLayer1(0, 0)
        self.bn3 = torch.nn.BatchNorm1d(32,eps = 1e-06,momentum = 0.01)
        self.sag3 = SAGPooling(32, ratio = 1e-05)

        self.lin = Linear(64,32)
        self.lin1 = Linear(32,16)
        self.lin2 = Linear(16,2)

    def forward(self, x, edge_index,edge_weight, batch):
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        edge_index,edge_weight = self.rcl(x, edge_index,edge_weight,batch)
        x = self.bn1(x)
        # y = self.sag1(x,edge_index,edge_weight,batch = batch)
        # x = y[0]
        # edge_index = y[1]
        # edge_weight = y[2]
        # batch = y[3]

        x = self.conv2(x, edge_index,edge_weight)
        edge_index, edge_weight = self.rcl2(x, edge_index, edge_weight, batch)
        x = x.relu()
        x = self.bn2(x)
        # y = self.sag2(x,edge_index,edge_weight,batch = batch)
        # x = y[0]
        # edge_index = y[1]
        # edge_weight = y[2]
        # batch = y[3]

        x = self.conv3(x, edge_index,edge_weight)
        edge_index, edge_weight = self.rcl3(x, edge_index, edge_weight, batch)
        x = x.relu()
        x = self.bn3(x)
        y = self.sag3(x,edge_index,edge_weight,batch = batch)
        x = y[0]

        # edge_index = y[1]
        # edge_weight = y[2]
        batch = y[3]

        x = global_sort_pool(x,batch,2)
        x = F.dropout(x, p = 0.4, training=self.training)
        x = self.lin(x)
        x = x.relu()
        x = self.lin1(x)
        x = self.lin2(x)
        return F.log_softmax(x,dim= 1)

# class ReconstructionLayer(nn.Module):
#     def __init__(self, shortest_length, generate_mode):
#         super(ReconstructionLayer, self).__init__()
#         # 定义权重和偏置
#         self.sl = shortest_length
#         self.gm = generate_mode
#
#     def forward(self, x, edge_index, edge_weight,batch):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         spectral_dim = x[...,-3:]
#         # print(x.shape)
#         # print(batch)
#         # print(spectral_dim.shape)
#         # print(edge_index.shape)
#         # print(edge_weight.shape)
#
#         # 初始化边索引和边权重的列表
#         new_edge_index = []
#         new_edge_weight = []
#
#         # 获取唯一的图索引
#         unique_graphs = torch.unique(batch)
#
#         # 遍历每个图
#         for graph in unique_graphs:
#             # 获取当前图的节点索引
#             node_indices = (batch == graph).nonzero(as_tuple=True)[0]
#
#             # 提取当前图的节点特征
#             current_spectral_dim = spectral_dim[node_indices]
#
#             # 计算节点之间的距离
#             distance_matrix = torch.norm(current_spectral_dim[:, None] - current_spectral_dim[None, :], dim=-1)
#
#             # 找到距离小于阈值的节点对
#             edge_indices = torch.nonzero(distance_matrix < self.sl, as_tuple=False)
#
#             # 确保 edge_indices 是有效的
#             if edge_indices.size(0) > 0:
#                 # 将当前图的节点索引映射到全局索引
#                 edge_indices = node_indices[edge_indices]
#
#                 # 添加到新的边索引
#                 new_edge_index.append(edge_indices.T)
#
#                 # 添加边权重（所有边的权重为 1）
#                 edge_weights_for_graph = torch.ones(edge_indices.size(0), dtype=x.dtype)
#
#                 # 添加边权重
#                 new_edge_weight.append(edge_weights_for_graph)
#
#         # 拼接所有图的边索引和边权重
#         new_edge_index = torch.cat(new_edge_index, dim=1)
#         new_edge_weight = torch.cat(new_edge_weight)
#
#         return new_edge_index.to(device), new_edge_weight.to(device)

class ReconstructionLayer1(nn.Module):
    def __init__(self, generate_mode, sl_init):
        super(ReconstructionLayer1, self).__init__()
        # 定义权重和偏置
        self.sl = nn.Parameter(torch.randn(1))
        nn.init.normal_(self.sl, mean=sl_init, std=0.1)
        self.gm = generate_mode

    def forward(self, x, edge_index, edge_weight, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        spectral_dim = x[...,-3:]
        # print(x.shape)
        # print(batch)
        # print(spectral_dim.shape)
        # print(edge_index.shape)
        # print(edge_weight.shape)

        # 初始化边索引和边权重的列表
        new_edge_index = []
        new_edge_weight = []

        # 获取唯一的图索引
        unique_graphs = torch.unique(batch)

        # 遍历每个图
        for graph in unique_graphs:
            # 获取当前图的节点索引
            node_indices = (batch == graph).nonzero(as_tuple=True)[0]

            # 提取当前图的节点特征
            current_spectral_dim = spectral_dim[node_indices]

            # 计算节点之间的距离
            distance_matrix = torch.norm(current_spectral_dim[:, None] - current_spectral_dim[None, :], dim=-1)

            # 找到距离小于阈值的节点对
            edge_indices = torch.nonzero(distance_matrix < self.sl, as_tuple=False)

            # 确保 edge_indices 是有效的
            if edge_indices.size(0) > 0:
                # 将当前图的节点索引映射到全局索引
                edge_indices = node_indices[edge_indices]

                # 添加到新的边索引
                new_edge_index.append(edge_indices.T)

                # 添加边权重（所有边的权重为 1）
                if self.gm == 0:
                    edge_weights_for_graph = torch.ones(edge_indices.size(0), dtype=x.dtype)
                elif self.gm == 1:
                    if edge_indices.size(0) > 0:
                        distances = distance_matrix[edge_indices[:, 0], edge_indices[:, 1]]
                        edge_weights_for_graph = 1 / distances  # 使用距离的倒数作为权重
                    else:
                        edge_weights_for_graph = torch.tensor([]).to(device)
                # 添加边权重
                new_edge_weight.append(edge_weights_for_graph)

        # 拼接所有图的边索引和边权重
        new_edge_index = torch.cat(new_edge_index, dim=1)
        new_edge_weight = torch.cat(new_edge_weight)

        return new_edge_index.to(device), new_edge_weight.to(device)


