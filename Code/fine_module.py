# import torch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import time


class TimeEffect(nn.Module):
    def __init__(self, node_hid_size, out_size, num_timestamps, adj):
        """
        Time-specific effect: Compute the fixed spatial dependency of each specific time slice
        """
        super(TimeEffect, self).__init__()

        self.gru = nn.GRU(input_size=node_hid_size, hidden_size=out_size, batch_first=True)

        self.adj = adj
        self.num_timestamps = num_timestamps
        self.tot_nodes = adj.shape[0]
        self.dim = self.num_timestamps * self.tot_nodes
        self.input_size = node_hid_size

    def forward(self, raw_features):
        node_features = raw_features.transpose(0, 1)
        temp_out, temp_hid = self.gru(node_features)

        temp = temp_out.transpose(0, 1)
        spatial_scores = torch.matmul(temp, temp.transpose(1, 2).contiguous()) / math.sqrt(self.input_size)
        spatial_weights = torch.sigmoid(spatial_scores)

        ST_weights = spatial_weights.repeat(self.num_timestamps, 1, 1)
        ST_weights = ST_weights.view(self.num_timestamps, self.num_timestamps, self.tot_nodes, self.tot_nodes)
        ST_weights = ST_weights.permute(0, 2, 1, 3)              # affected by fixed spatial dependency of each historical time slice
        ST_weights = ST_weights.reshape(self.dim, self.dim)

        return ST_weights


'''
Space-specific effect: Compute the fixed temporal weights at each specific region
'''
class GAT(nn.Module):

    def __init__(self, input_size):
        super(GAT, self).__init__()

        self.input_size = input_size

    def forward(self, input, adj):

        spatial_scores = torch.matmul(input, input.transpose(1, 2))/math.sqrt(self.input_size)
        zero_vec = -1e12 * torch.ones_like(spatial_scores)
        attention = torch.where(adj > 0, spatial_scores, zero_vec)
        attention = F.softmax(attention, dim=2)
        out_hid = torch.matmul(attention, input)

        return out_hid


class SpaceEffect(nn.Module):

    def __init__(self, node_input_size, adj, num_timestamps):
        super(SpaceEffect, self).__init__()

        self.gat = GAT(node_input_size)

        self.adj = adj
        self.num_timestamps = num_timestamps
        self.tot_nodes = adj.shape[0]
        self.dim = self.num_timestamps * self.tot_nodes
        self.input_size = node_input_size

    def forward(self, raw_features):
        node_features = self.gat(raw_features, self.adj)

        ## temporal dependency at each specific region
        spatial_features = node_features.transpose(0, 1)
        temp_scores = torch.matmul(spatial_features, spatial_features.transpose(1, 2).contiguous())/math.sqrt(self.input_size)
        spatial_weights = torch.sigmoid(temp_scores)

        ST_weights = spatial_weights.repeat(self.tot_nodes, 1, 1)
        ST_weights = ST_weights.view(self.tot_nodes, self.tot_nodes, self.num_timestamps, self.num_timestamps)
        ST_weights = ST_weights.permute(3, 1, 2, 0)            # affected by temporal dependency of self region
        ST_weights = ST_weights.reshape(self.dim, self.dim)

        return ST_weights


'''
Direct interaction effect: directly compute the weight between any two space-time points
'''
class DirectEffect(nn.Module):

    def __init__(self, input_size, adj, num_timestamps):
        super(DirectEffect, self).__init__()

        self.adj = adj
        self.num_timestamps = num_timestamps
        self.tot_nodes = adj.shape[0]
        self.input_size = input_size


    def forward(self, raw_features):
        node_features = raw_features.view(raw_features.size()[0] * raw_features.size()[1], -1)
        st_scores = torch.matmul(node_features, node_features.transpose(0, 1).contiguous()) / math.sqrt(self.input_size)
        st_scores = torch.sigmoid(st_scores)

        return st_scores


'''
Integrate three NT*NT dimensional spatio-temporal weight matrices generated by different space-time effects
'''
class MultiEffectFusion(nn.Module):

    def __init__(self, input_size, adj_lists, num_timestamps):
        super(MultiEffectFusion, self).__init__()

        self.input_size = input_size
        self.tot_nodes = adj_lists.shape[0]
        self.num_timestamps = num_timestamps
        self.adj = adj_lists
        self.dim = num_timestamps * self.tot_nodes

        self.Time = TimeEffect(input_size, input_size, num_timestamps, adj_lists)
        self.Space = SpaceEffect(input_size, adj_lists, num_timestamps)
        self.Direct = DirectEffect(input_size, adj_lists, num_timestamps)

        self.TW = nn.Parameter(torch.FloatTensor(1, 1))
        self.SW = nn.Parameter(torch.FloatTensor(1, 1))
        self.DW = nn.Parameter(torch.FloatTensor(1, 1))


    def forward(self, his_raw_features):     # dim(his_raw_features)=(timestamp, num_vertices, num_features)
        filter = Static_full(self.tot_nodes, self.num_timestamps, self.adj)

        temporal_matrix = self.Time(his_raw_features) * filter
        temporal_gate = self.TW * temporal_matrix

        spatial_matrix = self.Space(his_raw_features) * filter
        spatial_gate = self.SW * spatial_matrix

        direct_matrix = self.Direct(his_raw_features) * filter
        direct_gate = self.DW * direct_matrix

        gates = torch.cat((temporal_gate.unsqueeze(0), spatial_gate.unsqueeze(0), direct_gate.unsqueeze(0)), 0)
        gates = F.softmax(gates, dim=0)

        ST_matrix = gates[0, :, :] * temporal_matrix + gates[1, :, :] * spatial_matrix + gates[2, :, :] * direct_matrix

        return ST_matrix
