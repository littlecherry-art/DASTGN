import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import *

'''
Time-specific effect: Compute the fixed spatial weights of each specific time slice
'''
class TimeEffect(nn.Module):

    def __init__(self, node_hid_size, out_size, num_timestamps, adj):
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

        spatial_weights = []
        temp = temp_out.transpose(0, 1)
        for i in range(temp.size()[0]):
            temp_i = temp[i]

            spatial_scores = torch.matmul(temp_i, temp_i.transpose(0, 1).contiguous())/math.sqrt(self.input_size)
            attention = torch.sigmoid(spatial_scores)
            weights = attention.detach().numpy()
            spatial_weights.append(weights)

        spatial_weights = torch.tensor(spatial_weights)
        ST_weights = spatial_weights.repeat(self.num_timestamps, 1, 1)
        ST_weights = ST_weights.view(self.num_timestamps, self.num_timestamps, self.tot_nodes, self.tot_nodes)
        ST_weights = ST_weights.permute(0, 2, 1, 3)              # affected by historical spatial dependency
        ST_weights = ST_weights.reshape(self.dim, self.dim)

        # limit impact points based on spatial adjacency matrix, instead of the full connected matrix
        filter = Static_full(self.tot_nodes, self.num_timestamps, self.adj)

        output = ST_weights * filter

        return output


'''
Location-specific effect: Compute the fixed temporal weights at each specific region
'''
class GAT(nn.Module):

    def __init__(self, input_size):
        super(GAT, self).__init__()

        self.input_size = input_size

    def forward(self, input, adj):
        h = input

        out = []
        for i in range(h.size()[0]):

            h_i = h[i]

            spatial_scores = torch.matmul(h_i, h_i.transpose(0, 1).contiguous()) / math.sqrt(self.input_size)

            # set edges not connected as -1e12 in the mask matrix
            zero_vec = -1e12 * torch.ones_like(spatial_scores)
            # Only use weights of the connected edges to compute attention scores
            attention = torch.where(adj > 0, spatial_scores, zero_vec)
            attention = F.softmax(attention, dim=1)
            h_prime = torch.matmul(attention, h[i])
            out.append(h_prime)

        out_hid = torch.cat(out, dim=0).view(h.size()[0], h.size()[1], h.size()[2])

        return out_hid


class LocationEffect(nn.Module):

    def __init__(self, node_input_size, node_out_size, adj, num_timestamps):
        super(LocationEffect, self).__init__()

        self.gat = GAT(node_input_size)

        self.adj = adj
        self.num_timestamps = num_timestamps
        self.tot_nodes = adj.shape[0]
        self.dim = self.num_timestamps * self.tot_nodes
        self.input_size = node_input_size

    def forward(self, raw_features):
        node_features = self.gat(raw_features, self.adj)

        ## temporal dependency at each specific region
        spatial_weights = []
        spatial_features = node_features.transpose(0, 1)

        for i in range(spatial_features.size()[0]):
            spatial_i = spatial_features[i]

            temp_scores = torch.matmul(spatial_i, spatial_i.transpose(0, 1).contiguous()) / math.sqrt(self.input_size)
            attention = torch.sigmoid(temp_scores)
            weights = attention.detach().numpy()
            spatial_weights.append(weights)

        spatial_weights = torch.tensor(spatial_weights)
        ST_weights = spatial_weights.repeat(self.tot_nodes, 1, 1)
        ST_weights = ST_weights.view(self.tot_nodes, self.tot_nodes, self.num_timestamps, self.num_timestamps)
        ST_weights = ST_weights.permute(3, 1, 2, 0)            # affected by temporal dependency of self region
        ST_weights = ST_weights.reshape(self.dim, self.dim)

        # limit impact points based on spatial adjacency matrix, instead of the full connected matrix
        filter = Static_full(self.tot_nodes, self.num_timestamps, self.adj)

        output = ST_weights * filter

        return output




'''
Space-time interaction effect: directly generate the spatio-temporal weight matrix based on node features
'''
class STInter(nn.Module):

    def __init__(self, input_size, adj, num_timestamps):
        super(STInter, self).__init__()

        self.adj = adj
        self.num_timestamps = num_timestamps
        self.tot_nodes = adj.shape[0]
        self.input_size = input_size


    def forward(self, raw_features):
        node_features = raw_features.view(raw_features.size()[0] * raw_features.size()[1], -1)

        st_scores = torch.matmul(node_features, node_features.transpose(0, 1).contiguous()) / math.sqrt(self.input_size)
        st_scores = torch.sigmoid(st_scores)

        # limit impact points based on spatial adjacency matrix, instead of the full connected matrix
        filter = Static_full(self.tot_nodes, self.num_timestamps, self.adj)
        st_weight = torch.mul(st_scores, filter)

        return st_weight


'''
Integrate spatio-temporal weight matrices computed by characterizing three kinds of space-time effects
'''
class MultiChannel_layer(nn.Module):

    def __init__(self, input_size, adj_lists, num_timestamps):
        super(MultiChannel_layer, self).__init__()

        self.input_size = input_size
        self.tot_nodes = adj_lists.shape[0]
        self.dim = num_timestamps * self.tot_nodes

        self.Time = TimeEffect(input_size, input_size, num_timestamps, adj_lists)
        self.Location = LocationEffect(input_size, input_size * 2, adj_lists, num_timestamps)
        self.STInter = STInter(input_size, adj_lists, num_timestamps)

        self.TW = nn.Parameter(torch.FloatTensor(1, 1))
        self.SW = nn.Parameter(torch.FloatTensor(1, 1))
        self.STW = nn.Parameter(torch.FloatTensor(1, 1))

    def forward(self, his_raw_features):     # dim(his_raw_features)=(timestamp, num_vertices, num_features)

        temporal_matrix = self.Time(his_raw_features)
        temporal_gate = self.TW * temporal_matrix

        spatial_matrix = self.Location(his_raw_features)
        spatial_gate = self.SW * spatial_matrix

        st_matrix = self.STInter(his_raw_features)
        st_gate = self.STW * st_matrix

        gates = torch.cat((temporal_gate.unsqueeze(0), spatial_gate.unsqueeze(0), st_gate.unsqueeze(0)), 0)
        gates = F.softmax(gates, dim=0)

        ST_matrix = gates[0, :, :] * temporal_matrix + gates[1, :, :] * spatial_matrix + gates[2, :, :] * st_matrix

        return ST_matrix