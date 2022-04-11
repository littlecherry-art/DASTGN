from CoarseModule import *
from FineModule import *


'''
Spatio-Temporal GNN
'''
class SPTempGNN_layer(nn.Module):

    def __init__(self, D_temporal, A_temporal, tot_nodes, tw, fw):
        super(SPTempGNN_layer, self).__init__()

        self.tot_nodes = tot_nodes
        self.sp_temp = torch.mm(D_temporal, torch.mm(A_temporal, D_temporal))

        self.his_temporal_weight = tw
        self.his_final_weight = fw

    def forward(self, his_raw_features):
        his_self = his_raw_features
        his_temporal = self.his_temporal_weight.repeat(self.tot_nodes, 1) * his_raw_features
        his_temporal = torch.mm(self.sp_temp, his_temporal)

        his_combined = torch.cat([his_self, his_temporal], dim=1)
        his_raw_features = torch.relu(his_combined.mm(self.his_final_weight))

        return his_raw_features


class SPTempGNN_block(nn.Module):

    def __init__(self, input_size, out_size, adj_lists, device, GNN_layers, num_timestamps):
        super(SPTempGNN_block, self).__init__()

        self.num_timestamps = num_timestamps
        self.input_size = input_size
        self.out_size = out_size
        self.adj_lists = adj_lists
        self.tot_nodes = adj_lists.shape[0]
        self.device = device
        self.GNN_layers = GNN_layers
        self.dim = self.num_timestamps * self.tot_nodes

        self.his_temporal_weight = nn.Parameter(torch.FloatTensor(num_timestamps, out_size))
        self.his_final_weight = nn.Parameter(torch.FloatTensor(2 * out_size, out_size))
        self.final_weight = nn.Parameter(torch.FloatTensor(num_timestamps * out_size, num_timestamps * out_size))

        # a trainable parameter of the time-decaying interventions after quarantine
        self.theta = nn.Parameter(torch.FloatTensor(1, 1))

        self.Coarse_layer = Coarse_layer(self.tot_nodes, num_timestamps, input_size, adj_lists)
        self.Fine_layer = MultiChannel_layer(input_size, adj_lists, num_timestamps)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, attributes):     # dim(his_raw_features)=(T, S, F)
        his_raw_features = attributes[:, :, :self.input_size]              # delete time and intervention variables

        time_interval = attributes[:, :, self.input_size:(self.input_size+1)]
        interven = attributes[:, :, (self.input_size+1):(self.input_size+2)]
        interven_decay = torch.sigmoid(-self.theta * time_interval)
        interven_adjust = torch.where(interven == 0.5, interven_decay, interven)
        interven_adjust = interven_adjust.squeeze().view(self.dim, 1)

        for i in range(self.GNN_layers):
            his_raw_features = his_raw_features.contiguous().view(self.num_timestamps, self.tot_nodes, self.input_size)

            coarse_matrix = self.Coarse_layer(his_raw_features, interven_adjust)            # NT * NT
            fine_matrix = self.Fine_layer(his_raw_features)                                 # NT * NT

            A_temporal = coarse_matrix * fine_matrix                                        # final ST weighted matrix
            D_temporal = Degree_Matrix(A_temporal)

            his_raw_features = his_raw_features.contiguous().view(-1, self.input_size)

            sp_temp = SPTempGNN_layer(D_temporal, A_temporal, self.tot_nodes, self.his_temporal_weight, self.his_final_weight)
            his_raw_features = sp_temp(his_raw_features)

        his_list = []

        for timestamp in range(self.num_timestamps):
            st = timestamp * self.tot_nodes
            en = (timestamp + 1) * self.tot_nodes
            his_list.append(his_raw_features[st:en, :])

        his_embds = torch.cat(his_list, dim=1)
        embds = his_embds
        embds = torch.relu(self.final_weight.mm(embds.t()).t())

        return embds



"""
Downstream Task
"""
class Regression(nn.Module):

    def __init__(self, emb_size, out_size):
        super(Regression, self).__init__()

        self.layer = nn.Sequential(nn.Linear(emb_size, emb_size),
                                   nn.ReLU(),
                                   nn.Linear(emb_size, out_size),
                                   nn.ReLU())

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embds):
        logists = self.layer(embds)
        return logists