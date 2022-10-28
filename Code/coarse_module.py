# import torch
import torch.nn as nn
# import torch.nn.functional as F
from utils import *


'''
A general function to construct space-time neighboring blocks / space-time dependency structures
'''
def Static(n, t, A, rho_IT, rho_CT1, rho_CT2):
    """
    :param n: the dimension of the spatial adjacency matrix
    :param t: the length of periods
    :param A: the spatial adjacency matrix
    :param rho_IT: the trainable paramter of the current states of neighbors
    :param rho_CT1: the trainable paramter of the historical states of self
    :param rho_CT2: the trainable paramter of the historical states of neighbors
    :return: a space-time dependency structure matrix
    """
    I_S = torch.diag_embed(torch.ones(n))
    I_T = torch.diag_embed(torch.ones(t))

    C_S = A
    C_T = torch.tril(torch.ones(t, t), diagonal=-1)

    A_ST = kronecker(rho_IT * I_T, C_S) + kronecker(rho_CT1 * C_T, I_S) + kronecker(rho_CT2 * C_T, C_S)
    # A_ST = rho_CT1 * kronecker(C_T, I_S) + rho_CT2 * kronecker(C_T, C_S) + rho_IT * kronecker(I_T, C_S)

    return A_ST


'''
Coarse-grained module based on interventions
'''
class STNB_layer(nn.Module):
    # return NT * NT dimensional weight matrix for each type of space-time neighbor blocks
    def __init__(self, tot_nodes, num_timestamps, input_size, adj_sum):

        super(STNB_layer, self).__init__()
        self.dim = tot_nodes * num_timestamps
        self.judge = adj_sum * num_timestamps

        self.w1 = nn.Parameter(torch.FloatTensor(1, 1))
        self.w2 = nn.Parameter(torch.FloatTensor(1, 1))
        self.gate = nn.Sequential(nn.Linear(input_size, 1), nn.Sigmoid())

    def forward(self, features, interven, block_matrix):
        # process cumulative/current intervention
        block_sum = torch.sum(block_matrix)
        if block_sum > self.judge:
            interven_cum = torch.cumsum(interven, dim=0) - interven           # cumulative historical interventions
            interven_adjust = interven_cum.squeeze().view(self.dim, 1)
        else:
            interven_adjust = interven.squeeze().view(self.dim, 1)            # current intervention


        Infor = torch.mm(block_matrix, features)
        feat = self.w1 * Infor + interven_adjust.repeat(1,3) + self.w2 * features

        rho = self.gate(feat)
        # weights of all impact points in the same block are identical
        block_weight = torch.mul(rho.repeat(1, self.dim), block_matrix)

        return block_weight


class Coarse_module(nn.Module):
    # return NT * NT dimensional weight matrix of three types of space-time neighbor blocks
    def __init__(self, tot_nodes, num_timestamps, input_size, adj):

        super(Coarse_module, self).__init__()
        self.tot_nodes = tot_nodes
        self.num_timestamps = num_timestamps
        self.adj = adj
        self.adj_sum = torch.sum(self.adj)
        self.input_size = input_size

        self.Gate_IT = STNB_layer(self.tot_nodes, num_timestamps, input_size, self.adj_sum)
        self.Gate_CS = STNB_layer(self.tot_nodes, num_timestamps, input_size, self.adj_sum)
        self.Gate_CT = STNB_layer(self.tot_nodes, num_timestamps, input_size, self.adj_sum)

    def forward(self, his_raw_features, interven):
        features = his_raw_features.contiguous().view(-1, self.input_size)
        A_IT = Static(self.tot_nodes, self.num_timestamps, self.adj, rho_IT=1, rho_CT1=0, rho_CT2=0)     # current states of neighbors
        A_CS = Static(self.tot_nodes, self.num_timestamps, self.adj, rho_IT=0, rho_CT1=1, rho_CT2=0)     # historical states of self
        A_CT = Static(self.tot_nodes, self.num_timestamps, self.adj, rho_IT=0, rho_CT1=0, rho_CT2=1)     # historical states of neighbors

        gate_IT = self.Gate_IT(features, interven, A_IT)
        gate_CS = self.Gate_CS(features, interven, A_CS)
        gate_CT = self.Gate_CT(features, interven, A_CT)

        gate = gate_IT + gate_CS + gate_CT

        return gate
