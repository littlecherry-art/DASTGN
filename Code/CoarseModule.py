import torch
import torch.nn as nn
import torch.nn.functional as F
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

    A_ST = kronecker(rho_CT1 * C_T, I_S) + kronecker(rho_CT2 * C_T, C_S) + kronecker(rho_IT * I_T, C_S)
    # A_ST = rho_CT1 * kronecker(C_T, I_S) + rho_CT2 * kronecker(C_T, C_S) + rho_IT * kronecker(I_T, C_S)

    return A_ST


'''
Coarse-grained module based on interventions
'''
class STNeighbor_layer(nn.Module):
    # return NT * NT dimensional weight matrix for each type of space-time neighboring blocks
    def __init__(self, tot_nodes, num_timestamps, input_size):

        super(STNeighbor_layer, self).__init__()
        self.dim = tot_nodes * num_timestamps

        self.W1 = nn.Parameter(torch.FloatTensor(input_size, input_size))
        self.W2 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.W3 = nn.Parameter(torch.FloatTensor(input_size, input_size))
        self.gate = nn.Sequential(nn.Linear(input_size, 1), nn.Sigmoid())

    def forward(self, features, interven, block_matrix):
        Infor = torch.mm(block_matrix, features)
        # space-time neighboring block information + intervention + self information
        feat = torch.mm(Infor, self.W1) + torch.mm(interven, self.W2) + torch.mm(features, self.W3)

        rho = self.gate(feat)
        # all impact points in the same block have the same weights
        block_weight = torch.mul(rho.repeat(1, self.dim), block_matrix)

        return block_weight


class Coarse_layer(nn.Module):
    # return NT * NT dimensional weight matrix of three types of space-time neighboring blocks
    def __init__(self, tot_nodes, num_timestamps, input_size, adj):

        super(Coarse_layer, self).__init__()
        self.tot_nodes = tot_nodes
        self.num_timestamps = num_timestamps
        self.adj = adj
        self.input_size = input_size

        self.Gate_IT = STNeighbor_layer(self.tot_nodes, num_timestamps, input_size)
        self.Gate_IS = STNeighbor_layer(self.tot_nodes, num_timestamps, input_size)
        self.Gate_CT = STNeighbor_layer(self.tot_nodes, num_timestamps, input_size)

    def forward(self, his_raw_features, interven):
        features = his_raw_features.contiguous().view(-1, self.input_size)
        A_IT = Static(self.tot_nodes, self.num_timestamps, self.adj, rho_IT=1, rho_CT1=0, rho_CT2=0)
        A_IS = Static(self.tot_nodes, self.num_timestamps, self.adj, rho_IT=0, rho_CT1=1, rho_CT2=0)
        A_CT = Static(self.tot_nodes, self.num_timestamps, self.adj, rho_IT=0, rho_CT1=0, rho_CT2=1)

        gate_IT = self.Gate_IT(features, interven, A_IT)
        gate_IS = self.Gate_IS(features, interven, A_IS)
        gate_CT = self.Gate_CT(features, interven, A_CT)

        gate = gate_IT + gate_IS + gate_CT

        return gate
