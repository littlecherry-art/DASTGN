import os
import torch
import numpy as np
import pandas as pd


"""
Data Loader
"""
class DataLoader:

    def __init__(self, adj_path, stfeature_path, num_timestamps, pred_len, interven_start, interven_end, seed):

        super(DataLoader, self).__init__()

        self.adj_path = adj_path
        self.stfeature_path = stfeature_path

        self.num_timestamps = num_timestamps
        self.pred_len = pred_len
        self.seed = seed

        self.interven_start = interven_start
        self.interven_end = interven_end

    def load_data(self):
        # Import dynamic features
        stfeature_files = os.listdir(self.stfeature_path)

        stfeature = []

        for i in range(len(stfeature_files)):
            df = pd.read_csv(self.stfeature_path + "/" + stfeature_files[i])
            df['time_interval'] = [i + 1 - self.interven_end] * len(df)               # add timestamps for each df
            df['interven'] = 0                                                        # interventions before quarantine as 0
            if i >= (self.interven_start-1) and i < (self.interven_end-1):
                df['interven'] = 1                                                    # interventions during quarantine as 1
            elif i >= (self.interven_end-1):
                df['interven'] = 0.5                                                  # interventions after quarantine as 0.5
            stfeature.append(df.values)

        Attr = np.array(stfeature)
        Attr = Attr.astype(np.float32)

        ## Normalization
        infect = Attr[:, :, :1]
        time = Attr[:, :, (Attr.shape[2]-2):(Attr.shape[2]-1)]
        interven = Attr[:, :, (Attr.shape[2]-1):(Attr.shape[2])]
        X = Attr[:, :, 1:(Attr.shape[2]-2)]

        means = np.mean(X, axis=(0, 1))
        X = X - means.reshape(1, 1, -1)
        stds = np.std(X, axis=(0, 1))
        stds[stds <= 10e-5] = 10e-5  # Prevent infs
        X = X / stds.reshape(1, 1, -1)
        data = np.concatenate((infect, X, time, interven), axis=2)

        # Import adj
        A = pd.read_csv(self.adj_path).values

        return A, data


    def generate_dataset(self, data):
        """
        Take node features for the graph and divides them into multiple samples
        along the time-axis by sliding a window of size (num_timesteps_input+
        num_timesteps_output) across it with one step.
        :type data: dataframe
        :param data: Node features of shape (num_vertices, num_features,
        num_timesteps)
        :return:
            - Node features divided into multiple samples. Shape is
              (num_samples, timestamp, num_vertices, num_features).
            - Node targets for samples. Since the dimension of the predicted
              feature is 1 in this case, the shape of target is
              (num_samples, num_vertices, pred_len).
        """
        # Generate the beginning index and the ending index of a sample, which
        # contains (num_points_for_training + num_points_for_predicting) points

        indices = [(i, i + (self.num_timestamps + self.pred_len)) for i
                   in range(data.shape[0] - (
                    self.num_timestamps + self.pred_len) + 1)]

        # Save samples
        features, target = [], []
        for i, j in indices:
            features.append(
                data[i: i + self.num_timestamps, :, :])
            pred = data[i + self.num_timestamps: j, :, 0].transpose(1, 0)
            target.append(pred)

        # (number of samples, historical timestamps, number of regions, number of features)
        feat_torch = torch.from_numpy(np.array(features))
        # (number of samples, number of regions, pred_len)
        target_torch = torch.from_numpy(np.array(target))

        return feat_torch, target_torch

    def split_data(self):
        adj, X = self.load_data()

        split_line1 = int(X.shape[0] * 0.8)                             # 80% training data

        train_original_data = X[:split_line1, :, :]
        val_original_data = X[split_line1:, :, :]                        # validation dataset = test dataset
        test_original_data = X[split_line1:, :, :]

        training_input, training_target = self.generate_dataset(train_original_data)
        val_input, val_target = self.generate_dataset(val_original_data)
        test_input, test_target = self.generate_dataset(test_original_data)

        return training_input, training_target, val_input, val_target, test_input, test_target, adj



"""
Evaluation functions
"""
def evaluate(test_nodes, raw_features, labels, DASTGN, regression, test_loss):
    models = [DASTGN, regression]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)

    val_nodes = test_nodes
    embs = DASTGN(raw_features)
    predicts = regression(embs)
    loss_sup = torch.nn.MSELoss()(predicts, labels)
    loss_sup /= len(val_nodes)
    test_loss += loss_sup.item()

    for param in params:
        param.requires_grad = True

    return predicts, test_loss


"""
The binary spatio-temporal adjacency matrix used in USTGCN
"""
def Static_full(n, t, A):
    """
    :param n: the dimension of the spatial adjacency matrix
    :param t: the length of periods
    :param A: the spatial adjacency matrix
    :return: the full USTGCN spatio-temporal adjacency matrix
    """
    I_S = torch.diag_embed(torch.ones(n))
    I_T = torch.diag_embed(torch.ones(t))

    C_S = A
    C_T = torch.tril(torch.ones(t, t), diagonal=-1)

    S = I_S + C_S
    A_ST = kronecker(C_T, S) + kronecker(I_T, C_S)

    return A_ST


"""
Use kronecker product to construct the spatio-temporal adjacency matrix
"""
def kronecker(A, B):
    """
    :param A: the temporal adjacency matrix
    :param B: the spatial adjacency matrix
    :return: the adjacency matrix of one space-time neighboring block
    """
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.contiguous().view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB


'''
Construct the degree matrix based on the spatio-temporal matrix
'''
def Degree_Matrix(ST_matrix):

    row_sum = torch.sum(ST_matrix, 0)

    ## degree matrix
    dim = len(ST_matrix)
    D_matrix = torch.zeros(dim, dim)
    for i in range(dim):
        D_matrix[i, i] = 1 / max(torch.sqrt(row_sum[i]), 1)

    return D_matrix