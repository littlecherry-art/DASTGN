import torch
import argparse
import random
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
import torch.nn.functional as F

from model import *
from utils import *

"""
Disease Model
"""
class DiseaseModel:

    def __init__(self, train_data, train_label, val_data, val_label, test_data, test_label,
                 adj, input_size, out_size, GNN_layers, epochs, device, num_timestamps, pred_len,
                 save_flag, PATH, t_debug, b_debug):

        super(DiseaseModel, self).__init__()

        self.train_data, self.train_label = train_data, train_label
        self.val_data, self.val_label = val_data, val_label
        self.test_data, self.test_label = test_data, test_label
        self.adj = adj
        self.all_nodes = [i for i in range(self.adj.shape[0])]

        self.input_size = input_size
        self.out_size = out_size
        self.GNN_layers = GNN_layers
        self.day = input_size
        self.device = device
        self.epochs = epochs
        self.regression = Regression(input_size * num_timestamps, pred_len)
        self.num_timestamps = num_timestamps
        self.pred_len = pred_len

        self.node_bsz = 512
        self.PATH = PATH
        self.save_flag = save_flag

        self.train_data = torch.FloatTensor(self.train_data).to(device)
        self.val_data = torch.FloatTensor(self.val_data).to(device)
        self.test_data = torch.FloatTensor(self.test_data).to(device)
        self.train_label = torch.FloatTensor(self.train_label).to(device)
        self.val_label = torch.FloatTensor(self.val_label).to(device)
        self.test_label = torch.FloatTensor(self.test_label).to(device)
        self.all_nodes = torch.LongTensor(self.all_nodes).to(device)
        self.adj = torch.FloatTensor(self.adj).to(device)

        self.t_debug = t_debug
        self.b_debug = b_debug

    def run_model(self):

        timeStampModel = SPTempGNN_block(self.input_size, self.out_size, self.adj,
                                     self.device, self.GNN_layers, self.num_timestamps)
        timeStampModel.to(self.device)

        regression = self.regression
        regression.to(self.device)

        self.best_val = float("Inf")
        self.best_epoch = float("Inf")

        lr = 0.001
        # lr = 0.003

        train_loss_his = []
        loss_idx = []
        val_loss_his = []
        RMSE_his = []
        MAE_his = []

        train_loss = torch.tensor(0.).to(self.device)
        for epoch in range(self.epochs):
            print("Epoch: ", epoch, " running...")

            tot_timestamp = len(self.train_data)
            if self.t_debug:
                tot_timestamp = 60
            idx = np.random.permutation(tot_timestamp)

            for data_timestamp in idx:

                tr_data = self.train_data[data_timestamp]
                tr_label = self.train_label[data_timestamp]

                timeStampModel, regression, train_loss = apply_model(self.all_nodes, timeStampModel,
                                                                     regression, self.node_bsz, self.device, tr_data,
                                                                     tr_label, train_loss, lr, self.pred_len)

                if self.b_debug:
                    break

            train_loss /= len(idx)
            # if epoch <= 24 and epoch % 8 == 0:
            #     lr *= 0.5
            # else:
            #     lr = 0.0001

            print("Train avg loss: ", train_loss)

            pred = []
            label = []
            tot_timestamp = len(self.val_data)

            if self.t_debug:
                tot_timestamp = 60

            idx = np.random.permutation(tot_timestamp)
            val_loss = torch.tensor(0.).to(self.device)
            for data_timestamp in idx:

                # val_label
                raw_features = self.val_data[data_timestamp]
                val_label = self.val_label[data_timestamp]

                # evaluate
                temp_predicts, val_loss = evaluate(self.all_nodes, raw_features, val_label,
                                                    timeStampModel, regression, val_loss)

                label = label + val_label.detach().tolist()
                pred = pred + temp_predicts.detach().tolist()

                if self.b_debug:
                    break

            val_loss /= len(idx)
            print("Average Validation Loss: ", val_loss)

            if epoch > 0 and val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = epoch

                if self.save_flag:
                    torch.save(timeStampModel,
                               self.PATH + "/" + str(self.pred_len) + "day_" + str(self.GNN_layers) + "layer_bestTmodel.pth")
                    torch.save(regression,
                               self.PATH + "/" + str(self.pred_len) + "day_" + str(self.GNN_layers) + "layer_bestRegression.pth")

            if epoch - self.best_epoch > 30 and val_loss > self.best_val:
                break

            RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
            RMSE = torch.sqrt(RMSE).item()
            MAE = mean_absolute_error(pred, label)

            print("Min validation loss: ", self.best_val)
            print("Best epoch: ", self.best_epoch)
            print("===============================================")

            # Store historic results
            loss_idx.append(epoch)
            train_loss_his.append(train_loss.item())
            val_loss_his.append(val_loss.item())
            RMSE_his.append(RMSE)
            MAE_his.append(MAE)
            report = {"epoch": loss_idx,
                      "train_avg_loss": train_loss_his,
                      "val_avg_loss": val_loss_his,
                      "val_RMSE": RMSE_his,
                      "val_MAE": MAE_his}
            report = pd.DataFrame(report)
            print(report)
            print("===============================================")

        return

    def run_Trained_Model(self):
        timeStampModel = torch.load(self.PATH + "/" + str(self.pred_len) + "day_" + str(self.GNN_layers) + "layer_bestTmodel.pth")
        regression = torch.load(self.PATH + "/" + str(self.pred_len) + "day_" + str(self.GNN_layers) + "layer_bestRegression.pth")

        pred = []
        label = []
        tot_timestamp = len(self.test_data)
        idx = np.random.permutation(tot_timestamp)
        test_loss = torch.tensor(0.).to(self.device)
        for data_timestamp in idx:
            # test_label
            raw_features = self.test_data[data_timestamp]
            test_label = self.test_label[data_timestamp]

            # evaluate
            temp_predicts, test_loss = evaluate(self.all_nodes, raw_features, test_label,
                                                timeStampModel, regression, test_loss)

            label = label + test_label.detach().tolist()
            pred = pred + temp_predicts.detach().tolist()

        test_loss /= len(idx)
        print("Average Test Loss: ", test_loss)

        RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
        RMSE = torch.sqrt(RMSE).item()
        MAE = mean_absolute_error(pred, label)

        print("Test RMSE: ", RMSE)
        print("Test MAE: ", MAE)
        print("===============================================")

        report = {"Best_epoch": self.best_epoch,
                  "Min_val_loss": self.best_val.item(),
                  "Ave_test_loss": test_loss.item(),
                  "Test_RMSE": RMSE,
                  "Test_MAE": MAE}
        report = pd.DataFrame(report, index=[0])
        filename = self.PATH + "/" + str(self.pred_len) + "day_" + str(self.GNN_layers) + "layer.csv"
        report.to_csv(filename, encoding='gbk')

        return


"""
Applying Model
"""
def apply_model(train_nodes, SPTempGNN_block, regression,
                node_batch_sz, device, train_data, train_label, avg_loss, lr, pred_len):
    models = [SPTempGNN_block, regression]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)

    optimizer.zero_grad()  # set gradients in zero...
    for model in models:
        model.zero_grad()  # set gradients in zero

    node_batches = math.ceil(len(train_nodes) / node_batch_sz)

    loss = torch.tensor(0.).to(device)
    # window slide
    raw_features = train_data
    labels = train_label
    for index in range(node_batches):
        nodes_batch = train_nodes[index * node_batch_sz:(index + 1) * node_batch_sz]
        nodes_batch = nodes_batch.view(nodes_batch.shape[0], 1)
        labels_batch = labels[nodes_batch]
        labels_batch = labels_batch.view(len(labels_batch), pred_len)
        embs_batch = SPTempGNN_block(raw_features)  # Finds embeddings for all the ndoes in nodes_batch

        logists = regression(embs_batch)
        loss_sup = torch.nn.MSELoss()(logists, labels_batch)
        loss_sup /= len(nodes_batch)
        loss += loss_sup

    avg_loss += loss.item()

    loss.backward()
    for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    return SPTempGNN_block, regression, avg_loss


"""#Training"""
parser = argparse.ArgumentParser(description='pytorch version of ESTGCN')
parser.add_argument('-f')


parser.add_argument('--feature_path', type=str, default='../../China/Province_feature')
parser.add_argument('--adj_path', type=str, default='../../China/ChinaProvAdj.csv')
parser.add_argument('--save_path', type=str, default='../../China/GitHub_test')
parser.add_argument('--GNN_layers', type=int, default=2)
parser.add_argument('--num_timestamps', type=int, default=14)
parser.add_argument('--pred_len', type=int, default=7)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--trained_model', action='store_true')
parser.add_argument('--save_model', action='store_true', default=True)
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--out_size', type=int, default=3)
parser.add_argument('--interven_start', type=int, default=10)
parser.add_argument('--interven_end', type=int, default=49)

args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
print('DEVICE:', device)


"""
Main Function
"""

print('Static COVID-19 Forecasting GNN with Historical and Current Model')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

data_loader = DataLoader(args.adj_path, args.feature_path, args.num_timestamps, args.pred_len, args.interven_start,
                         args.interven_end, args.seed)
train_data, train_label, val_data, val_label, test_data, test_label, adj = data_loader.split_data()


save_flag = True
t_debug = False
b_debug = False
hModel = DiseaseModel(train_data, train_label, val_data, val_label, test_data, test_label, adj, args.input_size,
                      args.out_size, args.GNN_layers, args.epochs, device, args.num_timestamps, args.pred_len, save_flag,
                      args.save_path, t_debug, b_debug)


hModel.run_model()            # train model and evaluate
hModel.run_Trained_Model()    # run trained model

