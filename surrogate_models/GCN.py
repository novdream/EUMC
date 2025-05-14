# import torch
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# import torch.optim as optim
# import utils
# from copy import deepcopy
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim,dropout=0.5):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#         self.dropout = dropout
#         self.norm = torch.nn.LayerNorm(hidden_dim)
        
#     def forward(self, features,edge_index,edge_weight=None):
#         x,edge_index= features,edge_index
#         x = F.relu(self.conv1(x, edge_index,edge_weight))
#         x = self.norm(x)
#         x = F.dropout(x, self.dropout,training=self.training)
#         x = self.conv2(x, edge_index,edge_weight)
#         return x
#         return F.log_softmax(x, dim=1)
#     def fit(self,args,features,edge_index,edge_weights,labels, idx_train, idx_val,train_iters,verbose = False):
#         if verbose:
#             print('=== training gcn model ===')
#         optimizer = optim.Adam(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
#         loss = CrossEntropyLoss()
#         best_acc_val = 0
        
        
#         for i in range(train_iters):
#             self.train()
#             optimizer.zero_grad()
#             output = self.forward(features, edge_index,edge_weights)
#             loss_train = loss(output[idx_train], labels[idx_train])
#             loss_train.backward()
#             optimizer.step()
            
#             self.eval()
#             output = self.forward(features, edge_index,edge_weights)
#             loss_val = loss(output[idx_val], labels[idx_val])
#             acc_val = utils.calculate_accuracy(output[idx_val], labels[idx_val])
        
#             if verbose and i % 10 == 9:
#                 print('Epoch {}, training loss: {}'.format(i+1, loss_train.item()))
#                 print("acc_val: {:.4f}".format(acc_val))
#             if acc_val > best_acc_val:
#                 best_acc_val = acc_val
#                 self.output = output
#                 weights = deepcopy(self.state_dict())

#         if verbose:
#             print('=== picking the best model according to the performance on validation ===')
#         self.load_state_dict(weights)
#         self.eval()
    
    
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index,edge_weight=None):
        # edge_index = edge_index[:, edge_mask]
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1)
    def get_h(self, x, edge_index,edge_weight):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
    
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
       
        # torch.cuda.empty_cache()
    def new_fit(self,features_list, edge_index_list, edge_weight_list, labels_list, btk_nodes, idx_val, weight,train_iters=200, verbose=False):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        idx_train, idx_attach= btk_nodes
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            for i in range(len(features_list)):
                
                output = self.forward(features_list[i], edge_index_list[i], edge_weight_list[i])
                # TODO
                # loss_train = F.nll_loss(output[idx_train], labels_list[i][idx_train])
                loss_train = F.nll_loss(output[idx_train], labels_list[i][idx_train],weight=weight)
                loss_atk = F.nll_loss(output[idx_attach], labels_list[i][idx_attach])
                loss_train += loss_atk
                
                loss_train.backward(retain_graph=True)
            optimizer.step()

            self.eval()
            acc_val = 0
            for i in range(len(features_list)):
                output = self.forward(features_list[i], edge_index_list[i], edge_weight_list[i])
                loss_val = F.nll_loss(output[idx_val], labels_list[i][idx_val])
                acc_val += utils.calculate_accuracy(output[idx_val], labels_list[i][idx_val])
            acc_val = acc_val/len(features_list)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        
    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.calculate_accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # torch.cuda.empty_cache()

    

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = utils.calculate_accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.calculate_accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids

# %%
