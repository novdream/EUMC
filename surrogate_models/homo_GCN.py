import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn.init as init

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(GCN, self).__init__()
        # self.conv1 = GCNConv(in_feats, in_feats)
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = nn.Linear(hidden_size, out_feats)
        # self.conv2 = GCNConv(hidden_size, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index,edge_weight=None):
        h = self.conv1(x, edge_index,edge_weight)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        # h = self.conv2(h, edge_index,edge_weight)
        h = self.conv2(h)
        return h
    
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=3, dropout=0.5):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_size, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size * num_heads, out_feats, heads=1, concat=False, dropout=dropout)
        self.last = nn.Linear(hidden_size * num_heads, out_feats)
    def forward(self, x, edge_index,edge_weight=None):
        # Apply the first GAT layer
        x = self.gat1(x, edge_index,edge_weight)
        x = F.relu(x)  # Non-linear activation
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last(x)
        # Apply the second GAT layer
        # x = self.gat2(x, edge_index,edge_weight)
        
        return x