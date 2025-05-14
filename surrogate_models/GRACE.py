import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
import utils
import torch.optim as optim
from copy import deepcopy
import numpy as np
def clr_loss(poison_model,clean_model,idx_clean,idx_attach,features,edge_index):
    idx = torch.cat([idx_clean,idx_attach])
    poison_model.train()
    z_p = poison_model(features, edge_index)
    z_c = clean_model(features, edge_index)
    cos_sim = F.cosine_similarity(z_p[idx], z_c[idx], dim=-1)
    loss = cos_sim
    # print(loss.shape)
    return 1-loss.mean()

import torch

def bkd_loss(poison_model, idx_attach, idx_clean, poison_features, poison_edge_index, target_labels, top_k=5):
    poison_model.train()
    

    Z_p = poison_model(poison_features, poison_edge_index)
    

    z1 = Z_p[idx_attach]
    z2 = Z_p[idx_clean]
    
    z1_norm = z1.norm(p=2, dim=-1, keepdim=True)  #  [attach_num, 1]
    z2_norm = z2.norm(p=2, dim=-1, keepdim=True)  #  [clean_num, 1]
    
    dot_product = torch.mm(z1, z2.T)  #  [attach_num, clean_num]
    
    cos_sim = dot_product / (z1_norm * z2_norm.T)  #  [attach_num, clean_num]
    
    cos_sim = cos_sim.reshape(cos_sim.shape[0], -1, top_k)  
    labels_expanded = target_labels.unsqueeze(1).unsqueeze(2)  
    # labels_expanded = labels_expanded.repeat(1,cos_sim.shape[1], top_k)
    selected_values = cos_sim.gather(1, labels_expanded.repeat(1, cos_sim.shape[1],top_k))
    
    loss = selected_values.mean(dim=1).mean() 
    
    return 1 - loss


def trainGRACE(model, x, edge_index,labels,idx,num_class,p):
    model.train()
    edge_index_1 = dropout_adj(edge_index, p=0.2)[0]
    edge_index_2 = dropout_adj(edge_index, p=0.4)[0]
    x_1 = drop_feature(x, 0.3)
    x_2 = drop_feature(x, 0.4)
    # edge_index_1 = dropout_adj(edge_index, p=0.2)[0]
    # edge_index_2 = dropout_adj(edge_index, p=0.2)[0]
    # x_1 = drop_feature(x, 0.2)
    # x_2 = drop_feature(x, 0.2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1[idx], z2[idx], batch_size=0)
   
    return loss
def trainClass(classifer,optimizer,embedding,y):
    for _ in range(100):
        optimizer.zero_grad()
        output = classifer(embedding)
        loss =  F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
    
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x



def test(model ,x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = F.relu(self.conv[i](x, edge_index))
        return x


class GRACE(torch.nn.Module):
    def __init__(self, encoder,nfeat, nhid: int,nclass,
                 layer, device ,args,dropout=0.5,tau: float = 0.4,lambd = 1e-3,method = 'GRACE'):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.device = device
        # Encoder(nfeat, nhid,k=layer).to(device)
        self.args = args
        self.tau = self.args.tau
        self.fc1 = torch.nn.Linear(nhid, nhid)
        self.fc2 = torch.nn.Linear(nhid, nhid)
        self.method = method
        self.lambd = lambd
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,edge_weight=None) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def cca_loss(self, z1, z2, N):
        eps = 1e-8 
        C = 20
        if self.args.dataset=='Facebook':
            C = 30
        # elif self.args.dataset=='Bitcoin' and self.args.attack_single == True:
        #     C = 5
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + eps)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + eps)
        
        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(self.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2) + C
        
 

        return loss 
    
    def test(self, classifer,features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            z = self.forward(features, edge_index,edge_weight)
            output = classifer(z)
            acc_test = utils.calculate_accuracy(output[idx_test], labels[idx_test])
            
            class_wise_accuracy = {i: 0 for i in range(7)}
            total_samples_per_class = {i: 0 for i in range(7)}
            predictions = torch.argmax(output[idx_test], dim=1)
            
            for i in range(7):
                class_mask = (labels[idx_test] == i)  
                if torch.any(class_mask):  
                    num_correct = torch.sum((predictions == i) & class_mask).item()
                    num_samples = torch.sum(class_mask).item()
                    class_wise_accuracy[i] = num_correct / num_samples
                    total_samples_per_class[i] = num_samples  
            
            
            for cls, acc in class_wise_accuracy.items():
                print(f"Accuracy for class {cls}: {acc:.4f}",flush=True)
                print(f"Number of samples for class {cls}: {total_samples_per_class[cls]}",flush=True)
        return float(acc_test)
    
    def new_fit(self,features_list, edge_index_list, edge_weight_list, labels_list, btk_nodes, idx_val, weight,train_iters=200, verbose=False):
        if verbose:
            print('=== training gcn model ===')
        # optimizer = optim.Adam(self.parameters(), lr=self.args.train_lr, weight_decay=self.args.weight_decay)
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.args.weight_decay)
        best_loss = 1e6
        
        idx_train, idx_attach= btk_nodes
        if idx_attach == None or len(idx_attach) == 0:
            idx_new = idx_train
        else:
            idx_new = torch.cat([idx_train,idx_attach])
        for i in range(train_iters):
            optimizer.zero_grad()
            self.train()
            for t in range(len(features_list)):
                x = features_list[t]
                N = x.shape[0]
                edge_index = edge_index_list[t]
                edge_index_1 = dropout_adj(edge_index, p=self.args.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(edge_index, p=self.args.drop_edge_rate_2)[0]
                x_1 = drop_feature(x,self.args.drop_feature_rate_1)
                x_2 = drop_feature(x, self.args.drop_feature_rate_2)
                z1 = self.forward(x_1, edge_index_1)
                z2 = self.forward(x_2, edge_index_2)
                if self.method == 'GRACE':
                    loss = self.loss(z1[idx_new], z2[idx_new], batch_size=0)
                        
                elif self.method == 'CCA-SSG':
                    loss_train = self.cca_loss(z1[idx_train], z2[idx_train], N)
                    if idx_attach == None or len(idx_attach) == 0:
                        # loss_atk = 0.0
                        loss = loss_train
                    else:
                        loss_atk = self.cca_loss(z1[idx_attach], z2[idx_attach], N)
                        loss =self.args.atk_weights * loss_atk + loss_train
                loss.backward(retain_graph=True)
            
            optimizer.step()
            
            
            self.eval()
            x = features_list[0]
            N = x.shape[0]
            edge_index = edge_index_list[0]
            edge_index_1 = dropout_adj(edge_index, p=self.args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=self.args.drop_edge_rate_2)[0]
            x_1 = drop_feature(x,self.args.drop_feature_rate_1)
            x_2 = drop_feature(x, self.args.drop_feature_rate_2)
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            if self.method == 'GRACE':
               
                loss = self.loss(z1[idx_val], z2[idx_val], batch_size=0)
            elif self.method == 'CCA-SSG':
               
                loss = self.cca_loss(z1[idx_val], z2[idx_val], N)
                if (i+1) % 10 == 0:
                    print('loss:',loss.item(),flush=True)
            if loss < best_loss:
                weights = deepcopy(self.state_dict()) 
                best_loss = loss
          
        self.load_state_dict(weights)
    def single_attack_new_fit(self,clean_model,idx_train,idx_attach,features,edge_index,poison_x,poison_edge_index,poison_labels,train_iters=300):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.args.weight_decay)
        best_loss = 1e6
        for i in range(train_iters):
            optimizer.zero_grad()
            self.train()
            loss_clr = clr_loss(self,clean_model,idx_train,idx_attach,features,edge_index)
            loss_bkd = bkd_loss(self,idx_attach,idx_train,poison_x,poison_edge_index,poison_labels[idx_attach],self.args.top_k)
            loss = loss_clr + self.args.atk_weights * loss_bkd
            loss.backward()
            optimizer.step()
       