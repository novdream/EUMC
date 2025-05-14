
import torch
import numpy as np
from tqdm import tqdm
from surrogate_models.reconstruct import MLPAE
def get_split_k_shot(data,seed,device,k_shot, eval_radio=0.1, test_radio=0.2):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(data.num_nodes)
    class_num = data.y.max().item() + 1
    y_np = data.y.cpu().numpy()
    train_indices = []
    for cls in range(class_num):
     
        class_indices = perm[y_np[perm] == cls]
        
       
        selected_indices = rs.choice(class_indices, size=k_shot, replace=False)
        
     
        train_indices.append(selected_indices)
    

    train_indices = np.concatenate(train_indices)
    remaining_perm = np.setdiff1d(perm, train_indices)

    idx_train = torch.tensor(sorted(train_indices), device=device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True
    
    
    val_number = int(eval_radio*len(perm))
    idx_val = torch.tensor(sorted(remaining_perm[:val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(test_radio*len(perm))
    idx_test = torch.tensor(sorted(remaining_perm[val_number:val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]
    
    return data, idx_train, idx_val, idx_clean_test, idx_atk


def get_split(data,seed,device,train_radio=0.2, eval_radio=0.1, test_radio=0.2):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(data.num_nodes)
    train_number = int(train_radio*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True
    val_number = int(eval_radio*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(test_radio*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]

    return data, idx_train, idx_val, idx_clean_test, idx_atk

def get_split_topk(data,device,top_k = 100,train_radio=0.2, eval_radio=0.1, test_radio=0.2):
    rs = np.random.RandomState(10)
    labels = data.y
    perm = rs.permutation(data.num_nodes)
    
    nodes_idx =  label_num(torch.tensor(perm),labels)
    idx_clean = random_select(nodes_idx,top_k).to(device)
    idx_remain = list(set(perm) - set(idx_clean.cpu().numpy()))
    
    train_number = int(train_radio*len(perm))
    idx_train = torch.tensor(sorted(idx_remain[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True
    val_number = int(eval_radio*len(perm))
    idx_val = torch.tensor(sorted(idx_remain[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(test_radio*len(perm))
    idx_test = torch.tensor(sorted(idx_remain[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]

    return data, idx_train, idx_val, idx_clean_test, idx_atk

def get_index(edge_index,node_mask):
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    filtered_edge_index = edge_index[:, edge_mask]
    return filtered_edge_index,edge_mask

import torch
import torch.nn.functional as F

def calculate_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

    probs = F.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    correct = (predictions == labels).float()
    accuracy = correct.sum() / len(labels)
    return accuracy.item() * 100 

def calculate_similarity(subgraph_features, node_feature):
    
    node_feature = node_feature.unsqueeze(0)  
    similarities = F.cosine_similarity(subgraph_features, node_feature, dim=1)
    return similarities
import random
def get_similar_prompt(device,embedding,prompt_size,pool,label_full,label_origin,targetlabel,mode):
    
    
   
    expanded_vector = embedding.unsqueeze(0).unsqueeze(0).expand_as(pool)
   
    
   
    similarities = F.cosine_similarity(pool, expanded_vector, dim=2)

    
    summed_similarities = torch.sum(similarities, dim=1)
    
  
    if mode == 'F':
        valid_indices = [i for i in range(len(summed_similarities)) if label_origin[i] == targetlabel]
        max_index = valid_indices[torch.argmax(summed_similarities[valid_indices]).item()]
    elif mode == 'R':
        valid_indices = [i for i in range(len(summed_similarities)) if label_origin[i] == targetlabel]
        max_index = random.choice(valid_indices)

    return max_index



def get_similar_prompt_tensor(device, embedding, prompt_size, pool, label_full, label_origin, targetlabel,class_num ,mode):
    targetlabel = torch.tensor(targetlabel).to(device)
    label_origin = torch.tensor(label_origin).to(device)
    
    if mode == 'R' :
        max_index_per_valid_node = torch.randint(0, pool.shape[0], (len(targetlabel),))
        return max_index_per_valid_node
    
    mean_pool = pool.mean(dim=1)
    dot_product = torch.matmul(embedding, mean_pool.T)  
    norm_embedding = embedding.norm(dim=1, keepdim=True) 
    norm_mean_pool = mean_pool.norm(dim=1, keepdim=True) 


    summed_similarities = dot_product / (norm_embedding * norm_mean_pool.T)
    summed_similarities = summed_similarities.reshape(embedding.shape[0],class_num,-1)
    target_category_simi = summed_similarities[torch.arange(embedding.shape[0]), targetlabel, :]
    max_indices_in_target_category = target_category_simi.argmax(dim=1)


    
    if mode == 'F':

        max_index_per_valid_node = max_indices_in_target_category 

    
    return max_index_per_valid_node

import torch
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
from itertools import combinations

def sample_k_subgraphs(node_idx, num_hops, edge_index, total_nodes, k):
   
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, num_hops, edge_index, num_nodes=total_nodes)

    G = nx.Graph()
    
    for i in range(sub_edge_index.size(1)):
        src, dest = sub_edge_index[:, i]
        G.add_edge(int(src), int(dest))
    print('ok')
    k_node_subgraphs = []
    for nodes in combinations(G.nodes(), k):
        sg = G.subgraph(nodes)
        if sg.number_of_nodes() == k and nx.is_connected(sg): 
            k_node_subgraphs.append(sg)

    return k_node_subgraphs

import networkx as nx
from itertools import combinations

def create_graph(edge_index):
  
    G = nx.Graph()
    if isinstance(edge_index, np.ndarray):
        edges = edge_index.transpose() 
    else:
     
        edges = edge_index.t().cpu().numpy()

    for src, dst in edges:
        G.add_edge(int(src), int(dst))
    return G

from collections import deque

def bfs_subgraphs(G, start_node, m, k):
  
    queue = deque([(start_node, [start_node])])
    visited = set()
    subgraphs = []
    visited_nodes = {start_node}
    
    while queue and len(subgraphs) < k:
        current_node, path = queue.popleft()
        if len(path) == m:
            subgraph_tuple = tuple(sorted(path))
            if subgraph_tuple not in visited:
                visited.add(subgraph_tuple)
                subgraphs.append(G.subgraph(path))
                if len(subgraphs) == k:
                    break
        elif len(path) < m:
            neighbors = G.neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    
    return subgraphs

import os
def find_k_subgraphs_per_node(args,edge_index, node_indices, m, k):
   
    G = create_graph(edge_index)
    all_subgraphs = []
    file_path = 'subgraphs/{}/{}-subgraphs.pth'.format(args.dataset,args.prompt_size)
    if os.path.exists(file_path):
        print('== loading subgraphs ==')
        unique_subgraphs = torch.load(file_path)
        print('== end ==')
    else:
        for node in tqdm(node_indices,desc='Bfs subgraphs'):
            if G.has_node(node):
                subgraphs = bfs_subgraphs(G, node, m, k)
                all_subgraphs.extend(subgraphs)
        
        unique_subgraphs = []
        
        seen = set()
        for sg in all_subgraphs:
            nodes_tuple = tuple(sorted(sg.nodes()))
            if nodes_tuple not in seen:
                seen.add(nodes_tuple)
                unique_subgraphs.append(sg)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(unique_subgraphs,file_path)
    return unique_subgraphs



def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    # update structure
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    return updated_edge_index,updated_edge_weights

def adj_to_edges_batch(adj_matrix):
  
    t, m, _ = adj_matrix.size()
    edge_list = []
    
    for i in range(t):
        edges = adj_to_edges(adj_matrix[i])
        edge_list.append(edges)
    
    return edge_list

def adj_to_edges(adj_matrix):
   
    m = adj_matrix.size(0)
    
    
    if adj_matrix.size(0) != adj_matrix.size(1):
        raise ValueError("error")
    
    
    row_indices, col_indices = torch.triu_indices(m, m, offset=1)
    

    mask = adj_matrix[row_indices, col_indices] != 0
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    
    
    edges = torch.cat([row_indices, col_indices, col_indices, row_indices]).reshape(2, -1)
    # edges = torch.stack([row_indices, col_indices], dim=0)
    
    return edges

def model_construct(args,data,device):
    pass
def count_edges(edge_index, new_edge_index, m):
   
    edge_set = set(map(tuple, edge_index.T))
    new_edge_set = set(map(tuple, new_edge_index.T))
    removed_edges = edge_set - new_edge_set

    normal_edges_count = 0
    special_edges_count = 0

    for edge in removed_edges:
        if edge[0] >=m or edge[1] >=m:
            special_edges_count += 1
        else:
            normal_edges_count += 1

    return normal_edges_count, special_edges_count

def count_removed_edges(edge_index, new_edge_index, trojan_edge_index):
   
    edge_set = set(map(tuple, edge_index.T.tolist()))
    new_edge_set = set(map(tuple, new_edge_index.T.tolist()))

    removed_edges = edge_set - new_edge_set

    trojan_edge_set = set(map(tuple, trojan_edge_index.T.tolist()))
 
    trojan_removed_count = 0
    origin_removed_count = 0

    for edge in removed_edges:
        if edge in trojan_edge_set:
            trojan_removed_count += 1
        else:
            origin_removed_count += 1

    return origin_removed_count, trojan_removed_count

from torch_geometric.data import Data
def merge_graphs(device,orgin_graph, attack_graph, center_node_idx):
   
    num_nodes_origin = orgin_graph.x.size(0)
    
    
    updated_edge_index = attack_graph.edge_index + num_nodes_origin
    
   
    connection_edges = torch.tensor([[center_node_idx] * attack_graph.x.size(0), range(num_nodes_origin, num_nodes_origin + attack_graph.x.size(0))], dtype=torch.long)
    connection_edges = torch.cat([connection_edges, connection_edges.flip(0)], dim=1).to(device)  
    
    new_edge_index = torch.cat([orgin_graph.edge_index, updated_edge_index, connection_edges], dim=1)
    
    
    new_x = torch.cat([orgin_graph.x, attack_graph.x], dim=0)
    
    
    new_data = Data(x=new_x, edge_index=new_edge_index)
    
    return new_data
import random
def getTargetCLass(idx_attach,label_full,label_connect,orginLabels,index=None):
    target_classes = []
    # print(orginLabels.max().item())
    common_labels = set(label_full).intersection(set(label_connect))

    # 将 common_labels 
    common_labels = list(common_labels)
    # 
    if index is not None:
        target_classes = torch.tensor([index]*len(idx_attach))
    else:
        for idx in idx_attach:
            label = orginLabels[idx].item()
            # possible_classes = list(range(orginLabels.max().item()+1))
            possible_classes = [cls for cls in common_labels if cls != label]
        
            # possible_classes.remove(label)
            # attack_class = random.choice(possible_classes)
            # attack_class = random.choice([0,1])
            attack_class = 0
            # target_classes.append(0)
            target_classes.append(attack_class)

        target_classes = torch.tensor(target_classes)
    return target_classes

def getRandomClass(idx_attach,labels_num,label_list=None):
    
  

    target_classes = []
    if label_list == None:
        labels = list(range(labels_num))
    else:
        labels = label_list
   
    
    for i,idx in enumerate(idx_attach):
       
        attack_class = random.choice(labels)
        target_classes.append(attack_class)
    target_classes = torch.tensor(target_classes)
    return target_classes

def getSelfClass(idx_attach,labels):
    return labels[idx_attach]
    
def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        # calculate edge simlarity
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    edge_weights[dissim_edges_index] = 0
    # select the nodes between dissimilar edgesy
    dissim_edges = edge_index[:,dissim_edges_index]    # output: [[v_1,v_2],[u_1,u_2]]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))
    # update structure
    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    return updated_edge_index,updated_edge_weights,dissim_nodes 

def calculate_weights(idx_train,labels):
    atk_labels = [0]*(labels.max().item()+1)
    for item in idx_train:
        atk_labels[labels[item]]+=1
    times = sum(atk_labels)
    atk_labels = [ round(1-x/times,1) for x in atk_labels]
    return atk_labels

import matplotlib.pyplot as plt
import matplotlib as mpl   
# mpl.rcParams['font.family'] = 'Times New Roman'

def draw_pic(rec_score_ori,rec_score_triggers,name):
    rec_score_ori =rec_score_ori * 1e5
    rec_score_triggers = rec_score_triggers *1e5
    import matplotlib.pyplot as plt
    import matplotlib as mpl   
    # mpl.rcParams['font.family'] = 'Times New Roman'


    rec_score_ori_np = rec_score_ori.cpu().detach().numpy()
    rec_score_triggers_np = rec_score_triggers.cpu().detach().numpy()

    plt.figure(figsize=(10, 5))
    # mpl.rcParams['font.size'] = 36

    bins = np.arange(0, 3.5, 0.05)
    x_tack = np.arange(0, 3.5, 0.5)
    counts, bin_edges = np.histogram(rec_score_ori_np, bins=bins)  
    counts_new, bin_edges = np.histogram(rec_score_triggers_np, bins=bins)  
    #   
    total_elements = len(rec_score_ori_np)  
    total_elements_new = len(rec_score_triggers_np)
    #   
    percentages = (counts / (total_elements)) * 100  
    percentages_new =  (counts_new / (total_elements)) * 100
    #   
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

    #   
    plt.bar(bin_centers, percentages, width=np.diff(bin_edges).mean(), alpha=0.5, color='#aed6f1', edgecolor='#5cade2', label='Clean Edges')  

    plt.bar(bin_centers, percentages_new, width=np.diff(bin_edges).mean(), alpha=0.9, color='#f5cba7', edgecolor='#eb984e', label='Trigger Edges')  
    plt.legend(fontsize=30,loc='upper left')  

    #   
    # plt.ylabel('Percentage (%)')  
    plt.xlabel('Cosine Similarity') 
    plt.ylabel('Frequency')  
    # plt.yticks(np.arange(0,21,3))  #  
    from matplotlib.ticker import FuncFormatter 
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))  # 设置刻度格式  
    plt.xticks(x_tack)  #   


    plt.savefig(name)
    print('draw over',flush=True)
def reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,ori_x,ori_edge_index,device, idx, large_graph=True):
    poison_x = poison_x.to(device)
    AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE.fit()
    rec_score_ori = AE.inference(poison_x)
    # print(torch.mean(rec_score_ori))
    rec_score_triggers = AE.inference(poison_x[len(ori_x):])
    # print(rec_score)
   
    # print(torch.mean(rec_score_triggers))
    poison = rec_score_ori[len(ori_x):].detach().cpu().numpy()
    # Calculate the threshold for the top 3% largest values in rec_score_ori
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 97)
    mask = rec_score_ori>threshold
    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
    # Filter the edge_index by the edges we want to keep
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    # Filter the edge weights similarly
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]
    # Check each element in poison against this threshold
    top_3_percent_flag = poison >= threshold
    # Calculate the percentage of poison elements that are in the top 3%
    percentage_in_top_3 = np.mean(top_3_percent_flag) * 100  # Convert to percentage
    print('Percentage of Triggers in Top3 Reconstruction Loss:',percentage_in_top_3)
    
    if args.dataset == 'Cora' or args.dataset == 'Pubmed':
        path = '{}_loss.pth'.format(args.dataset)
        if os.path.exists(path):
            pass
        else:
            torch.save([rec_score_ori,rec_score_triggers],path)
            draw_pic(rec_score_ori,rec_score_triggers,'od_loss.png')
            
            
    return filtered_poison_edge_index,filtered_poison_edge_weights

import torch.nn.functional as F
import utils
import torch.optim as optim
from copy import deepcopy
from surrogate_models import MLP
def  train_classifer_without_val(model,ntk_nodes,idx_val,features,edge_index,labels,device,args,train_iters=1000, verbose=False):
    classifer = torch.nn.Linear(args.hidden,labels.max().item() + 1).to(device)
    # print('class num:{}'.format(labels.max().item() + 1),flush=True)
    optimizer = optim.Adam(classifer.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    if verbose:
        print('=== training classifer model ===')
    best_acc_val = 0
    idx_train , idx_attach = ntk_nodes
    model.eval()
    if idx_attach == None or len(idx_attach) == 0:
        idx_new = idx_train
    else:
        idx_new = torch.cat([idx_train,idx_attach])
            
    # idx_new = torch.cat([idx_train,idx_attach])
    for i in range(train_iters):
        optimizer.zero_grad()
       
        z = model(features,edge_index).detach().clone()
        output = classifer(z)
        loss_clean = F.cross_entropy(output[idx_train], labels[idx_train]) 
        if idx_attach == None or len(idx_attach) == 0:
            loss_bkd = 0.0
        else:
            loss_bkd = F.cross_entropy(output[idx_attach], labels[idx_attach]) 
        loss = args.class_clr_weight* loss_clean + loss_bkd
        acc1 = utils.calculate_accuracy(output[idx_new], labels[idx_new])
        loss.backward()
        optimizer.step()
        if verbose:
            print('train-acc:{:.4f}'.format(acc1))
        
        
    #     acc_val = utils.calculate_accuracy(output[idx_val], labels[idx_val])
    #     if acc_val > best_acc_val:
    #         best_acc_val = acc_val
    #         weights = deepcopy(classifer.state_dict())
        
    # classifer.load_state_dict(weights)

    z = model(features,edge_index).detach().clone()
    output = classifer(z)
    acc = utils.calculate_accuracy(output[idx_train], labels[idx_train])
    total_acc = utils.calculate_accuracy(output[idx_new], labels[idx_new])
    print('val-acc:{:.4f}'.format(best_acc_val),flush=True)
    print('train-clean-acc:{:.4f}'.format(acc),flush=True)
    print('train-total-acc:{:.4f}'.format(total_acc),flush=True)
    
    return classifer


def  train_classifer_cora(model,ntk_nodes,idx_val,features,edge_index,labels,device,args,train_iters=1000, verbose=False,with_val=False):
    classifer = torch.nn.Linear(args.gcl_hidden,labels.max().item() + 1).to(device)
    # print('class num:{}'.format(labels.max().item() + 1),flush=True)
    optimizer = optim.Adam(classifer.parameters(), lr=0.001, weight_decay=0.1)
    if verbose:
        print('=== training classifer model ===')
    best_acc_val = 0
    idx_train , idx_attach = ntk_nodes
    model.eval()
    if idx_attach == None or len(idx_attach) == 0:
        idx_new = idx_train
    else:
        idx_new = torch.cat([idx_train,idx_attach])

    
    loss_idx_new_labels = calculate_weights(idx_new,labels)
    loss_idx_new_labels = torch.tensor(loss_idx_new_labels).to(device)

    for i in range(train_iters):
        optimizer.zero_grad()
       
        z = model(features,edge_index).detach().clone()
        output = classifer(z)
        loss = F.cross_entropy(output[idx_new], labels[idx_new], weight = loss_idx_new_labels) 
        acc1 = utils.calculate_accuracy(output[idx_new], labels[idx_new])
        loss.backward()
        optimizer.step()
        if verbose:
            print('train-acc:{:.4f}'.format(acc1))
        
        if with_val:
            acc_val = utils.calculate_accuracy(output[idx_val], labels[idx_val])
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(classifer.state_dict())
    if with_val:
        classifer.load_state_dict(weights)

    z = model(features,edge_index).detach().clone()
    output = classifer(z)
    acc = utils.calculate_accuracy(output[idx_train], labels[idx_train])
    total_acc = utils.calculate_accuracy(output[idx_new], labels[idx_new])
    print('val-acc:{:.4f}'.format(best_acc_val),flush=True)
    print('train-clean-acc:{:.4f}'.format(acc),flush=True)
    print('train-total-acc:{:.4f}'.format(total_acc),flush=True)
    
    return classifer

def  train_classifer(model,ntk_nodes,idx_val,features,edge_index,labels,device,args,train_iters=200, verbose=False,with_val=False):
    classifer = torch.nn.Linear(args.gcl_hidden,labels.max().item() + 1).to(device)
    # print('class num:{}'.format(labels.max().item() + 1),flush=True)
    optimizer = optim.Adam(classifer.parameters(), lr=0.001, weight_decay=0.1)
    if verbose:
        print('=== training classifer model ===')
    best_acc_val = 0
    idx_train , idx_attach = ntk_nodes
    model.eval()
    if idx_attach == None or len(idx_attach) == 0:
        idx_new = idx_train
        loss_idx_attach_labels = None
    else:
        idx_new = torch.cat([idx_train,idx_attach])
        loss_idx_attach_labels = calculate_weights(idx_attach,labels)
        loss_idx_attach_labels = torch.tensor(loss_idx_attach_labels).to(device)
    
    loss_idx_train_labels = calculate_weights(idx_train,labels)
    # loss_idx_attach_labels = calculate_weights(idx_attach,labels)
    loss_idx_train_labels = torch.tensor(loss_idx_train_labels).to(device)

    for i in range(train_iters):
        optimizer.zero_grad()
       
        z = model(features,edge_index).detach().clone()
        output = classifer(z)
        loss_clean = F.cross_entropy(output[idx_train], labels[idx_train], weight = loss_idx_train_labels) 
        if idx_attach == None or len(idx_attach) == 0:
            loss_bkd = 0.0
        else:
            if args.attack_single: # TODO
                loss_bkd = F.cross_entropy(output[idx_attach], labels[idx_attach])
            else: 
                loss_bkd = F.cross_entropy(output[idx_attach], labels[idx_attach],weight = loss_idx_attach_labels) 
        loss = args.class_clr_weight* loss_clean + loss_bkd
        acc1 = utils.calculate_accuracy(output[idx_new], labels[idx_new])
        loss.backward()
        optimizer.step()
        if verbose:
            print('train-acc:{:.4f}'.format(acc1))
        
        if with_val:
            acc_val = utils.calculate_accuracy(output[idx_val], labels[idx_val])
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(classifer.state_dict())
    if with_val:
        classifer.load_state_dict(weights)

    z = model(features,edge_index).detach().clone()
    output = classifer(z)
    acc = utils.calculate_accuracy(output[idx_train], labels[idx_train])
    total_acc = utils.calculate_accuracy(output[idx_new], labels[idx_new])
    print('val-acc:{:.4f}'.format(best_acc_val),flush=True)
    print('train-clean-acc:{:.4f}'.format(acc),flush=True)
    print('train-total-acc:{:.4f}'.format(total_acc),flush=True)
    
    return classifer

def label_num(idx,labels):
    class_num = labels.max().item() + 1
    selected_labels = labels[idx]
    
    # 
    label_idx_list = [[] for _ in range(class_num)]
    
    #  selected_labels  idx 
    for label in range(class_num):

        mask = (selected_labels == label)  
        idx_for_label = idx[mask]  
        
    
        label_idx_list[label] = idx_for_label.tolist()

    return label_idx_list

def random_select(idx_list,top_k=5):
    num_classes = len(idx_list)
    

    selected_indices = []
    torch.manual_seed(106) 
    for i in range(num_classes):
      
        current_idx = torch.tensor(idx_list[i])
        
      
        shuffled_idx = current_idx[torch.randperm(current_idx.size(0))]
        

        selected_indices.append(shuffled_idx[:top_k])
    
    result = torch.stack(selected_indices)
    result = result.view(-1)
    return result


def check_model_params_unchanged(model, initial_params=None):
    

    if initial_params is None:
        initial_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    

    unchanged = True
    for name, param in model.named_parameters():
        if not torch.allclose(param.detach(), initial_params[name], atol=1e-5):
            print(f"Parameter '{name}' has changed!")
            unchanged = False
    
    return unchanged

from collections import defaultdict
def random_sample_per_class(idx, label, n, seed):
    
    rs = np.random.RandomState(seed)
    label = label.cpu().numpy()
    idx = idx.cpu().numpy()
    
    class_dict = defaultdict(list)
    for i, lbl in zip(idx, label):
        class_dict[lbl].append(i)
    

    selected_indices = []
    for class_idx, indices in class_dict.items():
        selected_indices.extend(rs.choice(indices, size=min(n, len(indices)), replace=False))
    
    return selected_indices

    
        
def center_embedding(input,index, label_num):
    device=input.device

    
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)
    c /= class_counts
    
    return c