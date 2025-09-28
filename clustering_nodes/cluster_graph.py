# 负责对图进行聚类，提取特征与度的中心
import torch
import numpy as np
from sklearn.cluster import KMeans
from surrogate_models.GraphEncoder import GCN_Encoder
from sklearn.cluster import KMeans
from sklearn_extra import cluster
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin_min
def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def obtain_attach_nodes_by_cluster(args,y_pred,model,node_idxs,x,labels,device,size):
    dis_weight = args.dis_weight
    cluster_centers = model.cluster_centers_
   
    distances = [] 
    distances_tar = []
    for id in tqdm(range(x.shape[0])):
        tmp_center_label = y_pred[id]
        tmp_tar_label = args.target_class
        
        tmp_center_x = cluster_centers[tmp_center_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        distances.append(dis)
        distances_tar.append(dis_tar)
        
    print('distance ok')
    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        labels_dict[i] = np.where(y_pred==i)[0]
        # filter out labeled nodes
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    print('lables')
    each_selected_num = int(size/len(label_list)-1)
    last_seleced_num = size - each_selected_num*(len(label_list)-2)
    candidate_nodes = np.array([])
    for label in label_list:
        if(label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]    # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes))).astype(int)

        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)
        # the closer to the center, the more far away from the target centers
        single_labels_dis_score = dis_weight * single_labels_nodes_dis + (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(single_labels_dis_score) # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if(label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    print('over')
    return candidate_nodes

def select_attach_nodes(data,args,node_idxs,num_attach):
    
    size = min(len(node_idxs),num_attach)
    rs = np.random.RandomState(args.seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice)
    return node_idxs[choice[:size]]
    # permuted_indices = torch.randperm(idx_train.size(0))
    
    # # 选择前num_attach个索引
    # selected_indices = permuted_indices[:num_attach]
    
    # # 获取这些索引对应的节点索引
    # attach_nodes = idx_train[selected_indices]
    
    # return attach_nodes
    
# 根据度和特征编码进行聚类
def cluster_degree_features_nodes(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    # 假设 data.x 是节点特征，data.edge_index 是边索引
    in_channels = data.x.shape[1]
    encoder = GCN_Encoder(in_channels, args.hidden,data.y.max().item()+1,device=device).to(device)
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    # 训练编码器
    encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    encoder_clean_test_ca = encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    
    
    # 获得节点编码进行聚类
    encoder_x = encoder.get_h(data.x, train_edge_index,None).clone().detach()
    encoder_output = encoder(data.x,train_edge_index,None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    print(y_pred.shape)
    kmeans = KMeans(n_clusters=nclass, random_state=0)
    kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    
    # kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    # kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    
    idx_attach = obtain_attach_nodes_by_cluster(args,y_pred,kmeans,unlabeled_idx.cpu().tolist(),encoder_x,data.y,device,size).astype(int)
    return torch.tensor(idx_attach)

def remap_edges(node_indices, edge_list):
    """ 重映射边索引 """
    mapping = {node: i for i, node in enumerate(node_indices)}
    return [(mapping[src], mapping[dst]) for src, dst in edge_list if src in mapping and dst in mapping]


def edge_tranform(args,edges):
    adj_matrix = np.zeros((args.prompt_size, args.prompt_size))
    std = edges[0,:]
    dst = edges[1,:]
   
    for i in range(len(std)):
        adj_matrix[std[i],dst[i]] = 1
        adj_matrix[dst[i],std[i]] = 1
    
    
    return adj_matrix

from torch_geometric.utils import degree

def cluster_subgraph(gcn_model,feature,subgraph_edges,subgraph_nodes,num,args,device):
    subgraph_embeddings = []
    subgraph_transform_edges = []
    # in_channels = data.x.shape[1]
    # gcn_model = GCN_Encoder(in_channels, args.hidden,data.y.max().item()+1,device=device).to(device)
    for nodes, edges in tqdm(zip(subgraph_nodes, subgraph_edges),total=len(subgraph_nodes), desc="Embedding Subgraphs"):
        # 提取特征
        sub_x = feature[list(nodes)].to(device)
        # 重映射边索引
        sub_edge_index = torch.tensor(remap_edges(nodes, edges), dtype=torch.long).t().contiguous().to(device)
        
        # 输入 GCN
        embedding = gcn_model.get_h(sub_x, sub_edge_index,None).clone().detach()
        subgraph_embeddings.append(embedding)
        subgraph_transform_edges.append(sub_edge_index)
        
    
    subgraph_embeddings = [embedding.view(-1) for embedding in subgraph_embeddings]
    subgraph_embeddings = torch.stack(subgraph_embeddings)
    # subgraph_transform_edges = torch.stack(subgraph_transform_edges)
    # 将合并后的张量从 CUDA 移到 CPU，并转换为 NumPy 数组
    subgraph_embeddings = subgraph_embeddings.cpu().numpy()
    
    print('=== Clustering Subgraphs ===')
    kmeans = KMeans(n_clusters=num, n_init='auto',random_state=42)
    cluster_labels = kmeans.fit_predict(subgraph_embeddings)
    print('=== end ===')
    centers = kmeans.cluster_centers_ 
    # 对于每个聚类，找到与中心点最相似的嵌入
    closest_embeddings = np.zeros_like(centers)  # 初始化数组存储最接近的嵌入
    prompt_edges = np.zeros((args.num_prompts,args.prompt_size,args.prompt_size))
    index_list = []
    print('=== Selecting Center Subgraphs ===')
    for i, center in enumerate(centers):
        # 计算每个嵌入与当前中心的距离
        distances = np.linalg.norm(subgraph_embeddings - center, axis=1)
        # 找到最小距离的索引
        closest_index = np.argmin(distances)
        # 存储最接近中心的嵌入
        index_list.append(closest_index)
        closest_embeddings[i] = subgraph_embeddings[closest_index]
        edge = edge_tranform(args,subgraph_transform_edges[closest_index])
        prompt_edges[i] = edge
    print('=== end ===')
    
    
    center_embedding = closest_embeddings.reshape(closest_embeddings.shape[0],args.prompt_size,-1)
    center_embedding = torch.tensor(center_embedding).to(device)
    prompt_edges = torch.tensor(prompt_edges).to(device)
    
    center_subgraphs = gcn_model.decode(center_embedding)
    
    
        
    
    return center_subgraphs,prompt_edges,index_list

def obtain_attach_nodes_by_cluster_degree_all(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
    distances = [] 
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    print(y_pred)
    nontarget_nodes = np.where(y_pred!=args.target_class)[0]
    non_target_node_idxs = np.array(list(set(node_idxs)))
    # non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    node_idxs = np.array(non_target_node_idxs)
    candiadate_distances = distances[node_idxs]
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    return selected_nodes

def obtain_attach_nodes_by_cluster_degree_single(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
    distances = [] 
  
    for i in range(node_idxs.shape[0]):
        id = node_idxs[i]
        tmp_center_label = y_pred[i]
        tmp_center_x = cluster_centers[tmp_center_label]
        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)
    distances = np.array(distances)
    print("y_pred",y_pred)
    print("node_idxs",node_idxs)

    candiadate_distances = distances
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    print("selected_nodes",sorted_node_idex,selected_nodes)
    return selected_nodes

def cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    in_channels = data.x.shape[1]
    gcn_encoder = GCN_Encoder(in_channels, args.hidden,data.y.max().item()+1,device=device).to(device)
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training encoder Finished!")

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()
    if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
        kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
        kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=nclass,random_state=1)
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())

    idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    idx_attach = idx_attach[:size]
    return idx_attach

def cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    in_channels = data.x.shape[1]
    gcn_encoder = GCN_Encoder(in_channels, args.hidden,data.y.max().item()+1,device=device).to(device)
    
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training encoder Finished!")
    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()

    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size/(nclass-1))
    idx_attach = np.array([])
    for label in range(nclass):
        # if(label == args.target_class):
        #     continue
        if(label != nclass-1):
            sing_class_size = each_class_size
        else:
            last_class_size= size - len(idx_attach)
            sing_class_size = last_class_size
        idx_sing_class = (y_pred == label).nonzero()[0]
        print("idx_sing_class",idx_sing_class)
        if(len(idx_sing_class) == 0):
            continue
        
        kmedoids = KMeans(n_clusters=2,random_state=1)
        kmedoids.fit(encoder_x[idx_sing_class].detach().cpu().numpy())
        sing_center = kmedoids.cluster_centers_
        cluster_ids_x = kmedoids.predict(encoder_x[idx_sing_class].cpu().numpy())
        cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist())&set(idx_sing_class)))
        if(label != nclass - 1):
            sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,each_class_size).astype(int)
            sing_idx_attach = sing_idx_attach[:each_class_size]
        else:
            last_class_size= size - len(idx_attach)
            sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,last_class_size).astype(int)
            sing_idx_attach = sing_idx_attach[:each_class_size]
        idx_attach = np.concatenate((idx_attach,sing_idx_attach))

    return idx_attach
