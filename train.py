import argparse
import torch
import numpy as np
from torch_geometric.datasets import Planetoid,Amazon,Flickr,FacebookPagePage,EllipticBitcoinDataset
import torch_geometric.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=104, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN','GRACE'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv','Amazon-Computer','Facebook','Bitcoin','Citeseer'])
parser.add_argument('--train_lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
parser.add_argument('--num_attach', type=int,  default=10, help='The number of attach nodes')
parser.add_argument('--prompt_size', type=int,  default=5, help='The number of nodes for one prompt')
parser.add_argument('--homo_loss_weight', type=float, default=100.0,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                    help="Threshold of increase similarity")
parser.add_argument('--device_id', type=int, default=1,
                    help="Threshold of prunning edges")
parser.add_argument('--selection_method', type=str, default='cluster',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--dis_weight', type=float, default=0.0,
                    help="Weight of cluster distance")

parser.add_argument('--finetune', type=bool, default=True,
                    help="Fine tune torjan feat similarity")

parser.add_argument('--norm_weight', type=float, default=0.0001,
                    help="regular fine tune features")
parser.add_argument('--max_subgraph', type=int, default=3,
                    help="the number of subgraph for one node ")
parser.add_argument('--num_prompts', type=int, default=200,
                    help="the total number of prompts")
parser.add_argument('--layer', type=int, default=3,
                    help="the layer of GCN")
parser.add_argument('--mode', type=bool, default=False,
                    help="flase --oldUIAP ture --newUIAP")
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none','reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.0,
                    help="Threshold of prunning edges")
parser.add_argument('--rec_epochs', type=int,  default=300, help='Number of epochs to train benign and backdoor model.')

parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# parser.add_argument('--exp', type=int, default=8)
parser.add_argument('--new', type=str, default="Y")
parser.add_argument('--position', type=bool, default=True,
                    help="true for homo-choose.False for choose-homo")
parser.add_argument('--index', type=int, default=1,
                    help="version of pth")

parser.add_argument('--train', type=bool, default=True,
                    help="train mode or test mode")
parser.add_argument('--fit_attach_num', type=int, default=80,
                    help="val/test fit attach node num")
parser.add_argument('--test_thr', type=float, default=0.2,
                    help="trojan thr for connection")
parser.add_argument('--val_feq', type=int, default=50,
                    help="val frequence")


args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from utils import get_split,get_index,get_split_k_shot
from torch_geometric.utils import to_undirected

device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 
elif(args.dataset =='Amazon-Computer'):
    dataset = Amazon(root='./data/Amazon/Computer', \
                    name='Computers',\
                    transform=transform)
elif(args.dataset =='Amazon-Photo'):
    dataset = Amazon(root='./data/Amazon/Photo', \
                    name='Photo',\
                    transform=transform)
elif(args.dataset =='Facebook'):
    dataset = FacebookPagePage(root='./data/Facebook', \
                    transform=transform)
elif(args.dataset =='Bitcoin'):
    dataset = EllipticBitcoinDataset(root='./data/Bitcoin', \
                    transform=transform)
    
data = dataset[0].to(device)
if args.dataset == 'ogbn-arxiv':
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif args.dataset == 'Amazon-Computer' or args.dataset =='Amazon-Photo'or args.dataset == 'Bitcoin' or args.dataset == 'Facebook':
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    

data.edge_index = to_undirected(data.edge_index)





data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device)
train_edge_index,edge_mask= get_index(data.edge_index,torch.bitwise_not(data.test_mask))

mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
features = data.x.clone()
total_nodes = features.shape[0]
edge_index = data.edge_index.clone()
labels = data.y.clone()

from clustering_nodes.cluster_graph import select_attach_nodes,cluster_degree_features_nodes,cluster_degree_selection,cluster_degree_selection_seperate_fixed


unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)&torch.bitwise_not(data.val_mask)).nonzero().flatten()
if args.selection_method == 'cluster':
    
    idx_attach = cluster_degree_features_nodes(args,data,idx_train,idx_eval,idx_clean_test,unlabeled_idx,train_edge_index,args.num_attach,device)
    idx_attach =  idx_attach.to(device)
elif args.selection_method =='none':
    idx_attach = select_attach_nodes(data,args,unlabeled_idx,args.num_attach)
elif args.selection_method =='cluster_degree':
    if(args.dataset == 'Pubmed'):
        idx_attach = cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_eval,idx_clean_test,unlabeled_idx,train_edge_index,args.num_attach,device)
        if len(idx_attach) >=args.num_attach:
            idx_attach = idx_attach[:args.num_attach]
            
    else:
        idx_attach = cluster_degree_selection(args,data,idx_train,idx_eval,idx_clean_test,unlabeled_idx,train_edge_index,args.num_attach,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
    
        
# idx_attach = select_attach_nodes(data,args,unlabeled_idx,args.num_attach)
unlabeled_idx =  torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)

from eumc import EUMC 

print(args,flush=True)
print(idx_attach,flush=True)
print(unlabeled_idx,flush=True)
print('train size:',len(idx_train)/data.num_nodes,flush=True)
print('eval size:',len(idx_eval)/data.num_nodes,flush=True)
print('test size:',len(idx_clean_test)/data.num_nodes,flush=True)
# nodes_idx =  label_num(unlabeled_idx,labels)
# idx_clean = random_select(nodes_idx,top_k = args.top_k).to(device)
from collections import Counter
train_label_counts = Counter([ label.item() for label in labels[idx_train]])
eval_label_counts = Counter([ label.item() for label in labels[idx_eval]])
clean_test_counts = Counter([ label.item() for label in labels[idx_clean_test]])
atk_label_counts = Counter([ label.item() for label in labels[idx_atk]])

print('train size:',train_label_counts,flush=True)
print('eval size:',eval_label_counts,flush=True)
print('cla test size:',clean_test_counts,flush=True)
print('atk test size:',atk_label_counts,flush=True)

model = EUMC(args,device,data,idx_attach)
model.fit(features, edge_index, None, labels, idx_train, idx_eval,idx_attach,unlabeled_idx,idx_atk,idx_clean_test,mask_edge_index)
print('==train over==')
