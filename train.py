import argparse
import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.datasets import Planetoid,Amazon,Flickr,FacebookPagePage,EllipticBitcoinDataset
import torch_geometric.transforms as T
from utils import calculate_accuracy,calculate_similarity,count_removed_edges
from surrogate_models.GCN import GCN
import os
import csv

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
parser.add_argument('--exp', type=int, default=7)
parser.add_argument('--new', type=str, default="Y")
parser.add_argument('--position', type=bool, default=False,
                    help="true for homo-choose.False for choose-homo")
parser.add_argument('--index', type=int, default=1,
                    help="version of pth")

parser.add_argument('--train', type=bool, default=False,
                    help="train mode or test mode")
parser.add_argument('--fit_attach_num', type=int, default=80,
                    help="val/test fit attach node num")
parser.add_argument('--moreFit', type=bool, default=False,
                    help="val/test fit one time or more")
parser.add_argument('--test_thr', type=float, default=0.2,
                    help="trojan thr for connection")
parser.add_argument('--val_feq', type=int, default=50,
                    help="val frequence")
parser.add_argument('--p', type=float, default=0.3,
                    help='parameter of gcl loss')
parser.add_argument('--k', type=int, default=20,
                    help='parameter of epoch ood')
parser.add_argument('--range', type=float, default=0.5,
                    help="xxx")
parser.add_argument('--weight_ood', type=float, default=1.0,
                    help="xxx")
parser.add_argument('--clr_weight', type=float, default=0.01,
                    help="xxx")
parser.add_argument('--top_k', type=int, default=5,
                    help="xxx")
parser.add_argument('--class_clr_weight', type=float, default=3.0,
                    help="xxx")
parser.add_argument('--Init', type=bool, default=False,
                    help="xxx")
parser.add_argument('--attack_num', type=int, default=7,
                    help="xxx")
parser.add_argument('--attack_single', type=bool, default=False,
                    help="xxx")
parser.add_argument('--target_label', type=int, default=0,
                    help="xxx")
parser.add_argument('--drop_edge_rate_1', type=float, default=0.4,
                    help="xxx")
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                    help="xxx")
parser.add_argument('--drop_feature_rate_1', type=float, default=0.0,
                    help="xxx")
parser.add_argument('--drop_feature_rate_2', type=float, default=0.2,
                    help="xxx")
parser.add_argument('--tau', type=float, default=0.4,
                    help="xxx")
parser.add_argument('--prompt_type', type=str, default="GPPT",
                    choices=['GPPT', 'GPF', 'All-in-one','Gprompt','none'],
                    help="prompt_type")
parser.add_argument('--lr_method', type=str, default="GSL",
                    choices=['GPL', 'GCL', 'GSL'],
                    help="prompt_type")
parser.add_argument('--gcl_hidden', type=int, default=128,
                    help="prompt_type")
parser.add_argument('--pre_train_model_path', type=str, default='ProG/pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth',
                    help="xxx")
parser.add_argument('--down_method', type=str, default="GSL",
                    choices=['GPL', 'GCL', 'GSL'],
                    help="prompt_type")

parser.add_argument('--atk_weights', type=float, default=1.0,
                    help="xxx")
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



if args.lr_method == 'GCL':

    if args.dataset == 'ogbn-arxiv':
        data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device,0.45,0.005,0.05)
    elif args.dataset == 'Bitcoin':
        data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device,0.2)
    else:
        data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device,0.45)
    
else:
    if args.dataset == 'ogbn-arxiv':
        data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device,0.2,0.005,0.05)
    # elif args.dataset == 'Facebook':
    # #     data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device,0.45)
        # data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split_k_shot(data,args.seed,device,k_shot = 1000)
    elif args.dataset == 'Bitcoin':
        data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split_k_shot(data,args.seed,device,k_shot = 3000,eval_radio=0.02, test_radio=0.05)
        # data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device,0.01,0.005,0.01)
    else:
        data,idx_train,idx_eval,idx_clean_test,idx_atk = get_split(data,args.seed,device)
train_edge_index,edge_mask= get_index(data.edge_index,torch.bitwise_not(data.test_mask))

mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
features = data.x.clone()
total_nodes = features.shape[0]
edge_index = data.edge_index.clone()
labels = data.y.clone()
#聚类选择的节点,训练集
from clustering_nodes.cluster_graph import select_attach_nodes,cluster_degree_features_nodes,cluster_degree_selection,cluster_degree_selection_seperate_fixed

# 编码器进行聚类
from surrogate_models.GraphEncoder import GCN_Encoder

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

from prompts.UIAP import UIAP
from prompts.prompt import UIAP as oldUIAP
from prompts.gcl_new_prompt import UIAP as gclUIAP
from prompts.gpl_eumc import UIAP as gplUIAP
from prompts.time_prompt import UIAP as timeEMUC
from utils import prune_unrelated_edge,label_num,random_select
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
Iftrain = True
if Iftrain:
    if args.mode:
        model = UIAP(args,device,features,edge_index,labels)
        model.init_promptPool(idx_attach,idx_train,idx_eval)
        model.fit(features, edge_index, None, labels, idx_train, idx_attach,unlabeled_idx)
        print('==train over==')
    #     poison_x, poison_edge_index,poison_labels,_,_= model.get_attach_poisoned()
    #     if(args.defense_mode == 'prune'):
    #         poison_edge_index,_ = prune_unrelated_edge(args,poison_edge_index,None,poison_x,device,large_graph=False)
    else:
        if args.lr_method == 'GSL':
            print('GSL',flush=True)
            model = oldUIAP(args,device,data,idx_attach)
        elif args.lr_method == 'GCL':
            print('GCL',flush=True)
            model = gclUIAP(args,device,data,idx_attach)
        # model = timeEMUC(args,device,data,idx_attach)
        elif args.lr_method == 'GPL':
            print('GPL',flush=True)
            model = gplUIAP(args,device,data,idx_attach)
        model.fit(features, edge_index, None, labels, idx_train, idx_eval,idx_attach,unlabeled_idx,idx_atk,idx_clean_test,mask_edge_index)
        print('==train over==')
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= model.get_attach_poisoned()
        # # poison_x, poison_edge_index,poison_edge_weights,poison_labels = features,edge_index,torch.ones_like(poison_edge_weights.shape[1]).to(device),labels
        
        # if(args.defense_mode == 'prune'):
        #     poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=True)
else:
    print('==test begin==')
    if args.mode:
        model = UIAP(args,device,features,edge_index,labels)
        model.load_state_dict(torch.load('./parameters/{}.pth'.format(args.dataset))) 
        poison_x, poison_edge_index,poison_labels,_,_= model.get_attach_poisoned()
        if(args.defense_mode == 'prune'):
            poison_edge_index,_ = prune_unrelated_edge(args,poison_edge_index,None,poison_x,device,large_graph=False)
    else:
        model = oldUIAP(args,device)
        print(model.state_dict().keys())
        model.load_state_dict(torch.load('./parameters/{}.pth'.format(args.dataset))) 
        poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= model.get_attach_poisoned()
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels = features,edge_index,torch.ones_like(poison_edge_weights.shape[1]).to(device),labels
        if(args.defense_mode == 'prune'):
            poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=True)


    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)

    from construct import model_construct
    edge_index = data.edge_index
    models = ['GCN','GAT', 'GraphSage']
    # models = ['GraphSage']
    total_overall_asr = 0
    total_overall_ca = 0
    for test_model in models:
        args.test_model = test_model
        rs = np.random.RandomState(args.seed)
        seeds = rs.randint(1000,size=5)
        # seeds = [args.seed]
        overall_asr = 0
        overall_ca = 0
        for seed in seeds:
            args.seed = seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            print(args)
            test_model = model_construct(args,args.test_model,data,device).to(device) 
            if args.mode:
                test_model.fit(poison_x,poison_edge_index,None,poison_labels, bkd_tn_nodes, idx_eval,args.epochs,verbose =False)
                output = test_model(poison_x,poison_edge_index,None)
            else:
                test_model.fit(poison_x,poison_edge_index,poison_edge_weights,poison_labels, bkd_tn_nodes, idx_eval,args.epochs,verbose =False)
                output = test_model(poison_x,poison_edge_index,poison_edge_weights)
            # train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
            # print("target class rate on Vs: {:.4f}".format(train_attach_rate))
            
            
            induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
            induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
            # poison_edge_index,poison_x,poison_edge_weights= model.test(idx_atk,idx_attach,features,edge_index)
            clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

            print("accuracy on clean test nodes: {:.4f}".format(clean_acc))

            total_normal_edges_count = 0
            total_special_edges_count = 0
            if(args.evaluate_mode == '1by1'):
                from torch_geometric.utils  import k_hop_subgraph
                overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
                asr = 0
                # flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
                for i, idx in enumerate(idx_atk):
                    idx=int(idx)
                    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                    
                    ori_node_idx = sub_induct_nodeset[sub_mapping]
                    relabeled_node_idx = sub_mapping
                    sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                    asr_all_target = 0
                    for index in range(labels.max().item()+1):
                        
                        with torch.no_grad():
                        
                            induct_x, induct_edge_index,induct_edge_weights,_,trojan_edge_index,target_class = model.get_attach_poision(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device,index)
                        
                            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                        
                            # # do pruning in test datas'''
                            if args.mode:
                                if(args.defense_mode == 'prune'):
                                    induct_edge_index,_ = prune_unrelated_edge(args,induct_edge_index,None,induct_x,device)
                                output = test_model(induct_x,induct_edge_index,None)
                            else:
                                if(args.defense_mode == 'prune'):
                                    
                                    origin_edge_index = induct_edge_index
                                
                                    
                                    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
                                    normal_edges_count,special_edges_count = count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index)
                                    # count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index, origin_edge_index)
                                    # normal_edges_count,special_edges_count = count_edges(old_edge_index,induct_edge_index,total_nodes)
                                    total_special_edges_count += special_edges_count
                                    total_normal_edges_count += normal_edges_count
                                    
                                
                                output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                                
                            # train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                            train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==target_class).float().mean()
                            asr_all_target +=train_attach_rate
                            induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                            output = output.cpu()
                    asr_all_target = asr_all_target/(labels.max().item()+1)
                    asr += asr_all_target
                asr = asr/(idx_atk.shape[0])
                # flip_asr = flip_asr/(flip_idx_atk.shape[0])
                print('Pruning Number:{}'.format(total_normal_edges_count+total_special_edges_count))
                if total_normal_edges_count+total_special_edges_count:
                    print('Pruning Ratio:{:.4f}'.format(total_special_edges_count/(total_normal_edges_count+total_special_edges_count)))
                else:
                    print('Pruning Ratio:{:.4f}'.format(0))
                print("Overall ASR: {:.4f}".format(asr))
                # print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            elif args.evaluate_mode == 'overall':
                if args.mode:
                    induct_x, induct_edge_index,_,_,_= model.get_attach_poision(idx_atk, poison_x,induct_edge_index,device)
                else:
                    induct_x, induct_edge_index,induct_edge_weights,_,_,_= model.get_attach_poision(idx_atk, poison_x,induct_edge_index,induct_edge_weights,device)
                induct_edge_weights = induct_edge_weights.clone().detach()
                induct_x, induct_edge_index= induct_x.clone().detach(), induct_edge_index.clone().detach()
                if args.mode:
                    if(args.defense_mode == 'prune'):
                        induct_edge_index,_ = prune_unrelated_edge(args,induct_edge_index,None,induct_x,device)
                    output = test_model(induct_x,induct_edge_index,None)
                else:
                    if(args.defense_mode == 'prune'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
                
                print("ASR: {:.4f}".format(train_attach_rate))
                
                asr+= train_attach_rate
            overall_asr += asr

        overall_asr = overall_asr/len(seeds)
        print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
        csv_file = 'result/{}_{}.csv'.format(args.test_model,args.dataset)

        
        os.makedirs('result', exist_ok=True)

    
        if not os.path.isfile(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['prompt_size', 'norm_position', 'num_prompts', 'layer', 'asr','defense','new'])


        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.prompt_size, args.norm_weight, args.num_prompts, args.layer, float(overall_asr),args.defense_mode,args.new])