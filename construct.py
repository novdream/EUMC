# %%
from surrogate_models.GCN import GCN
from surrogate_models.GAT import GAT
from surrogate_models.SAGE import GraphSage
from surrogate_models.GNNGuard import GNNGuard
from surrogate_models.RobustGCN import RobustGCN
from surrogate_models.GRACE import GRACE,Encoder

def model_construct(args,model_name,data,device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    if (model_name == 'GCN'):

        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device,
                    use_ln=use_ln,
                    layer_norm_first=layer_norm_first)
        
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(data.y.max()+1), 
                    heads=8,
                    dropout=args.dropout, 
                    lr=args.train_lr, 
                    weight_decay=args.weight_decay, 
                    device=device)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.train_lr,\
                weight_decay=args.weight_decay,\
                device=device)
    elif(model_name == 'GNNGuard'):
        model = GNNGuard(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    use_ln=use_ln,\
                    device=device)
    elif(model_name == 'RobustGCN'):
        model = RobustGCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device)
    elif model_name == 'GRACE':
        encoder = Encoder(data.x.shape[1], args.hidden,k=args.layer).to(device)
        model = GRACE(encoder = encoder,
                        nfeat=data.x.shape[1],
                        nhid=args.hidden,
                        nclass=int(data.y.max()+1),
                        layer = args.layer,
                        dropout=0.5, device=device,args = args)
    else:
        print("Not implement {}".format(model_name))
    return model
