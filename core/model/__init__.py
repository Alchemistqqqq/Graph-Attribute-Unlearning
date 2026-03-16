import torch
from core.model.gat import GraphAttNet
from core.model.gcn import GraphConvNet
from core.model.gin import GraphIsomNet
from core.model.sgc import SimpleGraphConv
from core.model.gcn_un import UnfeatGraphConvNet
from core.model.gin_un import UnfeatGraphIsomNet
from core.model.gat_un import UnfeatGraphAttNet
from core.model.MIA.mia import FCNet


def Model_Loader(args):

    if args.model == "gat":
        return GraphAttNet
    
    if args.model == "gcn":
        return GraphConvNet

    if args.model == "gin":
        return GraphIsomNet
    
    if args.model == "sgc":
        return SimpleGraphConv
    
def Unlearn_Model_Loader(args, ckpt):

    args.model = ckpt["model"]
    args.depth = ckpt["depth"]
    args.dropout = ckpt["depth"]
    args.batchnorm = ckpt["batchnorm"]
    args.latent_size = ckpt["latent_size"]

    if args.unlearn_method in ["contrastive", "retrain"]:
        
        if args.model == "gat":
            return GraphAttNet
        
        if args.model == "gcn":
            return GraphConvNet

        if args.model == "gin":
            return GraphIsomNet
        
        if args.model == "sgc":
            return SimpleGraphConv
    
    else:
        if args.unlearn_method == "unfeat":
            if args.model == "gat":
                return UnfeatGraphAttNet  
                
            if args.model == "gcn":
                return UnfeatGraphConvNet  
                
            if args.model == "gin":
                return UnfeatGraphIsomNet  
                
            if args.model == "sgc":
                return SimpleGraphConv  
    

def get_mia_attack_model(in_feat,
                         layer_dims,
                         device
                         ):
    model = FCNet(in_features=in_feat, layer_dims=layer_dims)
    model.to(device)

    return model