import copy
import torch
import torch.optim as optim
from functools import partial

def get_optim(args):
    if args.optimizer == "adam":
        return partial(optim.Adam, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.optimizer == "sgd":
        return partial(optim.SGD, lr=args.learning_rate, momentum=args.momentum)
    
    if args.optimizer == "adamw":
        return partial(optim.AdamW, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.optimizer == "adadelta":
        return partial(optim.Adadelta, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.optimizer == "adagrad":
        return partial(optim.Adagrad,  lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.optimizer == "adamax":
        return partial(optim.Adamax,  lr=args.learning_rate, weight_decay=args.weight_decay)

    
def to_device(blocks, device):    
    out_blocks = [b.to(device) for b in blocks]
    data = out_blocks[0].srcdata["feat"]
    label = out_blocks[-1].dstdata["label"]
    dst_feat = out_blocks[-1].dstdata["feat"]

    return out_blocks, data, label, dst_feat

def out_dict():
    out_dict = dict()
    out_dict["train_loss"] = list()
    out_dict["train_acc"] = list()
    
    return out_dict

def clone_blocks(blocks, data, ori_feat):
    out_blocks = [copy.deepcopy(b) for b in blocks]
    out_data = data.clone().detach()
    out_ori_feat = ori_feat.clone().detach()  

    return out_blocks, out_data, out_ori_feat  