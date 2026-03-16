import os
import torch
from typing import Dict
from dgl import DGLGraph


def Save_Splits(args, graph:DGLGraph)->Dict:

    save_dict = dict()

    save_dict["split"] = args.split
    try:
        save_dict["train_mask"] = graph.ndata["train_mask"]
    except:
        raise ValueError("No train mask specified")
    try:
        save_dict["test_mask"] = graph.ndata["test_mask"]
    except:
        save_dict["test_mask"] = None
    try:
        save_dict["valid_mask"] = graph.ndata["valid_mask"]
    except:
        save_dict["valid_mask"] = None

    return save_dict

def Checkpoint_Loader(args):
    # Get the model.pt directly.
    load_path = os.path.join(args.load_path, args.load_epoch, "model.pt")
    checkpoint = torch.load(load_path)
    state_dict = dict()
    
    state_dict["state_dict"] = checkpoint["state_dict"]

    try:
        state_dict["model"] = checkpoint["model"]
        state_dict["depth"] = checkpoint["depth"]
        state_dict["dropout"] = checkpoint["dropout"]
        state_dict["batchnorm"] = checkpoint["batchnorm"]
        state_dict["latent_size"] = checkpoint["latent_size"]
    except:
        state_dict["model"] = args.model
        state_dict["depth"] = args.depth
        state_dict["dropout"] = args.dropout
        state_dict["batchnorm"] = args.batchnorm
        state_dict["latent_size"] = args.latent_size    

    splits = dict()

    # try:
    #     splits["train_mask"] = checkpoint["train_mask"]
    # except:
    #     raise ValueError("Train mask not defiend")

    # try:
    #     splits["test_mask"] = checkpoint["test_mask"]
    # except:
    #     raise ValueError("Test mask not defined")
    
    # try:
    #     splits["valid_mask"] = checkpoint["valid_mask"]
    # except:
    #     splits["valid_mask"] = None
    
    return state_dict, splits

def Victim_Loader(args):

    load_path = os.path.join(args.load_path, str(args.victim_load_epoch), "model.pt")
    checkpoint = torch.load(load_path)
    
    state_dict = checkpoint["state_dict"]
    
    return state_dict