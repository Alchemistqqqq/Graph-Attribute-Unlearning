import os
import dgl
import copy
import torch
import numpy as np
from dgl.transforms import AddSelfLoop, RemoveSelfLoop
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
from core.data.datasets.datasets import GRAPHDATA
from core.data.remove.remove import UNLEARN_GRAPH
from core.data.transforms import InductiveGraphSplit, RandomNodeSplit
from core.data.sampler import UnlearnNodeNeighborSampler
from core.data.sampler import UnlearnKthNodeNeighborSampler, KthNodeNeighborSampler

# Typing
from typing import Dict

def Graph_Loader(args):

    """
    Load a graph
    conduct unlearning procedures if necessary 
    
    """
    graph = GRAPHDATA[args.dataset](args)
    
    # Split graph
    if args.split == "default":
        if "train_mask" not in graph.ndata:
            raise ValueError("Cannot find predefined graph split")

    elif args.split == "transductive":
        graph = RandomNodeSplit(num_val=args.val_ratio,
                                num_test=args.test_ratio,
                                shuffle=False)(graph)

    elif args.split == "inductive":
        graph = InductiveGraphSplit(num_val=args.val_ratio,
                                    num_test=args.test_ratio)(graph)

    elif args.split == "none":
        # Make everything train-label
        train_mask = torch.Tensor([True for _ in range(graph.num_nodes())])
        graph.ndata["train_mask"] = train_mask   
        graph.ndata["test_mask"] = torch.Tensor([False for _ in range(graph.num_nodes())])
        graph.ndata["valid_mask"] = torch.Tensor([False for _ in range(graph.num_nodes())])
    
    else:
        raise ValueError("Unrecognized split")

    graph = AddSelfLoop()(graph)
    return graph

def Graph_save(save_path, graph):
    """
    Save a graph
    """
    dgl.save_graphs(os.path.join(save_path, "graph.bin"), graph)

def Graph_Load_from(args):
    """
    load a graph
    """
    load_path = os.path.join(args.load_path, "graph.bin")
    graphs, _ = dgl.load_graphs(load_path)
    return graphs[0]

def Graph_Remove_Unlearnables(args, graph):

    """
    Remove components of graphs (node features, edges, etc.)
    for unlearning purpose

    """

    _graph = copy.deepcopy(graph)
    _graph = RemoveSelfLoop()(_graph)
    unlearn_graph = UNLEARN_GRAPH[args.unlearn_type](args, _graph)
    unlearn_graph = AddSelfLoop()(unlearn_graph)

    return unlearn_graph

def Graph_Remove_Mia(args, graph):
    """

    Remove portion of training nodes for the sake of MIA.
    
    """

    _graph = copy.deepcopy(graph)
    _train_mask = _graph.ndata["train_mask"].cpu().numpy()
    train_nodes = torch.where(_graph.ndata["train_mask"] > 0)[0]
    args.mia_train_ratio = int(len(train_nodes)*args.mia_train_ratio)
    used_train_nodes = np.random.choice(train_nodes, args.mia_train_ratio, replace=False)
    used_train_mask = np.in1d(np.arange(graph.num_nodes()), used_train_nodes)
    unused_train_mask = np.logical_and(~copy.deepcopy(used_train_mask), _train_mask)
    _graph.ndata["train_mask"] = torch.from_numpy(used_train_mask).to(_graph.device)
    _graph.ndata["unused_train_mask"] = torch.from_numpy(unused_train_mask).to(_graph.device)

    return _graph


def Graph_Dataloader(args, graph)->Dict:

    sampler = MultiLayerFullNeighborSampler(args.depth)

    loaders = dict()

    if "train_mask" in graph.ndata:

        train_loader = DataLoader(graph=graph,
                                batch_size=args.batch_size,
                                indices=torch.where(graph.ndata["train_mask"] == True)[0],
                                graph_sampler=sampler,
                                device=torch.device(args.device),
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)
        loaders["train"] = train_loader
    
    if "test_mask" in graph.ndata:
        test_loader  = DataLoader(graph=graph,
                                batch_size=args.batch_size,
                                indices=torch.where(graph.ndata["test_mask"] == True)[0],
                                graph_sampler=sampler,
                                device=torch.device(args.device),
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)
        loaders["test"] = test_loader

    if "valid_mask" in graph.ndata:    
        valid_loader = DataLoader(graph=graph,
                                batch_size=args.batch_size,
                                indices=torch.where(graph.ndata["valid_mask"] == True)[0],
                                graph_sampler=sampler,
                                device=torch.device(args.device),
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)
        loaders["valid"] = valid_loader
    
    return loaders


def Graph_Unlearn_Dataloader(args, unlearn_graph, graph) -> Dict:
    loaders = dict()

    if args.unlearn_method in ["contrastive", "unfeat"]:
        # For "contrastive" and "unfeat" methods
        graph = graph.to(args.device)
        unlearn_graph = unlearn_graph.to(args.device)

        unlearn_sampler = UnlearnKthNodeNeighborSampler(depth=args.depth, fanouts=-1)
        retain_sampler = MultiLayerFullNeighborSampler(args.depth)
        device = torch.device(args.device)

        loaders["train"] = DataLoader(graph=graph,
                                  batch_size=args.batch_size,
                                  indices=torch.where(unlearn_graph.ndata["train_mask"] == True)[0],
                                  graph_sampler=retain_sampler,
                                  device=device,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=0)

        loaders["unlearn"] = DataLoader(graph=unlearn_graph,
                                        batch_size=args.batch_size,
                                        indices=torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0],
                                        graph_sampler=unlearn_sampler,
                                        device=device,
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0)

        loaders["unlearn_test"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0],
                                             graph_sampler=retain_sampler,
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["retain_train"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(unlearn_graph.ndata["retain_mask"] == True)[0],
                                             graph_sampler=retain_sampler,
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["retain_test"] = DataLoader(graph=graph,
                                            batch_size=args.batch_size,
                                            indices=torch.where(graph.ndata["test_mask"] == True)[0],
                                            graph_sampler=retain_sampler,
                                            device=device,
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0)

        loaders["retain_valid"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(graph.ndata["valid_mask"] == True)[0],
                                             graph_sampler=retain_sampler,
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["stop_cond_ul"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0],
                                             graph_sampler=KthNodeNeighborSampler(depth=args.depth, fanouts=-1),
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["stop_cond_test"] = DataLoader(graph=graph,
                                               batch_size=args.batch_size,
                                               indices=torch.where(graph.ndata["test_mask"] == True)[0],
                                               graph_sampler=KthNodeNeighborSampler(depth=args.depth, fanouts=-1),
                                               device=device,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    elif args.unlearn_method in ["retrain"]:
        # For "retrain" method
        graph = graph.to(args.device)
        unlearn_graph = unlearn_graph.to(args.device)
        print("mask:",unlearn_graph.ndata["unlearn_mask"])
        print("unlearn_mask 总长度:", unlearn_graph.ndata["unlearn_mask"].shape[0])
        print("unlearn_mask 为 True 的数量:", unlearn_graph.ndata["unlearn_mask"].sum().item())

        unlearn_sampler = UnlearnKthNodeNeighborSampler(depth=args.depth, fanouts=-1)
        retain_sampler = MultiLayerFullNeighborSampler(args.depth)
        device = torch.device(args.device)
        all_indices = torch.cat([torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0], 
                         torch.where(unlearn_graph.ndata["retain_mask"] == True)[0]])

        # Create a new DataLoader for the combined dataset
        loaders["all"] = DataLoader(graph=graph,
                                    batch_size=args.batch_size,
                                    indices=all_indices,
                                    graph_sampler=retain_sampler,  # Use the same sampler for the combined dataset
                                    device=device,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=0)

        loaders["unlearn"] = DataLoader(graph=unlearn_graph,
                                        batch_size=args.batch_size,
                                        indices=torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0],
                                        graph_sampler=retain_sampler,
                                        device=device,
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0)

        loaders["unlearn_test"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0],
                                             graph_sampler=retain_sampler,
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["retain_train"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(unlearn_graph.ndata["retain_mask"] == True)[0],
                                             graph_sampler=retain_sampler,
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["retain_test"] = DataLoader(graph=graph,
                                            batch_size=args.batch_size,
                                            indices=torch.where(graph.ndata["test_mask"] == True)[0],
                                            graph_sampler=retain_sampler,
                                            device=device,
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0)

        loaders["retain_valid"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(graph.ndata["valid_mask"] == True)[0],
                                             graph_sampler=retain_sampler,
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["stop_cond_ul"] = DataLoader(graph=graph,
                                             batch_size=args.batch_size,
                                             indices=torch.where(unlearn_graph.ndata["unlearn_mask"] == True)[0],
                                             graph_sampler=KthNodeNeighborSampler(depth=args.depth, fanouts=-1),
                                             device=device,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=0)

        loaders["stop_cond_test"] = DataLoader(graph=graph,
                                               batch_size=args.batch_size,
                                               indices=torch.where(graph.ndata["test_mask"] == True)[0],
                                               graph_sampler=KthNodeNeighborSampler(depth=args.depth, fanouts=-1),
                                               device=device,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    return loaders

    
