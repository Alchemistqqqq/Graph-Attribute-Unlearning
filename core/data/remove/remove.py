import dgl
import copy
import torch
import numpy as np

from core.data.transforms import RemoveNodeEdges
from core.data.transforms import RemoveNodeFeatures
from core.data.transforms import RemoveNodes

UNLEARN_GRAPH = {}

def _add(fn):
    UNLEARN_GRAPH[fn.__name__] = fn
    return fn

@_add
def random_node_contrastive(args, graph):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_nodes = np.where(graph.ndata["train_mask"].numpy() > 0)[0]
    original_num_unlearn = args.num_unlearn
    if args.num_unlearn < 1:
        args.num_unlearn = int(len(train_nodes)*args.num_unlearn)
    else:
        args.num_unlearn = int(args.num_unlearn)
    node_list = np.random.choice(train_nodes, args.num_unlearn, replace=False)
    print(f"[INFO] Requested num_unlearn (arg): {original_num_unlearn}")
    print(f"[INFO] Actual num_unlearn used: {args.num_unlearn}")
    print(f"[INFO] Number of forgotten nodes: {len(node_list)}")
    node_list = torch.tensor(node_list)
    node_list = torch.sort(node_list).values
    print(f"[INFO] Forgotten node IDs: {node_list}")
    unlearn_mask = np.in1d(np.arange(graph.num_nodes()), node_list)
    print(f"[INFO] Unlearn mask: {unlearn_mask}")
    print("unlearn_mask 为 True 的数量:", np.count_nonzero(unlearn_mask))
    retain_mask = np.logical_and(~copy.deepcopy(unlearn_mask), graph.ndata["train_mask"].numpy())    
    out_graph = copy.deepcopy(graph)
    out_graph.ndata["unlearn_mask"] = torch.from_numpy(unlearn_mask)
    out_graph.ndata["retain_mask"] = torch.from_numpy(retain_mask)

    return out_graph 

@_add
def random_node_feature(args, graph):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_nodes = np.where(graph.ndata["train_mask"].numpy() > 0)[0]
    original_num_unlearn = args.num_unlearn
    if args.num_unlearn < 1:
        args.num_unlearn = int(len(train_nodes)*args.num_unlearn)
    else:
        args.num_unlearn = int(args.num_unlearn)
    node_list = np.random.choice(train_nodes, args.num_unlearn, replace=False)
    print(f"[INFO] Requested num_unlearn (arg): {original_num_unlearn}")
    print(f"[INFO] Actual num_unlearn used: {args.num_unlearn}")
    print(f"[INFO] Number of forgotten nodes: {len(node_list)}")
    print(f"[INFO] Forgotten node IDs: {node_list}")
    unlearn_mask = np.in1d(np.arange(graph.num_nodes()), node_list)
    print(f"[INFO] Unlearn mask: {unlearn_mask}")
    print("unlearn_mask 为 True 的数量:", np.count_nonzero(unlearn_mask))
    retain_mask = np.logical_and(~copy.deepcopy(unlearn_mask), graph.ndata["train_mask"].numpy())   
    out_graph = RemoveNodeFeatures(node_list)(copy.deepcopy(graph))
    out_graph.ndata["unlearn_mask"] = torch.from_numpy(unlearn_mask)
    out_graph.ndata["retain_mask"] = torch.from_numpy(retain_mask)
    
    return out_graph

@_add
def random_node_edge(args, graph):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_nodes = np.arange(graph.num_nodes())[graph.ndata["train_mask"].numpy()]
    if args.num_unlearn < 1:
        args.num_unlearn = int(len(train_nodes)*args.num_unlearn)
    else:
        args.num_unlearn = int(args.num_unlearn)
    node_list = np.random.choice(train_nodes, args.num_unlearn, replace=False)

    out_graph = RemoveNodeEdges(node_list)(copy.deepcopy(graph))
    return out_graph

@_add
def random_node(args, graph):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_nodes = np.arange(graph.num_nodes())[graph.ndata["train_mask"].numpy()]
    if args.num_unlearn < 1:
        args.num_unlearn = int(len(train_nodes)*args.num_unlearn)
    else:
        args.num_unlearn = int(args.num_unlearn)
    node_list = np.random.choice(train_nodes, args.num_unlearn, replace=False)
    out_graph = RemoveNodes(node_list)(copy.deepcopy(graph))

    return out_graph
