import dgl
import torch
import dgl.data as DS
# import torch_geometric.datasets as TDS
# from torch_geometric.utils import to_dgl
from ogb.nodeproppred import DglNodePropPredDataset


GRAPHDATA = {}

def _add(fn):
    GRAPHDATA[fn.__name__] = fn
    return fn

@_add
def cora(args):
    graph = DS.CoraGraphDataset(raw_dir=args.data_path, transform=None)[0]
    return graph

@_add
def pubmed(args):
    graph = DS.PubmedGraphDataset(raw_dir=args.data_path,  transform=None)[0]
    return graph

@_add
def cs(args):
    graph = DS.CoauthorCSDataset(raw_dir=args.data_path,  transform=None)[0]
    return graph

@_add
def citeseer(args):
    graph = DS.CiteseerGraphDataset(raw_dir=args.data_path,  transform=None)[0]
    return graph

@_add
def reddit(args):
    graph = DS.RedditDataset(raw_dir=args.data_path, transform=None)[0]
    return graph

@_add
def ogbn_arxiv(args):
    # You can specify which OGBN dataset to use via args.dataset_name
    # Common options are: 'ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root=args.data_path)
    graph, labels = dataset[0]
    
    # Add node labels
    graph.ndata['label'] = labels.squeeze()
    
    # Add train/val/test masks
    split_idx = dataset.get_idx_split()
    graph.ndata['train_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    graph.ndata['val_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    graph.ndata['test_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    
    graph.ndata['train_mask'][split_idx['train']] = True
    graph.ndata['val_mask'][split_idx['valid']] = True
    graph.ndata['test_mask'][split_idx['test']] = True
    
    return graph

@_add
def physics(args):
    graph = DS.CoauthorPhysicsDataset(raw_dir=args.data_path, transform=None)[0]
    return graph

# @_add
# def obgn(args):
#     graph = 

# @_add
# def karate(args):
#     t_graph = TDS.KarateClub(transform=None)[0]
#     row, col = t_graph.edge_index
#     d_graph = dgl.graph((row, col))
#     d_graph.ndata["feat"] = t_graph.x.clone()
#     d_graph.ndata["label"] = t_graph.y.clone()
#     d_graph.ndata["train_mask"] = t_graph.train_mask.clone()
#     d_graph.ndata["test_mask"] = ~t_graph.train_mask.clone()
#     return d_graph


