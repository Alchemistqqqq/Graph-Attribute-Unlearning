import dgl
import torch
import numpy as np
import networkx as nx

# Typing
import torch.Tensor as Tensor
import dgl.DGLGraph as DGLGraph
import netwokrx.Graph as NXGraph
from typing import List
"""

Utils for unlearn.py

"""

"""
TODO

1. Maybe consider ISTA algorithm later for PPRs

"""

def _to_networkx(g:DGLGraph)->NXGraph:
    _g = dgl.remove_self_loop(g)
    nx_g = nx.Graph(dgl.to_networkx(_g))

    return nx_g

def get_ppr(g:DGLGraph, target:int)->Tensor:
    nx_g = _to_networkx(g)
    personalize = {i:0 for i in nx_g.nodes()}
    personalize[target] = 1
    ppr_vec = nx.pagerank(nx_g, personalization=personalize)

    return torch.Tensor(ppr_vec)


def get_reverse_ppr(g:DGLGraph, source:int, target:int, normalize:bool=True)->float:

    nx_g = _to_networkx(g)
    personalize = {i:0 for i in nx_g.nodes()}
    personalize[source] = 1
    ppr_vec = nx.pagerank(nx_g, personalization=personalize)
    
    if normalize:
        _max = max(ppr_vec)
        _min = min(ppr_vec)

        return (ppr_vec[target] - _max)(_max - _min)
    else:
        return ppr_vec[target]


def get_ppr_list(g:DGLGraph, targets: List[int])->List:
    
    nx_g = _to_networkx(g)

    ppr_list = list()
    for t in targets:
        personalize = {i:0 for i in nx_g.nodes()}
        personalize[t] = 1
        ppr_vec = nx.pagerank(nx_g, personalization=personalize)
        ppr_list.append(torch.Tensor(ppr_vec))

    return ppr_list


def get_reverse_ppr_list(g:DGLGraph, sources: List[int], targets: List[int])->Tensor:

    nx_g = _to_networkx(g)

    ppr_sources = dict()
    for s in sources:
        personalize = {i:0 for i in nx_g.nodes()}
        personalize[t] = 1
        ppr_vec = nx.pagerank(nx_g, personalization=personalize)
        ppr_sources[s] = ppr_vec

