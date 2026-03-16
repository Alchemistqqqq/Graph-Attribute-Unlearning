import dgl
import torch
import numpy as np
from dgl.dataloading import BlockSampler
from dgl.dataloading import set_src_lazy_features
from dgl.dataloading import set_dst_lazy_features
from dgl.dataloading import set_edge_lazy_features
from dgl.transforms import to_block
from dgl.sampling import sample_neighbors
from dgl.heterograph import DGLBlock


EID = "_ID"
NID = "_ID"


class CustomBlockSampler(BlockSampler):
    """
    Override BlockSampler
    """

    def __init__(self,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
    
    def sample(self, g, seed_nodes, exclude_eids=None):
        result=self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return self.assign_lazy_features(result)

    def assign_lazy_features(self, result):
        """
        Results: Blocks

        3/3/ Adding KthNodNeighborSampler
        """
        out_results = list()
        for res in result: 
            if isinstance(res[0], DGLBlock):
                set_src_lazy_features(res[0], self.prefetch_node_feats)
                set_dst_lazy_features(res[-1], self.prefetch_labels)
                for block in res:
                    set_edge_lazy_features(block, self.prefetch_edge_feats)

            elif isinstance(res, dict):
                for _k, block_list in res.items():
                    if isinstance(block_list[0], DGLBlock):
                        set_src_lazy_features(block_list[0], self.prefetch_node_feats)
                        set_dst_lazy_features(block_list[-1], self.prefetch_labels)

            out_results.append(res)
        
        return tuple(out_results)

class UnlearnNodeNeighborSampler(CustomBlockSampler):
    """
    Sample for 2-layers only 
    Not used anymore. depricated.

    """
    def __init__(self, depth:int, fanouts:int, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = [fanouts for _ in range(depth)]
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        ul_blocks = list() # Blocks for MPNN for unlearning nodes
        nb1_blocks = list() # Blocks for MPNN for neighbors of unlearning nodes
        nb2_blocks = list() # Blocks for MPNN for neighbors of neighbors of unlearning nodes

        # ul_blocks
        ul_nodes = seed_nodes.clone()
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            ul_blocks.insert(0, block)

        # nb1_blocks & nb2_blocks
        # Need to find edges to each other
        src, dst = g.edges()
        src_t = torch.isin(src, ul_nodes)
        dst_t = torch.isin(dst, ul_nodes)
        _t = src_t * dst_t
        exclude_eids = g.edge_ids(src[_t], dst[_t])
        
        neighbors = g.sample_neighbors(ul_nodes, fanout, edge_dir=self.edge_dir,
                                       prob=self.prob, replace=self.replace,
                                       exclude_edges=exclude_eids, output_device=self.output_device)
        
        seed_nodes = torch.unique(neighbors.edges()[0])

        self.fanouts.append(self.fanouts[-1])
        exclude_eids = g.out_edges(ul_nodes, form="eid")
        
        for idx, fanout in enumerate(reversed(self.fanouts)):
            frontier = g.sample_neighbors(seed_nodes, fanout, edge_dir=self.edge_dir,
                                          prob=self.prob, replace=self.replace,
                                          exclude_edges=exclude_eids, output_device=self.output_device)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            if idx + 1 < len(self.fanouts):
                nb1_blocks.insert(0, block)
            if idx > 0: 
                nb2_blocks.insert(0, block)
                
            if idx == 0:
                src_nodes = block.srcdata[NID].numpy()
                dst_nodes = block.dstdata[NID].numpy()

                nb_rel = np.zeros((len(dst_nodes), len(src_nodes)), dtype=bool)
            
                src, dst = frontier.edges()
                src, dst = src.numpy(), dst.numpy()
                for s, d in zip(src, dst):
                    s_idx = np.where(src_nodes == s)
                    d_idx = np.where(dst_nodes == d)
                    nb_rel[d_idx, s_idx] = True
                    
                nb_rel = torch.from_numpy(nb_rel)


        return ul_blocks, nb1_blocks, nb2_blocks, nb_rel
    
class UnlearnKthNodeNeighborSampler(CustomBlockSampler):
    """

    1. Obtain blocks for inferring unlearning nodes (seed nodes)
    2. Obtain blocks for (depth+1) hop neighbors of unlearning nodes

    """
    def __init__(self, depth:int, fanouts:int, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = [fanouts for _ in range(depth)]
        self.edge_dir = edge_dir
        self.prob = prob # Sampling probability for n-th nodes
        self.replace = replace
        self.depth = depth
        self.nb_hops = depth+1

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        ul_blocks = list() # Blocks for MPNN for unlearning nodes
        nb_blocks = {i:list() for i in range(self.nb_hops)} # List of blocks for MPNN for neighbors
        ul_link = dict()
        nb_link = dict()
        # print("Seed nodes len ", len(seed_nodes))
        # ul_blocks
        ul_nodes = seed_nodes.clone()
        test_nodes = torch.from_numpy(np.where(g.ndata["train_mask"].cpu().numpy() > 0)[0])
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=None,
                replace=self.replace)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            ul_blocks.insert(0, block)
            # print(len(block.srcdata[NID]), len(block.dstdata[NID]))

        _neighbors = torch.unique(ul_blocks[-1].edges()[0])

        unlearn_link = list()
        for n in _neighbors:
            unlearn_link.append(g.has_edges_between(n, ul_nodes))
        unlearn_link = torch.stack(unlearn_link)
        ul_link[0] = unlearn_link

        # nb1_blocks & nb2_blocks
        # Need to find edges to each other
        src, dst = g.edges()
        test_nodes = test_nodes.to(g.device)
        # src_t = torch.isin(src, ul_nodes)
        # dst_t = torch.isin(dst, ul_nodes)
        # src_t_2 = torch.isin(src, test_nodes)
        # dst_t_2 = torch.isin(dst, test_nodes)
        # _t = src_t * dst_t * src_t_2 * dst_t_2
        _t = torch.isin(src, ul_nodes) * torch.isin(dst, ul_nodes) * \
             torch.isin(src, test_nodes) * torch.isin(dst, test_nodes)
        exclude_eids = g.edge_ids(src[_t], dst[_t])
        
        neighbors = g.sample_neighbors(ul_nodes, fanout, edge_dir=self.edge_dir,
                                       prob=self.prob, replace=self.replace,
                                       exclude_edges=exclude_eids, output_device=self.output_device)
        
        seed_nodes = torch.unique(neighbors.edges()[0])
        # Need to identify who are whose neighbors
        unlearn_link = list()
        for n in seed_nodes:
            unlearn_link.append(g.has_edges_between(n, ul_nodes))
        unlearn_link = torch.stack(unlearn_link)
        # print("unlearn_link", unlearn_link.shape)
        nb_link[-1] = unlearn_link

        nb_fanouts = [self.fanouts[-1] for _ in range(self.depth+len(self.fanouts))]
        exclude_eids = g.out_edges(ul_nodes, form="eid")
        exclude_eids2 = g.out_edges(test_nodes, form="eid")
        
        for idx, fanout in enumerate(reversed(nb_fanouts)):
            frontier = g.sample_neighbors(seed_nodes, fanout, edge_dir=self.edge_dir,
                                          prob=self.prob, replace=self.replace,
                                          exclude_edges=exclude_eids, output_device=self.output_device)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]

            for _k, _block in nb_blocks.items():
                # Store each block to appropriate blocklists
                if idx - _k >= 0 and idx - _k < self.depth:
                    _block.insert(0, block)
                
            # Store node relations
            if idx < self.depth:
                src_nodes = block.srcdata[NID].cpu().numpy()
                dst_nodes = block.dstdata[NID].cpu().numpy()

                _nb_rel = np.zeros((len(dst_nodes), len(src_nodes)), dtype=bool)
            
                src, dst = frontier.edges()
                src, dst = src.cpu().numpy(), dst.cpu().numpy()
                for s, d in zip(src, dst):
                    s_idx = np.where(src_nodes == s)
                    d_idx = np.where(dst_nodes == d)
                    _nb_rel[d_idx, s_idx] = True
                    
                nb_link[idx] = torch.from_numpy(_nb_rel)

        return ul_blocks, nb_blocks, ul_link, nb_link
    
class KthNodeNeighborSampler(CustomBlockSampler):
    """

    1. Obtain blocks for inferring unlearning nodes (seed nodes)
    2. Obtain blocks for (depth+1) hop neighbors of unlearning nodes

    """
    def __init__(self, depth:int, fanouts:int, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = [fanouts for _ in range(depth)]
        self.edge_dir = edge_dir
        self.prob = prob # Sampling probability for n-th nodes
        self.replace = replace
        self.depth = depth
        self.nb_hops = depth+1

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        ul_blocks = list() # Blocks for MPNN for unlearning nodes
        nb_blocks = {i:list() for i in range(self.nb_hops)} # List of blocks for MPNN for neighbors
        ul_link = dict()
        nb_link = dict()
        # print("Seed nodes len ", len(seed_nodes))
        # ul_blocks
        ul_nodes = seed_nodes.clone()
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=None,
                replace=self.replace)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            ul_blocks.insert(0, block)
            # print(len(block.srcdata[NID]), len(block.dstdata[NID]))

        _neighbors = torch.unique(ul_blocks[-1].edges()[0])

        unlearn_link = list()
        for n in _neighbors:
            unlearn_link.append(g.has_edges_between(n, ul_nodes))
        unlearn_link = torch.stack(unlearn_link)
        ul_link[0] = unlearn_link

        # nb1_blocks & nb2_blocks
        # Need to find edges to each other
        
        neighbors = g.sample_neighbors(ul_nodes, fanout, edge_dir=self.edge_dir,
                                       prob=self.prob, replace=self.replace,
                                       output_device=self.output_device)
        
        seed_nodes = torch.unique(neighbors.edges()[0])
        # Need to identify who are whose neighbors
        unlearn_link = list()
        for n in seed_nodes:
            unlearn_link.append(g.has_edges_between(n, ul_nodes))
        unlearn_link = torch.stack(unlearn_link)
        # print("unlearn_link", unlearn_link.shape)
        # nb_link[-1] = unlearn_link
        ul_link[0] = unlearn_link

        nb_fanouts = [self.fanouts[-1] for _ in range(self.depth+len(self.fanouts))]
        exclude_eids = g.out_edges(ul_nodes, form="eid")
        
        for idx, fanout in enumerate(reversed(nb_fanouts)):
            frontier = g.sample_neighbors(seed_nodes, fanout, edge_dir=self.edge_dir,
                                          prob=self.prob, replace=self.replace,
                                          output_device=self.output_device)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]

            for _k, _block in nb_blocks.items():
                # Store each block to appropriate blocklists
                if idx - _k >= 0 and idx - _k < self.depth:
                    _block.insert(0, block)
                
            # Store node relations
            if idx < self.depth:
                src_nodes = block.srcdata[NID].numpy()
                dst_nodes = block.dstdata[NID].numpy()

                _nb_rel = np.zeros((len(dst_nodes), len(src_nodes)), dtype=bool)
            
                src, dst = frontier.edges()
                src, dst = src.numpy(), dst.numpy()
                for s, d in zip(src, dst):
                    s_idx = np.where(src_nodes == s)
                    d_idx = np.where(dst_nodes == d)
                    _nb_rel[d_idx, s_idx] = True
                    
                nb_link[idx] = torch.from_numpy(_nb_rel)

        return ul_blocks, nb_blocks, ul_link