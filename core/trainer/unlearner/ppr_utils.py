import dgl
import torch
import numpy as np
import torch.nn.functional as F
import dgl.sparse as dglsp
from core.trainer.unlearner.dglsp_utils import *
from dgl.heterograph import DGLBlock
import torch.nn as nn
from torch import Tensor, device
import numba


def get_ppr_node_ista(nnodes,
                      alpha,
                      epsilon,
                      rho,
                      out_degree,
                      node_index,
                      deg_inv,
                      indices,
                      indptr,
                      device):
    """
    nnodes:     nodes (from adjacent matrix)
    alpha:      hyperparameter
    epsilon:    hyperparameter
    rho:        hyperparameter
    out_degree: out degree from adjacent matrix
    node_index: target node (of calculating PPR)
    deg_inv:    inverse degree
    indices:    CSR indices of adjacent matrix
    indptr:     CSR indptr of adjacent matrix
    device:     device to process
    """

    s = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    s[node_index] = 1

    indices = indices.to(device)
    indptr = indptr.to(device)
    out_degree = out_degree.to(device)
    
    p_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    p_new = torch.zeros_like(nnodes, device=device, dtype=torch.float)

    d_fp_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    d_fp_old[node_index] = -alpha * deg_inv[node_index]

    _i = 0
    while (torch.linalg.vector_norm(d_fp_old, float("inf")) > \
           (1+ epsilon) * rho * alpha):
        # Logging
        # print("iteration ", _i)
        _i+=1

        S_k = ((p_old - d_fp_old) >= rho * alpha).nonzero()
        d_pk = -(d_fp_old[S_k.to(torch.int)] + rho * alpha)
        p_new[S_k.to(torch.int)] = p_old[S_k.to(torch.int)] + d_pk
        d_fp_new = d_fp_old

        is_d_pk = torch.zeros_like(nnodes, device=device, dtype=torch.float)
        is_d_pk[S_k.to(torch.int)] = d_pk

        for i in S_k:
            tmp_sum=0
            for l in indices[indptr[i]:indptr[i+1]]:
                if torch.isin(l, S_k).any():
                    tmp_sum += is_d_pk[l] / out_degree[l]
            d_fp_new[i] =   (1 - 1. / out_degree[i]) * d_fp_old[i] - \
                            rho * alpha / out_degree[i] - \
                            0.5 * (1 - alpha) * is_d_pk[i] / out_degree[i] - \
                            0.5 * (1 - alpha) / out_degree[i] * tmp_sum
        neighbors_set = []
        for s in S_k:
            for neighbor in indices[indptr[s]:indptr[s + 1]]:
                if neighbor not in neighbors_set:
                    if not torch.isin(neighbor, S_k):
                        neighbors_set.append(neighbor)
        # print("neighbor", neighbors_set)
        for j in neighbors_set:
            tmp_sum = 0
            for l in indices[indptr[j]:indptr[j+1]]:
                if torch.isin(l, S_k).any():
                    tmp_sum += is_d_pk[l] / out_degree[l]
            
            
            d_fp_new[j] = d_fp_old[j] - 0.5 * (1 - alpha) / out_degree[j] * tmp_sum
        d_fp_old = d_fp_new
        p_old = p_new
    
    return p_old

def get_ppr_node_ista_p0(nnodes,
                      alpha,
                      epsilon,
                      rho,
                      out_degree,
                      node_index,
                      deg_inv,
                      adj,
                      device):
    # Depricated.
    # Not finished
    """
    nnodes:     nodes (from adjacent matrix)
    alpha:      hyperparameter
    epsilon:    hyperparameter
    rho:        hyperparameter
    out_degree: out degree from adjacent matrix
    node_index: target node (of calculating PPR)
    deg_inv:    inverse degree
    adj:        Adjacency matrix in dglsp
    device:     device to process
    """

    s = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    s[node_index] = 1

    indices = adj.indices()
    val = adj.val.to(torch.int)
    adj_sp = dglsp.spmatrix(indices, val).to(device)

    out_degree = out_degree.to(device)
    
    p_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    p_new = torch.zeros_like(nnodes, device=device, dtype=torch.float)

    d_fp_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    d_fp_old[node_index] = -alpha * deg_inv[node_index]

    _i = 0
    while (torch.linalg.vector_norm(d_fp_old, float("inf")) > \
           (1+ epsilon) * rho * alpha):
        # Logging
        # print("iteration ", _i)
        _i+=1

        S_k = ((p_old - d_fp_old) >= rho * alpha)
        S_k_idx = S_k.nonzero().to(torch.int)
        d_pk = -(d_fp_old[S_k_idx] + rho * alpha)
        p_new[S_k_idx] = p_old[S_k_idx] + d_pk
        d_fp_new = d_fp_old

        is_d_pk = torch.zeros_like(nnodes, device=device, dtype=torch.float)
        is_d_pk[S_k_idx] = d_pk

        S_k_bool_sp = duplicate_rows(to_dglsp(S_k.to(torch.int)), n=len(S_k_idx), device=device)
        is_d_pk_sp = duplicate_rows(to_dglsp(is_d_pk), n=len(S_k_idx), device=device)
        adj_s_k = select_rows(adj_sp, S_k_idx, device)

        print(S_k_bool_sp, adj_s_k)

        adj_S_k = dglsp.mul(S_k_bool_sp, adj_s_k).to(dtype=torch.float)
        tmp_sum = dglsp.sum(dglsp.mul(adj_S_k, is_d_pk_sp), 1)
        
        d_fp_new[S_k_idx] = (1 - 1. / out_degree[S_k_idx]) * d_fp_old[S_k_idx] - \
                        rho * alpha / out_degree[S_k_idx] - \
                        0.5 * (1-alpha) * is_d_pk[S_k_idx] /  out_degree[S_k_idx] -  \
                        0.5 * (1-alpha) * out_degree[S_k_idx] * tmp_sum

        # for i in S_k:
        #     tmp_sum=0
        #     for l in indices[indptr[i]:indptr[i+1]]:
        #         if torch.isin(l, S_k).any():
        #             tmp_sum += is_d_pk[l] / out_degree[l]
        #     d_fp_new[i] =   (1 - 1. / out_degree[i]) * d_fp_old[i] - \
        #                     rho * alpha / out_degree[i] - \
        #                     0.5 * (1 - alpha) * is_d_pk[i] / out_degree[i] - \
        #                     0.5 * (1 - alpha) / out_degree[i] * tmp_sum

        S_k_bool_sp_neg = duplicate_rows(to_dglsp(~S_k), n=len(S_k_idx), device=device)
        neighbors_set = (dglsp.sum(dglsp.mul(adj_s_k, S_k_bool_sp_neg), dim=0) > 0).to(torch.int)
        neighbors_idx = neighbors_set.nonzero()
        neighbors_set = duplicate_rows(to_dglsp(neighbors_set), len(neighbors_idx), device)

        adj_ngbr = select_rows(adj, neighbors_idx, device)
        S_k_bool_sp = duplicate_rows(to_dglsp(S_k), n=len(neighbors_idx), device=device)
        adj_ngbr_S_k = dglsp.mul(adj_ngbr, S_k_bool_sp).to(torch.float())

        is_d_pk_sp = duplicate_rows(to_dglsp(is_d_pk), n=len(neighbors_idx), device=device)
        out_degree_sp = duplicate_rows(to_dglsp(out_degree), n=len(neighbors_idx), device=device)

        is_d_pk_sp = dglsp.mul(is_d_pk_sp, adj_ngbr_S_k)
        out_degree_sp = dglsp.mul(out_degree_sp, adj_ngbr_S_k)
        tmp_sum = dglsp.sum(dglsp.div(is_d_pk_sp, out_degree_sp), 1)

        d_fp_new[neighbors_idx] = d_fp_new[neighbors_idx] - 0.5 * (1 - alpha) / out_degree[neighbors_idx] * tmp_sum

        # neighbors_set = []
        # for s in S_k:
        #     for neighbor in indices[indptr[s]:indptr[s + 1]]:
        #         if neighbor not in neighbors_set:
        #             if not torch.isin(neighbor, S_k):
        #                 neighbors_set.append(neighbor)
        # # print("neighbor", neighbors_set)
        # for j in neighbors_set:
        #     tmp_sum = 0
        #     for l in indices[indptr[j]:indptr[j+1]]:
        #         if torch.isin(l, S_k).any():
        #             tmp_sum += is_d_pk[l] / out_degree[l]    
        #     d_fp_new[j] = d_fp_old[j] - 0.5 * (1 - alpha) / out_degree[j] * tmp_sum
        
        d_fp_old = d_fp_new
        p_old = p_new
    
    return p_old

def get_ppr_node_ista_p(nnodes:Tensor,
                        alpha:float,
                        epsilon:float,
                        rho:float,
                        out_degree:Tensor,
                        node_index:int,
                        deg_inv:Tensor,
                        adj_indices:Tensor,
                        adj_val:Tensor,
                        device):
    """
    nnodes:     nodes (from adjacent matrix)
    alpha:      hyperparameter
    epsilon:    hyperparameter
    rho:        hyperparameter
    out_degree: out degree from adjacent matrix
    node_index: target node (of calculating PPR)
    deg_inv:    inverse degree
    adj:        Adjacency matrix in dglsp
    device:     device to process
    """

    adj_sp = dglsp.spmatrix(adj_indices, adj_val.to(torch.int)).to(device)

    out_degree = out_degree.to(device, dtype=torch.float)
    
    p_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    p_new = torch.zeros_like(nnodes, device=device, dtype=torch.float)

    d_fp_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    d_fp_old[node_index] = -alpha * deg_inv[node_index]

    _i = 0
    while (torch.linalg.vector_norm(d_fp_old, float("inf")) > (1 + epsilon) * rho * alpha):
        # Logging
        # print("iteration ", _i, " ", torch.linalg.vector_norm(d_fp_old, float("inf")), (1 + epsilon) * rho * alpha)
        _i+=1

        S_k = ((p_old - d_fp_old) >= rho * alpha)
        S_k_idx = S_k.nonzero().flatten().to(torch.int)
        d_pk = -(d_fp_old[S_k_idx] + rho * alpha)
        p_new[S_k_idx] = p_old[S_k_idx] + d_pk
        d_fp_new = d_fp_old
        
        is_d_pk = torch.zeros_like(nnodes, device=device, dtype=torch.float)
        is_d_pk[S_k_idx] = d_pk
        adj_s_k = select_rows(adj_sp, S_k_idx, device)
        adj_S_k = dglsp.sp_mul_v(adj_s_k, S_k.to(torch.int)).to(dtype=torch.float)
        is_d_pk_sp = dglsp.sp_mul_v(adj_S_k, is_d_pk)
        out_degree_sp = dglsp.sp_mul_v(adj_S_k, out_degree)
        out_degree_sp = dglsp.sp_add_v(out_degree_sp, 1e-12*torch.ones(len(nnodes), dtype=torch.float).to(device))
        tmp_sum = dglsp.sum(dglsp.div(is_d_pk_sp, out_degree_sp), 1).flatten()
        d_fp_new[S_k_idx] = (1 - 1. / out_degree[S_k_idx]) * d_fp_old[S_k_idx] - \
                        rho * alpha / out_degree[S_k_idx] - \
                        0.5 * (1-alpha) * is_d_pk[S_k_idx] /  out_degree[S_k_idx] -  \
                        0.5 * (1-alpha) / out_degree[S_k_idx] * tmp_sum

        neighbors = dglsp.sp_mul_v(adj_s_k, (~S_k).to(torch.int))
        neighbors_set = (dglsp.sum(neighbors, 0) > 0).to(torch.int)
        neighbors_idx = neighbors_set.nonzero().flatten()
        # print(neighbors_idx)

        if len(neighbors_idx) != 0:
            adj_ngbr = select_rows(adj_sp, neighbors_idx, device)
            adj_ngbr_S_k = dglsp.sp_mul_v(adj_ngbr, S_k.to(torch.int)).to(dtype=torch.float)

            is_d_pk_sp = dglsp.sp_mul_v(adj_ngbr_S_k, is_d_pk)
            out_degree_sp = dglsp.sp_mul_v(adj_ngbr_S_k, out_degree)

            out_degree_sp  = dglsp.sp_add_v(out_degree_sp, 1e-12*torch.ones(len(nnodes), dtype=torch.float).to(device))
            tmp_sum = dglsp.sum(dglsp.div(is_d_pk_sp, out_degree_sp), 1)

            d_fp_new[neighbors_idx] = d_fp_old[neighbors_idx] - 0.5 * (1 - alpha) / out_degree[neighbors_idx] * tmp_sum
        
        # print(sum(d_fp_old))
        d_fp_old = d_fp_new
        p_old = p_new
    
    return p_old

@torch.jit.script
def get_ppr_node_ista_p_torch_only(nnodes:Tensor,
                                   alpha:float,
                                   epsilon:float,
                                   rho:float,
                                   out_degree:Tensor,
                                   node_index:int,
                                   deg_inv:Tensor,
                                   adj_row:Tensor,
                                   adj_col:Tensor,
                                   adj_val:Tensor,
                                   device:device)->Tensor:
    
    # def _select_rows(mat:Tensor, target_rows:Tensor)->Tensor:
    #     row, col = mat.coo()
    #     val = mat.val
    #     col_len = mat.shape[1]
    #     mask = torch.isin(row, target_rows),
    #     row_m = row[mask]
    #     new_r = torch.zeros_like(row_m)
        
    #     for idx, r in enumerate(target_rows):
    #         new_r += idx* torch.isin(row_m, r).to(torch.int)

    #     new_c = col[mask]
    #     new_v = val[mask]
    #     new_sp = torch.sparse_coo_tensor((new_r, new_c), new_v,
    #                                      size=(len(target_rows), col_len),
    #                                      device=device)
    #     return new_sp

    out_degree = out_degree.to(device, dtype=torch.float)
    p_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    p_new = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    d_fp_old = torch.zeros_like(nnodes, device=device, dtype=torch.float)
    d_fp_old[node_index] = -alpha * deg_inv[node_index]

    _i = 0
    while(torch.linalg.vector_norm(d_fp_old, float("inf")) > (1+epsilon)*rho*alpha):
        _i+=1
        
        S_k = ((p_old - d_fp_old) >= rho*alpha)
        S_k = ((p_old - d_fp_old) >= rho * alpha)
        S_k_idx = S_k.nonzero().flatten().to(torch.int)
        d_pk = -(d_fp_old[S_k_idx] + rho * alpha)
        p_new[S_k_idx] = p_old[S_k_idx] + d_pk
        d_fp_new = d_fp_old
        
        is_d_pk = torch.zeros_like(nnodes, device=device, dtype=torch.float)
        is_d_pk[S_k_idx] = d_pk
        
        ### def select_rows()
        # adj_s_k = _select_rows(adj_sp, S_k_idx)
        _col_len = len(nnodes)
        _mask = torch.isin(adj_row, S_k_idx)
        _row_m = adj_row[_mask]
        _new_r = torch.zeros_like(_row_m)
        for idx, _r in enumerate(S_k_idx):
            _new_r += idx * torch.isin(_row_m, _r).to(torch.int)        
        _new_c = adj_col[_mask]
        _new_v = adj_val[_mask]
        adj_s_k = torch.sparse_coo_tensor(torch.stack((_new_r, _new_c)), _new_v, size=(len(S_k_idx), _col_len))
        ### def ends
        
        adj_S_k = adj_s_k * S_k

        is_d_pk_sp = adj_s_k * is_d_pk
        out_degree_sp = (adj_S_k * out_degree)
        out_degree_sp = 1e-12*torch.ones(out_degree_sp.size(), dtype=torch.float, device=device) + out_degree_sp
        
        tmp_sum = torch.sum(is_d_pk_sp.to_dense() / out_degree_sp.to_dense(), dim=1)

        d_fp_new[S_k_idx] = (1 - 1. / out_degree[S_k_idx]) * d_fp_old[S_k_idx] - \
                        rho * alpha / out_degree[S_k_idx] - \
                        0.5 * (1-alpha) * is_d_pk[S_k_idx] /  out_degree[S_k_idx] -  \
                        0.5 * (1-alpha) / out_degree[S_k_idx] * tmp_sum
        
        neighbors = (adj_s_k * (~S_k).to(torch.int)).to_dense()
        neighbors_set = torch.sum(neighbors > 0, dim=1).to(torch.int)
        neighbors_idx = neighbors_set.nonzero()

        if len(neighbors_idx) != 0:
            # def select_rows()
            # adj_ngbr = _select_rows(adj_sp, neighbors_idx, device)
            _mask = torch.isin(adj_row, neighbors_idx)
            _row_m = adj_row[_mask]
            _new_r = torch.zeros_like(_row_m)
            for idx, _r in enumerate(neighbors_idx):
                _new_r += idx * torch.isin(_row_m, _r).to(torch.int)
            _new_c = adj_col[_mask]
            _new_v = adj_val[_mask]

            adj_ngbr = torch.sparse_coo_tensor(torch.stack((_new_r, _new_c)), _new_v, size=(len(neighbors_idx), _col_len))            
            #def ends

            adj_ngbr_S_k = adj_ngbr * S_k
            is_d_pk_sp = adj_ngbr_S_k * is_d_pk
            out_degree_sp =  (adj_ngbr * out_degree)
            out_degree_sp = 1e-12*torch.ones(out_degree_sp.size(), dtype=torch.float, device=device) + out_degree_sp
            tmp_sum = torch.sum(is_d_pk_sp.to_dense() / out_degree_sp.to_dense(), dim=1)

            d_fp_new[neighbors_idx] = d_fp_old[neighbors_idx] - 0.5 * (1 - alpha) / out_degree[neighbors_idx] * tmp_sum

        d_fp_old = d_fp_new
        p_old=p_new

    return p_old

def get_ppr_node_ista_parallel(nnodes,
                      alpha,
                      epsilon,
                      rho,
                      out_degree,
                      node_index,
                      deg_inv,
                      adj,
                      device):
    """
    nnodes:     nodes (from adjacent matrix)
    alpha:      hyperparameter
    epsilon:    hyperparameter
    rho:        hyperparameter
    out_degree: out degree from adjacent matrix
    node_index: target node (of calculating PPR)
    deg_inv:    inverse degree
    adj:        Adjacency matrix in dglsp
    device:     device to process

    Parallelize ISTA algorithm via 3D matrix
    dim 0: nodes
    dim 1: variable (1, S_k, etc)
    dim 2: size of nnodes    
    """

    indices = adj.indices()
    val = adj.val.to(torch.int)
    adj_sp = dglsp.spmatrix(indices, val).to(device)
    
    out_degree = out_degree.to(device)
    
    _indices = torch.arange()

def _get_ppr_node_ista_parallel(nnodes,
                               alpha,
                               epsilon,
                               rho,
                               out_degree,
                               node_index,
                               deg_inv,
                               adj,
                               device):
    """
    Parallelizing get_node_ista

    nnodes:     nodes (from adjacent matrix)
    alpha:      hyperparameter
    epsilon:    hyperparameter
    rho:        hyperparameter
    out_degree: out degree from adjacent matrix
    node_index: target nodes (of calculating PPR)
    deg_inv:    inverse degree
    indices:    CSR indices of adjacent matrix
    indptr:     CSR indptr of adjacent matrix
    device:     device to process
    """


    p_old = torch.zeros((len(node_index), len(nnodes)), dtype=torch.float).to(device)
    p_new = torch.zeros((len(node_index), len(nnodes)), dtype=torch.float).to(device)
    d_fp_old = torch.sparse_coo_tensor(indices=torch.stack((torch.arange(len(node_index)), node_index)),
                                       values=deg_inv[node_index],
                                       size=(len(node_index), len(nnodes))).to(device).to_dense()

    global_fp_norm = torch.linalg.vector_norm(d_fp_old, float("inf"), dim=1)
    cont = global_fp_norm > (1+epsilon) * rho * alpha
    while cont.any() :  

        S_k = (p_old - d_fp_old) >= rho*alpha # S_k : boolean matrix
        d_pk = -(d_fp_old + rho  * alpha) * S_k # masking using S_k
        p_new = (p_old + d_pk) * S_k
        d_fp_new = d_fp_old

        is_d_pk = torch.zeros_like(p_old, device=device).to(torch.float)
        is_d_pk = is_d_pk + d_pk * S_k

        # indptr[i]:indptr[i+1]: 해당 row에 있는 노드들
        # i의 outgoing nodes 들의 is_d_pk를 더해야함..
        
        # 1. Change is_d_pk to dgl.sparseMatrix
        # 2. adj @ is_d_pk.T, didivde
        # 3. S_k maksing, sum

        is_d_pk_sp = to_dglsp(is_d_pk).to(device)
        out_degree_sp = to_dglsp(out_degree).to(device)        
        adj_S_k = dglsp.matmul(adj, to_dglsp(S_k.to(torch.int)))
        _is_d_pk_sp = dglsp.matmul(adj_S_k, is_d_pk_sp).to_dense()
        _out_degree_sp = dglsp.sp_add_v(dglsp.matmul(adj_S_k, out_degree_sp), 1e-12*torch.ones(len(nnodes)))
        _tmp_sum = torch.div(_tmp_sum, out_degree)
        tmp_sum = torch.sum(torch.mul(tmp_sum, S_k), dim=1, keepdim=True)

        _d_fp_new = (1 - 1. / out_degree) * d_fp_old  - \
                    rho * alpha / out_degree - \
                    0.5 * (1 - alpha) * is_d_pk / out_degree - \
                    0.5 * (1 - alpha) / out_degree * tmp_sum
        d_fp_new = d_fp_new * ~S_k  + _d_fp_new * S_k

@numba.njit(cache=True)
def get_ppr_node_ista_numpy(nnodes,
                            alpha,
                            epsilon,
                            rho,
                            out_degree,
                            node_index,
                            deg_inv,
                            indices,
                            indptr,
                            ):
    s_vector = np.zeros(nnodes)
    s_vector[node_index] = 1

    p_vector_old = np.zeros(nnodes)
    p_vector_new = np.zeros(nnodes)

    delta_fp_old = np.zeros(nnodes)
    delta_fp_old[node_index] = -alpha * deg_inv[node_index]
    delta_fp_shape = delta_fp_old.shape

    while np.linalg.norm(delta_fp_old, np.inf) > (1 + epsilon) * rho * alpha:
        S_k = np.where(p_vector_old - delta_fp_old >= rho * alpha)[0]
        Sk_list = list(S_k)
        delta_pk = -(delta_fp_old[S_k] + rho * alpha)

        p_vector_new[S_k] = p_vector_old[S_k] + delta_pk 

        delta_fp_new = delta_fp_old

        Is_delta_pk = np.zeros(nnodes)
        Is_delta_pk[S_k] = delta_pk
        for i in S_k:
            tmp_sum = 0
            for l in indices[indptr[i]:indptr[i + 1]]:
                if l in Sk_list:
                    tmp_sum += Is_delta_pk[l] / out_degree[l]

            delta_fp_new[i] = (1 - 1. / out_degree[i]) * delta_fp_old[i] - \
                              rho * alpha / out_degree[i] - \
                              0.5 * (1 - alpha) * Is_delta_pk[i] / out_degree[i] - \
                              0.5 * (1 - alpha) / out_degree[i] * tmp_sum

        neighbors_set = []
        for s in S_k:
            for neighbor in indices[indptr[s]:indptr[s + 1]]:
                if neighbor not in neighbors_set:
                    if neighbor not in Sk_list:
                        neighbors_set.append(neighbor)
        for j in neighbors_set:
            tmp_sum = 0
            for l in indices[indptr[j]:indptr[j + 1]]:
                if l in Sk_list:
                    tmp_sum += Is_delta_pk[l] / out_degree[l]

            delta_fp_new[j] = delta_fp_old[j] - 0.5 * (1 - alpha) / out_degree[j] * tmp_sum

        delta_fp_old = delta_fp_new
        p_vector_old = p_vector_new 

    return p_vector_old


def get_ppr(block,
            graph,
            alpha,
            epsilon,
            rho,
            ul_link,
            device,
            topk=None,
            adj_data=None
            ):
    """
    Get PPR of nodes in blocks. Calculate PPR vector of dstnodes of block,
    return corresponding ppr value of srcnodes (1st neighbor of dstnodes).
    When obtaining ppr value of a srcnode, average value of ppr vector of dstnodes who have edges to srcnodes only.

    block:      DGLBlock of nodes for calculating PPR
    graph:      DGlGraph
    alpha:      hyperparameter for ISTA algo
    epsilon:    hyperparameter for ISTA algo
    rho:        hyperparameter for ISTA algo
    device:     device
    topk:       (optional) select TopK PPRs for each nodes
    adj_data:   (optional) Edge attribute for calculating PPR
    """

    # Extract graph info
    nodes = block.dstdata["_ID"]
    targets = block.srcdata["_ID"]

    nnodes = graph.nodes()
    out_degree = graph.out_degrees()
    degree = graph.out_degrees()
    deg_inv = 1. / torch.maximum(degree.to(device), 1e-12*torch.ones_like(degree, device=device))
    adj = graph.adj()
    indices = adj.indices()
    val = adj.val()

    ppr_vec = list()
    for idx, node_idx in enumerate(nodes):
        # ppr = get_ppr_node_ista_p(nnodes=nnodes,
        #                         alpha=alpha,
        #                         epsilon=epsilon,
        #                         rho=rho,
        #                         out_degree=out_degree,
        #                         node_index=node_idx,
        #                         deg_inv=deg_inv,
        #                         indices=indices,
        #                         indptr=indptr,
        #                         device=device)
        ppr = get_ppr_node_ista_p(nnodes=nnodes,
                                alpha=alpha,
                                epsilon=epsilon,
                                rho=rho,
                                out_degree=out_degree,
                                node_index=node_idx,
                                deg_inv=deg_inv,
                                adj_indices=indices,
                                adj_val=val,
                                device=device)
        
        # Need to find the neighbor of idx
        mask = torch.isin(torch.arange(len(ppr), device=device),
                          torch.where(ul_link[idx] == True)[0]).to(device)
        # print(ppr.nonzero())
        # print(sum(ppr))
        # print(ppr.shape)
        ppr /= sum(ppr)
        ppr *= mask.to(device)

        block.edges()
        ppr_vec.append(ppr)

    ppr_vec = torch.stack(ppr_vec)
    ppr_vec.mean(dim=1)

    return ppr_vec

def get_rppr(nodes,
             target,
             graph,
             alpha,
             epsilon,
             rho,
             ul_link,
             device,
             topk=None,
             adj_data=None
             ):
    """
    Get Reverse-ppr of target node from given nodes

    nodes:      indices of nodes to calculate PPR
    target:     index of target node of ppr
    graph:      graph,
    alpha:      hyperparameter for ISTA algo
    epsilon:    hyperparameter for ISTA algo
    rho:        hyperparameter for ISTA algo
    device:     device
    topk:       (optional) select TopK PPRs for each nodes
    adj_data:   (optional) Edge attribute for calculating PPR
    """

    # Extract graph info
    nnodes = graph.nodes()
    out_degree = graph.out_degrees()
    degree = graph.out_degrees()
    deg_inv = 1. / torch.maximum(degree, 1e-12*torch.ones_like(degree, device=device))
    adj = graph.adj()

    print(nodes.shape, ul_link.shape)

    ppr_vec = list()
    for idx, node_idx in enumerate(nodes):
        ppr = get_ppr_node_ista_p(nnodes=nnodes,
                                alpha=alpha,
                                epsilon=epsilon,
                                rho=rho,
                                out_degree=out_degree,
                                node_index=node_idx,
                                deg_inv=deg_inv,
                                adj=adj,
                                device=device)
        ppr /= sum(ppr)
        print(ppr.nonzero())
        mask = torch.isin(torch.arange(len(ppr), device=device), torch.where(ul_link[idx] == 1)[0])
        ppr = torch.sum(ppr*mask)
        ppr_vec.append(ppr)
    
    print(ppr_vec)
    
    ppr_vec = torch.cat(ppr_vec)

    return ppr_vec

# Modulized
class Get_PPR(nn.Module):

    def __init__(self,
                 graph,
                 alpha,
                 epsilon,
                 rho,
                 device
                 ):
        super(Get_PPR, self).__init__()
        self.graph = graph
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho

        self.nnodes = self.graph.nodes()
        self.out_degree = self.graph.out_degrees()
        self.deg_inv = 1. / torch.maximum(self.out_degree.to(device),
                                          1e-12*torch.ones_like(self.out_degree, device=device))
        self.adj = graph.adj()
        self.device = device


    def forward(self, block:DGLBlock, ul_link:torch.Tensor):
        
        nodes = block.dstdata[dgl.NID]
        targets = block.srcdata[dgl.NID]
        row, col = self.adj.coo()
        val = self.adj.val

        ppr_vec = list()
        for idx, node_idx in enumerate(nodes):
            ppr = get_ppr_node_ista_p_torch_only(nnodes=self.nnodes,
                                      alpha=self.alpha,
                                      epsilon=self.epsilon,
                                      rho=self.rho,
                                      out_degree=self.out_degree,
                                      node_index=node_idx,
                                      deg_inv=self.deg_inv,
                                      adj_row=row,
                                      adj_col=col,
                                      adj_val=val,
                                      device=self.device)
            # Need to find the neighbor of idx
            # mask = torch.isin(torch.arange(len(ppr), device=self.device),
            #                 torch.where(ul_link[idx] == True)[0]).to(self.device)
            # # print(ppr.nonzero())
            # # print(sum(ppr))
            # # print(ppr.shape)
            # ppr /= sum(ppr)
            # ppr *= mask.to(self.device)

            # block.edges()
            # ppr_vec.append(ppr)
            ppr_vec.append(ppr.nonzero())


        # ppr_vec = torch.stack(ppr_vec)
        # ppr_vec.mean(dim=1)

        return ppr_vec


class Get_RPPR(nn.Module):

    def __init__(self,
                 graph,
                 alpha,
                 epsilon,
                 rho,
                 device
                 ):
        super(Get_PPR, self).__init__()
        self.graph = graph
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho

        self.nnodes = self.graph.nodes()
        self.out_degree = self.graph.out_degrees()
        self.deg_inv = 1. / torch.maximum(self.out_degree.to(device),
                                          1e-12*torch.ones_like(self.out_degree, device=device))
        self.adj = graph.adj()
        self.device = device


    def forward(self, block:DGLBlock, target:Tensor, ul_link:torch.Tensor):
        
        nodes = block.dstdata[dgl.NID]
        targets = block.srcdata[dgl.NID]
        row, col = self.adj.coo()
        val = self.adj.val

        ppr_vec = list()
        for idx, node_idx in enumerate(nodes):
            ppr = get_ppr_node_ista_p_torch_only(nnodes=self.nnodes,
                                      alpha=self.alpha,
                                      epsilon=self.epsilon,
                                      rho=self.rho,
                                      out_degree=self.out_degree,
                                      node_index=node_idx,
                                      deg_inv=self.deg_inv,
                                      adj_row=row,
                                      adj_col=col,
                                      adj_val=val,
                                      device=self.device)
            # Need to find the neighbor of idx
            mask = torch.isin(torch.arange(len(ppr), device=self.device),
                            torch.where(ul_link[idx] == True)[0]).to(self.device)
            # print(ppr.nonzero())
            # print(sum(ppr))
            # print(ppr.shape)
            ppr /= sum(ppr)
            ppr *= mask.to(self.device)

            block.edges()
            ppr_vec.append(ppr)

        ppr_vec = torch.stack(ppr_vec)
        ppr_vec.mean(dim=1)

        return ppr_vec


def get_ppr_np(block,
               graph,
               alpha,
               epsilon,
               rho,
               ul_link,
               device):
    
    nodes = block.dstdata["_ID"]
    targets = block.srcdata["_ID"]
    nnodes = len(graph.nodes())
    out_degree = graph.out_degrees().cpu()
    deg_inv = 1. / torch.maximum(out_degree, 1e-12*torch.ones_like(out_degree))
    indptr, indices, values = graph.adj().csr()
    ppr_vec = list()
    
    for idx, node_idx in enumerate(nodes):
        ppr = get_ppr_node_ista_numpy(nnodes=nnodes,
                                      alpha=alpha,
                                      epsilon=epsilon,
                                      rho=rho,
                                      out_degree=out_degree.numpy(),
                                      deg_inv=deg_inv.numpy(),
                                      node_index=node_idx.item(),
                                      indptr=indptr.cpu().numpy(),
                                      indices=indices.cpu().numpy())
        ppr /= sum(ppr)
        ppr_vec.append(ppr)
    
    ppr_vec = torch.from_numpy(np.stack(ppr_vec)).to(device)
    select = ppr_vec[:, targets]
    # print("select ", select.shape)
    select *= ul_link
    return select


def get_rppr_np(block,
               graph,
               target,
               alpha,
               epsilon,
               rho,
               ul_link,
               device):
    
    nodes = block.dstdata["_ID"]
    nnodes = len(graph.nodes())
    out_degree = graph.out_degrees().cpu()
    deg_inv = 1. / torch.maximum(out_degree, 1e-12*torch.ones_like(out_degree))
    indptr, indices, values = graph.adj().csr()
    ppr_vec = list()
    
    for idx, node_idx in enumerate(nodes):
        ppr = get_ppr_node_ista_numpy(nnodes=nnodes,
                                      alpha=alpha,
                                      epsilon=epsilon,
                                      rho=rho,
                                      out_degree=out_degree.numpy(),
                                      deg_inv=deg_inv.numpy(),
                                      node_index=node_idx.item(),
                                      indptr=indptr.cpu().numpy(),
                                      indices=indices.cpu().numpy())
        
        ppr /= sum(ppr)
        ppr_vec.append(ppr)
    ppr_vec = torch.from_numpy(np.stack(ppr_vec)).to(device)
    select = ppr_vec[:,target]
    # print(select.shape, ul_link.shape)
    select *= ul_link
    
    return select