import dgl
import torch
import dgl.sparse as dglsp

def to_dglsp(mat, device="cpu"):
    mat = mat.to_sparse()
    shape = list(mat.shape)
    if len(shape) == 1:
        shape.insert(0, 1)
    indices = mat.indices()
    try:
        row, col = indices[0], indices[1]
    except:
        # single row
        col = indices[0]
        row = torch.zeros_like(col)
    val = mat.values()
    
    sp_mat = dglsp.from_coo(row, col, val, shape)
    return sp_mat.to(device)

def to_torch_dense(sp_mat, device="cpu"):
    
    indptr, indices, values = sp_mat.csr()

    mat = torch.sparse_csr_tensor(indptr, indices, values, device=device)
    mat = mat.to_dense()
    mat = mat.to(device)

    return mat

def select_rows(mat, rows, device="cpu"):
    row, col = mat.coo()
    val = mat.val
    col_len = mat.shape[1]
    mask = torch.isin(row, rows),
    row_m = row[mask]
    new_r = torch.zeros_like(row_m)

    for idx, r in enumerate(rows):
        new_r += idx * torch.isin(row_m, r).to(torch.int)

    new_c = col[mask]
    new_v = val[mask]
    new_sp = dglsp.from_coo(new_r, new_c, new_v, shape=(len(rows), col_len))
    return new_sp.to(device)

def duplicate_rows(mat, n, device="cpu"):
    row, col = mat.coo()
    val = mat.val
    col_len = mat.shape[1]
    new_r = [row + i for i in range(n)]
    new_r = torch.cat(new_r)
    new_c = col.repeat(n)
    new_val = val.repeat(n)
    new_sp = dglsp.from_coo(new_r, new_c, new_val, shape=(n, col_len))
    return new_sp.to(device)    

