import torch
import dgl

def generate_random_walk_embeddings(graph, walk_length=10, walks_per_node=10, embedding_dim=None, device='cuda'):
    """
    基于 DGL 内置随机游走生成节点结构编码。
    
    Args:
        graph (dgl.DGLGraph): 输入图，必须包含 'label' 和 'unlearn_mask' 节点属性
        walk_length (int): 每次随机游走长度
        walks_per_node (int): 每个节点进行游走次数
        embedding_dim (int or None): 结构编码维度，如果为 None，则使用类别数
        device (str or torch.device): 输出特征所在设备（GPU/CPU）
    
    Returns:
        torch.Tensor: 节点结构编码 [num_nodes, embedding_dim]，在指定 device 上
    """
    # 设备设置
    device = torch.device(device)
    
    unlearn_mask = graph.ndata['unlearn_mask']  # 遗忘节点 mask
    unlearn_nodes = torch.nonzero(unlearn_mask, as_tuple=True)[0]  # 遗忘节点索引
    num_nodes = graph.num_nodes()
    
    # 默认 embedding_dim 为图的类别数
    if embedding_dim is None:
        embedding_dim = int(graph.ndata['label'].max().item()) + 1
    
    # 初始化节点结构编码
    node_embeddings = torch.zeros((num_nodes, embedding_dim), dtype=torch.float32, device=device)
    
    # 随机游走生成编码
    for _ in range(walks_per_node):
        traces, _ = dgl.sampling.random_walk(graph, unlearn_nodes, length=walk_length)
        
        for idx, node in enumerate(unlearn_nodes):
            walk_nodes = traces[idx]  # 当前节点所有游走节点
            walk_nodes = walk_nodes[walk_nodes >= 0]  # 去掉 -1
            # CPU 索引 -> GPU
            labels = graph.ndata['label'][walk_nodes].to(device)
            counts = torch.bincount(labels, minlength=embedding_dim).float()
            node_embeddings[node] += counts
    
    # 归一化
    node_embeddings = node_embeddings / node_embeddings.sum(dim=1, keepdim=True).clamp(min=1e-6)
    
    return node_embeddings
