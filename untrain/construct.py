import os
import sys
import time
import wandb
import torch
import logging
import numpy as np
from datetime import datetime
sys.path.append(os.getcwd())
from core.args import Untrain_Parser
from core.data import Graph_Load_from, Graph_Unlearn_Dataloader
from core.data import Graph_Remove_Unlearnables
from core.model import Unlearn_Model_Loader
from core.trainer import Untrainer_Loader
from core.utils import Checkpoint_Loader
from core.data import Graph_save


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def compute_homogeneity(graph, unlearn_mask, labels):
    """
    计算每个被遗忘节点的邻居的标签同质性比例，并输出邻居数量。
    :param graph: 图数据
    :param unlearn_mask: 遗忘节点的mask (bool型)
    :param labels: 节点标签
    :return: 各个被遗忘节点的同质性比例和邻居数量
    """
    homogeneity = []
    
    # 获取所有被遗忘的节点
    unlearn_nodes = torch.nonzero(unlearn_mask).squeeze()

    for node in unlearn_nodes:
        # 获取当前节点的所有后继节点 (邻居)
        neighbors = graph.successors(node)

        num_neighbors = len(neighbors)  # 邻居数量
        if num_neighbors == 0:
            # 如果没有邻居，跳过该节点
            homogeneity.append((0.0, 0))  # 同质性比例为0，邻居数量为0
            continue
        
        # 获取邻居节点的标签
        neighbor_labels = labels[neighbors]
        
        # 当前节点的标签
        node_label = labels[node]
        
        # 计算同质性比例: 邻居中与该节点标签相同的比例
        same_label_count = torch.sum(neighbor_labels == node_label).item()
        homogeneity_ratio = same_label_count / num_neighbors
        
        # 记录同质性比例和邻居数量
        homogeneity.append((homogeneity_ratio, num_neighbors))
    
    return homogeneity


def run(args):
    run_name = args.model + "_" \
               + args.dataset + "_" \
               + str(args.epochs) + "_" \
               + args.unlearn_method + "_" \
               + time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.wandb_run_name != "":
        run_name = args.wandb_run_name + "_" + run_name

    _wandb = wandb.init(entity="user_name",
                        project=args.wandb_project,
                        config=args,
                        name=run_name)

    # Seed setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Configure save path
    save_path = os.path.join(args.save_path, args.wandb_project, _wandb.name)
    os.makedirs(save_path, exist_ok=True)

    # Load trained graph
    ckpt, splits = Checkpoint_Loader(args)

    graph = Graph_Load_from(args)
    # 遗忘
    ul_graph = Graph_Remove_Unlearnables(args, graph)
    print(ul_graph)
    src, dst = ul_graph.edges()
    # for s, d in zip(src.tolist(), dst.tolist()):
    # print(f"{s} -> {d}")
    labels = ul_graph.ndata['label']
    print(labels)

    unlearn_mask = ul_graph.ndata['unlearn_mask']
    
    # 计算每个被遗忘节点的同质性比例和邻居数量
    homogeneity = compute_homogeneity(ul_graph, unlearn_mask, labels)
    print("Homogeneity and neighbor count of unlearned nodes:")
    for node_idx, (ratio, num_neighbors) in enumerate(homogeneity):
        print(f"Node {node_idx} homogeneity ratio: {ratio:.4f}, Neighbors count: {num_neighbors}")

    dataloaders = Graph_Unlearn_Dataloader(args, ul_graph, graph)
    print(f"[INFO] Unlearn nodes total in dataloader: {len(dataloaders['unlearn'].indices)}")
    Graph_save(save_path, ul_graph)

    for k, v in dataloaders.items():
        print(k, len(v))

    _wandb.finish()


if __name__ == "__main__":
    parser = Untrain_Parser()
    args = parser.parse_args()
    run(args)
