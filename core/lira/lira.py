import os
import sys
import time
import wandb
import torch
import copy
import numpy as np
import torch.nn.functional as F
sys.path.append(os.getcwd())
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from core.data import Graph_Load_from
from core.model import Model_Loader
from core.args import LiRA_Parser
from dgl.dataloading import DataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler
import dgl.sparse as dglsp
from core.model.gcn import GraphConvNet



def metric(score, x):
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc    

class LIRA:
    
    def __init__(self, args, graph):
        self.device=torch.device(args.device)
        num_classes = len(torch.unique(graph.ndata["label"]))
        sampler = MultiLayerFullNeighborSampler(args.depth)
        self.args = args
        self.graph = graph
        
        if args.use_same_size:
            unlearn_indices = torch.where(graph.ndata["unlearn_mask"] == True)[0]
            retain_indices = torch.where(graph.ndata["retain_mask"] == True)[0]
            test_indices = torch.where(graph.ndata["test_mask"] == True)[0]
            num_samples = min(len(unlearn_indices), min(len(test_indices), len(retain_indices)))
            self.unlearn_indices = unlearn_indices
            self.retain_indices = retain_indices
            self.test_indices = test_indices
            self.num_samples = [num_samples, num_samples, num_samples]

        else:
            unlearn_indices = torch.where(graph.ndata["unlearn_mask"] == True)[0]
            retain_indices = torch.where(graph.ndata["retain_mask"] == True)[0]
            test_indices = torch.where(graph.ndata["test_mask"] == True)[0]
            self.unlearn_indices = unlearn_indices
            self.retain_indices = retain_indices
            self.test_indices = test_indices
            self.num_samples = [len(unlearn_indices), len(test_indices), len(retain_indices)]

        self.unlearn_loader = DataLoader(graph=graph,
                                    batch_size=args.batch_size,
                                    indices=unlearn_indices[:self.num_samples[0]],
                                    graph_sampler=sampler,
                                    device=torch.device(args.device),
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=0)
    
        self.test_loader = DataLoader(graph=graph,
                                    batch_size=args.batch_size,
                                    indices=test_indices[:self.num_samples[1]],
                                    graph_sampler=sampler,
                                    device=torch.device(args.device),
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=0)
        
        self.retain_loader = DataLoader(graph=graph,
                                    batch_size=args.batch_size,
                                    indices=retain_indices[:self.num_samples[2]],
                                    graph_sampler=sampler,
                                    device=torch.device(args.device),
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=0)

        # Prepare shadow model backbone
        args.batchnorm = True
        # self.shadow_model_backbone = GraphConvNet(args,
        #                                         in_dim=graph.ndata["feat"].size(1),
        #                                         hidden_dim=args.latent_size,
        #                                         num_classes=num_classes)
        args.batchnorm = True
        self.shadow_model_backbone = Model_Loader(args)(args,
                                                        in_dim=graph.ndata["feat"].size(1),
                                                        hidden_dim=args.latent_size,
                                                        num_classes=num_classes)

    
        total_scores = list()
        total_keep = list()
        for shadow_models_path in os.listdir(args.shadow_path):
            scores, keep= self.get_shadow_model_score(shadow_models_path)
            total_scores.append(scores)
            total_keep.append(keep)
        total_scores = np.array(total_scores)
        total_keep = np.array(total_keep)

        print(total_scores.shape, np.max(total_scores[0]), np.min(total_scores[0]))

        in_scores, out_scores = self.get_in_out_scores(total_keep, total_scores)

        print("in_scores", in_scores.shape)
        print("out_scores", out_scores.shape)
        self.in_scores = in_scores
        self.out_scores = out_scores

    
        retain_labels = np.ones(self.num_samples[2], dtype=np.bool_)
        test_labels = np.zeros(self.num_samples[1], dtype=np.bool_)
        unlearn_labels = np.ones(self.num_samples[0], dtype=np.bool_)

        #用于区分的二元分类标签
        self.labels = np.concatenate((retain_labels, test_labels)).astype(np.bool_)
        self.u_labels = np.concatenate((unlearn_labels, test_labels)).astype(np.bool_)



    def get_shadow_model_score(self, shadow_model_path):

        load_path = os.path.join(self.args.shadow_path, shadow_model_path, "last", "model.pt")
        checkpoint = torch.load(load_path)

        self.shadow_model_backbone.load_state_dict(checkpoint["state_dict"])
        self.shadow_model_backbone.eval()
        self.shadow_model_backbone.to(self.device)
        sampler = MultiLayerFullNeighborSampler(self.args.depth)

        shadow_train_mask = checkpoint["keep"]
        train_loader = DataLoader(graph=self.graph,
                                  batch_size=self.args.batch_size,
                                  indices=torch.arange(self.graph.num_nodes()),
                                  graph_sampler=sampler,
                                  device=self.device,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=0)
        
        # Get confidence from shadow model
        # Run multiple queries
        total_logits = list()

        for _ in range(self.args.n_queries):
            _logits = list()
            for _, (in_nodes, out_nodes, blocks) in enumerate(train_loader):
                blocks = [b.to(self.device) for b in blocks]
                data = blocks[0].srcdata["feat"].to(self.device)
                labels = blocks[-1].dstdata["label"].to(self.device)
                _pred, feat = self.shadow_model_backbone(blocks, data)
                _logits.extend(_pred.cpu().detach().numpy())
            logits = np.stack(_logits, axis=0)
            total_logits.append(logits)
        total_logits = np.stack(total_logits, axis=1)

        # Get scores

        scores = self.get_scores(total_logits, self.graph.ndata["label"].cpu().detach().numpy()) 

        return scores, shadow_train_mask

    def get_in_out_scores(self, keep, scores):

        """
        in_mask: size: (num_shadow, num_nodes)
        out_mask: size: (num_shadow, num_nodes)
        scores: size: (num_shadow, num_samples, num_queries)
        """

        dat_in = list()
        dat_out = list()
        for j in range(scores.shape[1]):
            dat_in.append(scores[keep[:, j], j, :]) 
            dat_out.append(scores[~keep[:, j], j, :])

        in_size = min(min(map(len, dat_in)), self.graph.num_nodes())
        out_size = min(min(map(len, dat_out)), self.graph.num_nodes())

        in_scores = np.array([x[:in_size] for x in dat_in])
        out_scores = np.array([x[:out_size] for x in dat_out])

        return in_scores, out_scores

    def get_confidence(self, model, device):
        model.eval()
        model.to(device)
        unlearn_logits = list()
        retain_logits = list()
        test_logits = list()

        for _ in range(self.args.n_queries):
            _unlearn_logits = list()
            for _, (in_nodes, out_nodes, blocks) in enumerate(self.unlearn_loader):
                blocks = [b.to(device) for b in blocks]
                data = blocks[0].srcdata["feat"].to(device)
                labels = blocks[-1].dstdata["label"].to(device)
                _pred, feat = model(blocks, data)
                _unlearn_logits.extend(_pred.cpu().detach().numpy())
            unlearn_logits.append(np.stack(_unlearn_logits))
        unlearn_logits = np.stack(unlearn_logits, axis=1)   

        for _ in range(self.args.n_queries):
            _retain_logits = list()
            for _, (in_nodes, out_nodes, blocks) in enumerate(self.retain_loader):
                blocks = [b.to(device) for b in blocks]
                data = blocks[0].srcdata["feat"].to(device)
                labels = blocks[-1].dstdata["label"].to(device)
                _pred, feat = model(blocks, data)
                _retain_logits.extend(_pred.cpu().detach().numpy())
            retain_logits.append(np.stack(_retain_logits))
        retain_logits = np.stack(retain_logits, axis=1)
        
        for _ in range(self.args.n_queries):
            _test_logits = list()
            for _, (in_nodes, out_nodes, blocks) in enumerate(self.test_loader):
                blocks = [b.to(device) for b in blocks]
                data = blocks[0].srcdata["feat"].to(device)
                labels = blocks[-1].dstdata["label"].to(device) 
                _pred, feat = model(blocks, data)
                _test_logits.extend(_pred.cpu().detach().numpy())
            test_logits.append(np.stack(_test_logits))
        test_logits = np.stack(test_logits, axis=1)
        
        return unlearn_logits, retain_logits, test_logits

    def get_scores_orig(self, logits, labels):
        predictions = logits - np.max(logits, axis=-1, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
        
        COUNT = predictions.shape[0]
        y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]
        predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=-1)
        score = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)

        return score
    
    def get_scores(self, logits, labels):
        """
        Calculate likelihood ratio scores with improved numerical stability.
        
        Args:
            logits: Raw model outputs (before softmax)
            labels: True class labels
        Returns:
            score: Likelihood ratio scores
        """
        # Constants for numerical stability
        EPSILON = 1e-15
        MIN_VAL = -1e7
        MAX_VAL = 1e7
        
        # 1. Clip logits to prevent extreme values
        logits = np.clip(logits, MIN_VAL, MAX_VAL)
        
        # 2. Log-sum-exp trick for softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
        predictions = exp_logits / (sum_exp_logits + EPSILON)
        
        # 3. Ensure predictions are in valid range
        predictions = np.clip(predictions, EPSILON, 1.0)
        
        COUNT = predictions.shape[0]
        
        # 4. Calculate true class probabilities
        y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]
        y_true = np.clip(y_true, EPSILON, 1.0)
        
        # 5. Calculate wrong class probabilities
        predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=-1)
        y_wrong = np.clip(y_wrong, EPSILON, 1.0)
        
        # 6. Compute log likelihood ratio
        score = np.log(y_true) - np.log(y_wrong)
        
        return score


    def predict(self, logits, indices):

        """
        Need to select indices of in & out scores
        """
        print("indices:", indices)
        print("self.in_scores:", self.in_scores)
        print("self.out_scores:", self.out_scores)

        mean_in = np.median(self.in_scores, 1)[indices]
        mean_out = np.median(self.out_scores, 1)[indices]
        std_in = np.std(self.in_scores, 1)[indices]
        std_out = np.std(self.out_scores, 1)[indices]

        print("logits.shape", logits.shape)
       
        pr_in = -scipy.stats.norm.logpdf(logits, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(logits, mean_out, std_out+1e-30)
        score = pr_in-pr_out
        return score.mean(1)

    def test_model(self, model, name):
        model.eval()
        model.to(self.device)

        # Get scores
        _unlearn_node_labels = self.graph.ndata["label"][self.unlearn_indices[:self.num_samples[0]]].cpu().detach().numpy()
        _retain_node_labels = self.graph.ndata["label"][self.retain_indices[:self.num_samples[2]]].cpu().detach().numpy()
        _test_node_labels = self.graph.ndata["label"][self.test_indices[:self.num_samples[1]]].cpu().detach().numpy()

        unlearn_conf, retain_conf, test_conf = self.get_confidence(model, self.device)
        
        unlearn_scores = self.get_scores(unlearn_conf, _unlearn_node_labels)
        retain_scores = self.get_scores(retain_conf, _retain_node_labels)
        test_scores = self.get_scores(test_conf, _test_node_labels)


        # Get prediction（confidence score）
        unlearn_pred = self.predict(unlearn_scores, self.unlearn_indices[:self.num_samples[0]])
        unlearn_pred_rounded = np.round(unlearn_pred, 4)
        print("unlearn_pred", unlearn_pred)
        print(unlearn_pred.shape)
        retain_pred = self.predict(retain_scores, self.retain_indices[:self.num_samples[2]])
        retain_pred_rounded = np.round(retain_pred, 4)
        test_pred = self.predict(test_scores, self.test_indices[:self.num_samples[1]])
        test_pred_rounded = np.round(test_pred, 4)


        # Get AUC
        unlearn_score = np.concatenate((unlearn_pred, test_pred))       
        retain_score = np.concatenate((retain_pred, test_pred))

        unlearn_fpr, unlearn_tpr, unlearn_auc, unlearn_acc = metric(unlearn_score, self.u_labels)
        retain_fpr, retain_tpr, retain_auc, retain_acc = metric(retain_score, self.labels)

        unlearn_auc = [unlearn_fpr, unlearn_tpr, unlearn_auc, unlearn_acc]
        retain_auc = [retain_fpr, retain_tpr, retain_auc, retain_acc]

        #print_statistics(unlearn_pred_rounded, "unlearn_pred")
        #print_statistics(retain_pred_rounded, "retain_pred")
        #print_statistics(test_pred_rounded, "test_pred")
        ul_graph = self.graph
        labels = ul_graph.ndata['label']
        unlearn_mask = ul_graph.ndata['unlearn_mask']
        homogeneity = compute_homogeneity(ul_graph, unlearn_mask, labels)
        print("Homogeneity and neighbor count of unlearned nodes:")
        for node_idx, (ratio, num_neighbors) in enumerate(homogeneity):
            unlearn_pred_for_node = unlearn_pred_rounded[node_idx]
            print(f"Node {node_idx} homogeneity ratio: {ratio:.4f}, Neighbors count: {num_neighbors}, Prediction: {unlearn_pred_for_node:.4f}")

        return unlearn_auc, retain_auc

def print_statistics(pred_array, name):
    print(f"{name} Statistics:")
    print(f"  Max: {np.max(pred_array)}")
    print(f"  Min: {np.min(pred_array)}")
    print(f"  Mean: {np.mean(pred_array)}")
    print(f"  Std Dev: {np.std(pred_array)}")
    print(f"  25th percentile: {np.percentile(pred_array, 25)}")
    print(f"  Median: {np.median(pred_array)}")
    print(f"  75th percentile: {np.percentile(pred_array, 75)}")
    print(f"  Range: {np.ptp(pred_array)}\n")  # ptp is peak-to-peak (max - min)

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