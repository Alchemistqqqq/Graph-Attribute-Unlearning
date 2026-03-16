import dgl
import torch
import numpy as np
from tqdm import tqdm
from dgl.dataloading import DataLoader as GraphDataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler

def get_attack_data_by_class(model, num_classes, graph, indices, label, args):
    """
    Get a graph,
    For each node, get a prediction, and generate a pred-label pair
    """

    device = args.device

    sampler = MultiLayerFullNeighborSampler(args.depth)
    graph = graph.to(device)
    indices = indices.to(device)
    train_loader = GraphDataLoader(graph=graph,
                                   batch_size=1,
                                   indices=indices,
                                   graph_sampler=sampler,
                                   shuffle=False,
                                   num_workers=0)
    model.to(device)
    model.eval()
    attack_data = dict()
    for c in range(num_classes):
        attack_data[c] = dict()
        attack_data[c]["data"] = list()
        attack_data[c]["labels"] = list()

    pbar = tqdm(len(train_loader))
    for idx, (in_nodes, out_nodes, blocks) in enumerate(train_loader):

        blocks = [b.to(device) for b in blocks]
        data = blocks[0].srcdata["feat"].to(device)
        target = blocks[-1].dstdata["label"].to(device)

        pred, feat = model(blocks, data)
        _class = target.item()
        

        attack_data[_class]["data"].append(pred.detach().cpu())
        attack_data[_class]["labels"].append(torch.Tensor([label]))
        pbar.update(1)

    pbar.close()

    for _class in range(num_classes):
        if len(attack_data[_class]["data"]) > 0:
            attack_data[_class]["data"] = torch.vstack(attack_data[_class]["data"])
            attack_data[_class]["labels"] = torch.vstack(attack_data[_class]["labels"])
        else:
            print(_class)
            attack_data[_class]["data"] = None
            attack_data[_class]["labels"] = None

    return attack_data

def get_attack_data(model, graph, indices, label, args):
    """
    Get a graph,
    For each node, get a prediction, and generate a pred-label pair
    """

    device = args.device

    sampler = MultiLayerFullNeighborSampler(args.depth)
    graph = graph.to(device)
    indices = indices.to(device)
    train_loader = GraphDataLoader(graph=graph,
                                   batch_size=1,
                                   indices=indices,
                                   graph_sampler=sampler,
                                   shuffle=False,
                                   num_workers=0)
    model.to(device)
    model.eval()
    attack_data = dict()
    attack_data["data"] = list()
    attack_data["labels"] = list()

    pbar = tqdm(len(train_loader))
    for idx, (in_nodes, out_nodes, blocks) in enumerate(train_loader):

        blocks = [b.to(device) for b in blocks]
        data = blocks[0].srcdata["feat"].to(device)
        target = blocks[-1].dstdata["label"].to(device)

        pred, feat = model(blocks, data)
        _class = target.item()
        

        attack_data["data"].append(pred.detach().cpu())
        attack_data["labels"].append(torch.Tensor([label]))
        pbar.update(1)

    pbar.close()

    attack_data["data"] = torch.vstack(attack_data["data"])
    attack_data["labels"] = torch.vstack(attack_data["labels"])

    return attack_data
        
def split_train_unlearn_data(graph, ratio=0.5):
    """
    Split retain nodes and test nodes for training MIA.
    """

    retain_indices = torch.where(graph.ndata["retain_mask"] == True)[0].numpy()
    unlearn_indices = torch.where(graph.ndata["unlearn_mask"] == True)[0].numpy()
    test_indices = torch.where(graph.ndata["test_mask"] == True)[0].numpy()

    num_train = int(len(test_indices) * ratio)
    num_test = int(len(test_indices)*(1-ratio))
    print(len(retain_indices), len(unlearn_indices), len(test_indices), num_train, num_test)
    np.random.shuffle(retain_indices)
    np.random.shuffle(unlearn_indices)
    np.random.shuffle(test_indices)

    member_train = retain_indices[:num_train]
    member_test = retain_indices[num_train:num_train+num_test]
    nomem_train = test_indices[:num_train]
    nomem_test = test_indices[num_train:num_train+num_test]
    eval_test = unlearn_indices

    member_train = torch.from_numpy(member_train)
    member_test = torch.from_numpy(member_test)
    nomem_train = torch.from_numpy(nomem_train)
    nomem_test = torch.from_numpy(nomem_test)
    eval_test = torch.from_numpy(eval_test)

    return member_train, member_test, nomem_train, nomem_test, eval_test

def split_train_unlearn_data_2(graph, ratio=0.5):
    """
    Split retain nodes and test nodes for tranining MIA
    """
    retain_indices = torch.where(graph.ndata["retain_mask"] == True)[0].numpy()
    unused_train_indices = torch.where(graph.ndata["unused_train_mask"] == True)[0].numpy()
    unlearn_indices = torch.where(graph.ndata["unlearn_mask"] == True)[0].numpy()

    num_samples = min(len(retain_indices), len(unused_train_indices))
    num_train = int(num_samples * ratio)
    num_test = int(num_samples * (1-ratio))

    np.random.shuffle(retain_indices)
    np.random.shuffle(unused_train_indices)
    np.random.shuffle(unlearn_indices)
    
    member_train = retain_indices[:num_train]
    member_test = retain_indices[num_train:num_train+num_test]
    nomem_train = unused_train_indices[:num_train]
    nomem_test = unused_train_indices[num_train:num_train+num_test]
    eval_test = unlearn_indices

    member_train = torch.from_numpy(member_train)
    member_test = torch.from_numpy(member_test)
    nomem_train = torch.from_numpy(nomem_train)
    nomem_test = torch.from_numpy(nomem_test)
    eval_test = torch.from_numpy(eval_test)

    return member_train, member_test, nomem_train, nomem_test, eval_test


def merge_member_nonmem_data_by_class(member_data, nomem_data):

    combined = dict()
    for (m_k, m_v), (nm_k, nm_v) in zip(member_data.items(), nomem_data.items()):
        assert m_k == nm_k
        combined_preds = torch.cat((m_v["data"], nm_v["data"]), 0)
        combined_labels = torch.cat((m_v["labels"], nm_v["labels"]), 0)
        combined[m_k] = {
            "data": combined_preds,
            "labels": combined_labels
        }
        
    return combined

def merge_member_nonmem_data(member_data, nomem_data):

    combined_preds = torch.cat((member_data["data"], nomem_data["data"]), 0)
    combined_labels = torch.cat((member_data["labels"], nomem_data["labels"]), 0)
    combined = {
        "data": combined_preds,
        "labels": combined_labels
    }

    return combined