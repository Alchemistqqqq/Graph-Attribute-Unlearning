import torch
import numpy as np


def Gen_in_or_out(args, graph):
    unused_idx = torch.where(graph.ndata["unused_train_mask"]  > 0)[0]
    unlearn_idx = torch.where(graph.ndata["unlearn_mask"] > 0)[0]
    unused_idx = unused_idx.cpu().numpy()
    unlearn_idx = unlearn_idx.cpu().numpy()
    orig_train_idx = torch.where(graph.ndata["train_mask"] > 0)[0].cpu().numpy()

    if args.num_member == 0:
        num_member = len(unlearn_idx)
    else:
        num_member = args.num_member

    if args.shadow_type == "in":
        if args.selection_strategy == "unlearning":
            num_member = max(num_member, len(unlearn_idx))
            unused_size = int(len(unused_idx)*args.shadow_train_ratio)
            _train_idx = np.random.choice(unused_idx, unused_size, replace=False)
            member_idx = np.random.choice(unlearn_idx, num_member, replace=False)
            train_idx = np.concatenate([_train_idx, member_idx])

        elif args.selection_strategy == "random":
            # select random samples from train datasets
            unused_size = int(len(unused_idx)*args.shadow_train_ratio)
            _train_idx = np.random.choice(unused_size, unused_size, replace=False)
            member_idx = np.random.choice(orig_train_idx, num_member, replace=False)
            train_idx = np.concatenate([_train_idx, member_idx])

    else:
        unused_size  = int(len(unused_idx)*args.shadow_train_ratio)
        _train_idx = np.random.choice(unused_idx, unused_size, replace=False)
        train_idx = _train_idx
    
    train_mask = np.in1d(np.arange(graph.num_nodes()), train_idx)
    graph.ndata["train_mask"] = torch.from_numpy(train_mask).to(graph.device)

    if args.shadow_type == "in":
        member_mask = np.in1d(np.arange(graph.num_nodes()), member_idx)
        graph.ndata["member_mask"] = torch.from_numpy(member_mask).to(graph.device)

    return graph


def Gen_shadow_training_graph(args, graph, use_unused=True):

    if use_unused:
        unused_idx = torch.where(graph.ndata["unused_train_mask"]  > 0)[0]
    
    else:
        train_idx = torch.where(graph.ndata["train_mask"] > 0)[0]
        num_sub_train = int(args.shadow_train_ratio * len(train_idx))
        sub_train_idx = np.random.choice(train_idx, num_sub_train, replace=False)
        sub_train_mask = np.in1d(np.arange(graph.num_nodes()), sub_train_idx)
        graph.ndata["sub_train_mask"] = torch.from_numpy(sub_train_mask).to(graph.device)

    return graph