import os
import sys
import dgl
import time
import torch
import numpy as np 
sys.path.append(os.getcwd())
from dgl.dataloading import DataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler
from tqdm import tqdm
from core.args import LiRA_Shadow_Parser
from core.model import Model_Loader
from core.trainer import Trainer_Loader
from core.data import Graph_Loader, Graph_Dataloader
from core.lira.utils.shadow import Gen_shadow_training_graph

def Gen_keep(args, graph, shadow_id):
    np.random.seed(0)
    size = sum(graph.ndata["train_mask"])
    train_indices = (torch.where(graph.ndata["train_mask"] == True)[0]).numpy()
    keep = np.random.uniform(0, 1, size=(args.num_shadow_models, size))
    order = keep.argsort(0)
    keep = order < int(args.shadow_train_ratio * args.num_shadow_models)
    keep = np.array(keep[shadow_id], dtype=bool)
    keep = keep.nonzero()[0]

    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    sub_train_indices = train_indices[keep_bool]
    sub_train_mask = np.in1d(np.arange(graph.num_nodes()), sub_train_indices)

    return keep_bool, torch.from_numpy(sub_train_mask)


def train_shadow(args):

    for model_number in range(args.num_shadow_models):
        run_name = str(args.model)+"_" \
                + args.dataset+"_" \
                + str(args.epochs)+"_" \
                + args.train_method+"_" \
                + str(model_number)

        graph = Graph_Loader(args)
        graph = Gen_keep(args, graph, model_number)
        dataloaders = Graph_Dataloader(args, graph)
        sampler = MultiLayerFullNeighborSampler(args.depth)
        save_path = os.path.join(args.save_path, args.wandb_project, run_name)

        train_loader = DataLoader(graph=graph,
                                batch_size=args.batch_size,
                                indices=torch.where(graph.ndata["sub_train_mask"] == True)[0],
                                graph_sampler=sampler,
                                device=torch.device(args.device),
                                shuffle=True,
                                drop_last=False,
                                num_workers=0)

        test_loader = DataLoader(graph=graph,
                                batch_size=args.batch_size,
                                indices=torch.where(graph.ndata["test_mask"] == True)[0],
                                graph_sampler=sampler,
                                device=torch.device(args.device),
                                shuffle=True,
                                drop_last=False,
                                num_workers=0) 

        num_classes = len(torch.unique(graph.ndata["label"]))

        dataloaders = {
            "train": train_loader,
            "test": test_loader,
        }

        model = Model_Loader(args)(args,
                                    in_dim=graph.ndata["feat"].size(1),
                                    hidden_dim=args.latent_size,
                                    num_classes=num_classes)
        
        trainer = Trainer_Loader(args)(args,
                                    model=model,
                                    dataloaders=dataloaders)
        
        training_stats = {
            "state_dict": model.state_dict(),
            "test_acc": 0.0,
            "last_acc": 0.0,
            "model": args.model,
            "depth": args.depth,
            "dropout": args.dropout,
            "batchnorm": args.batchnorm,
            "latent_size": args.latent_size,
            "keep": graph.ndata["sub_train_mask"].cpu()
        }

        for e in range(args.epochs):
            out_dict = trainer.fit(e)
            test_acc, _ = trainer.evaluate()
            out_dict["test_acc"] = test_acc[0]

            if args.save_best:
                if test_acc[0] > training_stats["test_acc"]:
                    training_stats["state_dict"] = trainer.model.state_dict()
                    training_stats["test_acc"] = test_acc[0]
                    os.makedirs(os.path.join(save_path, "best"), exist_ok=True)
                    torch.save(training_stats, os.path.join(save_path, "best", "model.pt"))

        training_stats["last_acc"] = test_acc[0]
        os.makedirs(os.path.join(save_path, "last"), exist_ok=True)
        torch.save(training_stats, os.path.join(save_path, "last", "model.pt"))