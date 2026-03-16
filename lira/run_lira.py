import os
import dgl
import sys
import time
import wandb
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
sys.path.append(os.getcwd())
from core.args import LiRA_Parser
from core.lira import LIRA
from sklearn.metrics import auc, roc_curve
from core.data import Graph_Load_from
from core.model import Model_Loader
from core.args import LiRA_Parser
from dgl.dataloading import DataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler
import dgl.sparse as dglsp

def run(args):
    run_name = "MIA_" \
              + args.model+"_" \
              + args.dataset+"_" \
              + time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.wandb_run_name != "":
        run_name = args.wandb_run_name + "_" + run_name    

    _wandb = wandb.init(entity="user_name",
                        project=args.wandb_project,
                        config=args,
                        name=run_name)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    graph = Graph_Load_from(args)
    graph.to(args.device)
    num_classes = len(torch.unique(graph.ndata["label"]))
    print(graph.ndata.keys())

    lira = LIRA(args, graph)
    load_path = os.path.join(args.orig_load_path, args.orig_load_epoch, "model.pt")
    checkpoint = torch.load(load_path)
    args.batchnorm = checkpoint["batchnorm"]
    orig_model = Model_Loader(args)(args,
                                in_dim=graph.ndata["feat"].size(1),
                                hidden_dim=args.latent_size,
                                num_classes=num_classes)

    
    for k, v in checkpoint.items():
        if k == "state_dict":
            for kk, vv in v.items():
                print(kk)
        else:
            print(k, v)

    for k, v in orig_model.state_dict().items():
        print(k)

    orig_model.load_state_dict(checkpoint["state_dict"])
    orig_unlearn_auc, orig_retain_auc = lira.test_model(orig_model, "original")

    print("orig unlearn auc", orig_unlearn_auc[2])
    print("orig retain auc", orig_retain_auc[2])

    # First figure (log scale)
    fig_log, ax_log = plt.subplots(2, 2, figsize=(12,8))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    # Second figure (regular scale)
    fig_reg, ax_reg = plt.subplots(2, 2, figsize=(12,8))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    fig_total, ax_total = plt.subplots(1, 1, figsize=(6,4))

    # Configure log-scale plots
    for i in range(2):
        for j in range(2):
            ax_log[i,j].semilogx()
            ax_log[i,j].semilogy()
            ax_log[i,j].set_xlim(1e-5,1)
            ax_log[i,j].set_ylim(1e-5,1)
            ax_log[i,j].set_xlabel("False Positive Rate")
            ax_log[i,j].set_ylabel("True Positive Rate")
            
            # Regular scale plots
            ax_reg[i,j].set_xlim(0,1)
            ax_reg[i,j].set_ylim(0,1)
            ax_reg[i,j].set_xlabel("False Positive Rate")
            ax_reg[i,j].set_ylabel("True Positive Rate")

    ax_total.set_xscale("log")
    ax_total.set_yscale("log")
    ax_total.set_xlim(1e-5, 1)
    ax_total.set_ylim(1e-5, 1)
    ax_total.set_xlabel("False Positive Rate")
    ax_total.set_ylabel("True Positive Rate")

    # Set titles and plot original data
    titles = ["AUC of original model on train & test data",
              "AUC of unlearned model on train & test data",
              "AUC of original model on unlearn & test data",
              "AUC of victim model on unlearn & test data"]
    
    for ax, title in zip(ax_log.flat, titles):
        ax.title.set_text(title)
    for ax, title in zip(ax_reg.flat, titles):
        ax.title.set_text(title)

    ax_total.title.set_text("AUC of original & unlearned models")

    # Plot original data
    ax_log[0, 0].plot(orig_retain_auc[0], orig_retain_auc[1])
    ax_log[1, 0].plot(orig_unlearn_auc[0], orig_unlearn_auc[1])
    ax_reg[0, 0].plot(orig_retain_auc[0], orig_retain_auc[1])
    ax_reg[1, 0].plot(orig_unlearn_auc[0], orig_unlearn_auc[1])
    ax_total.plot(orig_unlearn_auc[0], orig_unlearn_auc[1], label="Original")

    _auc_out = dict()
    _auc_save = dict()

    for _path, _epoch, _name, _color in zip(args.victim_load_path, args.victim_load_epoch, args.victim_name, args.victim_color):
        # Load victim models
        load_path = os.path.join(_path, _epoch, "model.pt")
        checkpoint = torch.load(load_path)

        # Load graph
        load_path = os.path.join(args.load_path, "graph.bin")
        graphs, _ = dgl.load_graphs(load_path)
        _graph = graphs[0]

        lira = LIRA(args, _graph)

        victim_model = Model_Loader(args)(args,
                                    in_dim=graph.ndata["feat"].size(1),
                                    hidden_dim=args.latent_size,
                                    num_classes=num_classes)
        victim_model.load_state_dict(checkpoint["state_dict"])
        unlearn_auc, retain_auc = lira.test_model(victim_model, _name)

        print(f"{_name}, unlearn auc", unlearn_auc[2])
        print(f"{_name}, retain auc", retain_auc[2])
        _auc_out[f"{_name}_unlearn_auc"] = unlearn_auc[2]
        _auc_out[f"{_name}_retain_auc"] = retain_auc[2]

        ax_log[1, 1].plot(unlearn_auc[0], unlearn_auc[1], label=_name, color=_color)
        ax_log[0, 1].plot(retain_auc[0], retain_auc[1], label=_name, color=_color)
        ax_reg[1, 1].plot(unlearn_auc[0], unlearn_auc[1], label=_name, color=_color)
        ax_reg[0, 1].plot(retain_auc[0], retain_auc[1], label=_name, color=_color)
        ax_total.plot(unlearn_auc[0], unlearn_auc[1], label=_name, color=_color)


        _auc_save[f"{_name}"] = {
            "unlearn_auc_x": unlearn_auc[0],
            "unlearn_auc_y": unlearn_auc[1],
            "retain_auc_x": retain_auc[0],
            "retain_auc_y": retain_auc[1]
        }

    # Add legends to both figures
    ax_log[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_log[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_reg[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_reg[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_total.legend(loc='upper left', bbox_to_anchor=(1, 1))

    out_dict = {
        "orig_unlearn_auc": orig_unlearn_auc[2],
        "orig_retain_auc": orig_retain_auc[2],
        "plot_log": wandb.Image(fig_log),
        "plot_reg": wandb.Image(fig_reg),
        "plot_total": wandb.Image(fig_total)
    }
    for k, v in _auc_out.items():
        out_dict[k] = v
    wandb.log(out_dict)

    # Save the AUC values
    save_path = os.path.join(args.save_path, run_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "auc_values.pkl"), 'wb') as f:
        pickle.dump(_auc_save, f)


if __name__ == "__main__":
    parser = LiRA_Parser()
    args = parser.parse_args()
    run(args)