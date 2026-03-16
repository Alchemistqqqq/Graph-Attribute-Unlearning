import os
import sys
import time
import wandb
import torch
sys.path.append(os.getcwd())
from core.args import Train_Parser
from core.data import Graph_Loader, Graph_Dataloader, Graph_save
from core.model import Model_Loader
from core.trainer import Trainer_Loader
from core.utils import Save_Splits


def run(args):

    run_name = args.model+"_" \
              + args.dataset+"_" \
              + str(args.epochs)+"_" \
              + args.train_method+"_" \
              + time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.wandb_run_name != "":
        run_name = args.wandb_run_name + "_" + run_name    

    _wandb = wandb.init(entity="user_name",
                        project=args.wandb_project,
                        config=args,
                        name=run_name)

    # Configure save path
    save_path = os.path.join(args.save_path, args.wandb_project, _wandb.name)
    os.makedirs(save_path, exist_ok=True)

    graph = Graph_Loader(args)
    dataloaders = Graph_Dataloader(args, graph.to(args.device))
    num_classes=len(torch.unique(graph.ndata["label"]))
    Graph_save(save_path, graph)
   
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
        "model": args.model,
        "depth": args.depth,
        "dropout": args.dropout,
        "batchnorm": args.batchnorm,
        "latent_size": args.latent_size
    }
    training_stats.update(Save_Splits(args, graph))
    
    for e in range(args.epochs):
        
        # train
        out_dict = trainer.fit(e)
        
        # evaluate
        test_acc, _ = trainer.evaluate()
        out_dict["test_acc"] = test_acc[0]

        # save
        if args.save_best:
            if test_acc[0] > training_stats["test_acc"]:
                training_stats["state_dict"] = trainer.model.state_dict()
                training_stats["test_acc"] = test_acc[0]
                os.makedirs(os.path.join(save_path, "best"), exist_ok=True)
                torch.save(training_stats, os.path.join(save_path, "best", "model.pt"))
        if (e+1) % args.save_interval == 0:
            _save_stats = {
                "state_dict": trainer.model.state_dict(),
                "test_acc": test_acc[0],
                "model": args.model,
                "depth": args.depth,
                "dropout": args.dropout,
                "batchnorm": args.batchnorm,
                "latent_size": args.latent_size
                }
            os.makedirs(os.path.join(save_path, str(e)), exist_ok=True)
            torch.save(_save_stats, os.path.join(save_path, str(e), "model.pt"))

        wandb.log(out_dict)

    # Training finished
    if not args.save_best:
        # save the last ones
        training_stats["state_dict"] = trainer.model.state_dict()
        training_stats["test_acc"] = test_acc[0]

    _wandb.finish()
    print("Terminated")

if __name__ == "__main__":
    parser = Train_Parser()
    args = parser.parse_args()
    run(args)
