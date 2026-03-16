import os
import sys
import time
import wandb
import torch
import logging
import numpy as np
from datetime import datetime
sys.path.append(os.getcwd())
from core.args import Train_Parser
from core.args import Untrain_Parser
from core.data import Graph_Dataloader, Graph_Load_from, Graph_Unlearn_Dataloader
from core.data import Graph_Remove_Unlearnables
from core.model import Unlearn_Model_Loader
from core.trainer import Untrainer_Loader
from core.utils import Checkpoint_Loader
from core.data import Graph_save
from core.utils import Save_Splits
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_device_info(args, model, dataloaders):
    logger = logging.getLogger(__name__)
    
    logger.info("\nDevice Information:")
    logger.info(f"Using device: {args.device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    logger.info("\nModel device check:")
    logger.info(f"Model is on {next(model.parameters()).device}")

    logger.info("\nDataloader device check:")
    for name, loader in dataloaders.items():
        if hasattr(loader, 'device'):
            logger.info(f"{name} loader device: {loader.device}")
        else:
            try:
                batch = next(iter(loader))
                if isinstance(batch, (list, tuple)):
                    device_info = [x.device if torch.is_tensor(x) else 'N/A' for x in batch]
                    logger.info(f"{name} loader first batch devices: {device_info}")
                else:
                    logger.info(f"{name} loader first batch device: {batch.device if torch.is_tensor(batch) else 'N/A'}")
            except:
                logger.info(f"{name} loader device: Could not determine")

def run(args):

    run_name = args.model+"_" \
              + args.dataset+"_" \
              + str(args.epochs)+"_" \
              + args.unlearn_method+"_" \
              + time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.wandb_run_name != "":
        run_name = args.wandb_run_name + "_" + run_name    

    _wandb = wandb.init(entity="user_name",
                        project=args.wandb_project,
                        config=args,
                        name = run_name)
    
    # Seed setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Configure save path
    save_path = os.path.join(args.save_path, args.wandb_project, _wandb.name)
    os.makedirs(save_path, exist_ok=True)

    # Load trained graph
    
    ckpt, splits = Checkpoint_Loader(args)

    graph = Graph_Load_from(args)
    #遗忘
    ul_graph = Graph_Remove_Unlearnables(args, graph)
    print("ul_graph mask:",ul_graph.ndata["unlearn_mask"])
    dataloaders = Graph_Unlearn_Dataloader(args, ul_graph, graph)
    print(f"[INFO] Unlearn nodes total in dataloader: {len(dataloaders['unlearn'].indices)}")
    Graph_save(save_path, ul_graph)

    for k, v in dataloaders.items():
        print(k, len(v))

    num_classes = len(torch.unique(graph.ndata["label"]))
   
    model = Unlearn_Model_Loader(args, ckpt=ckpt)(args,
                               in_dim=graph.ndata["feat"].size(1),
                               hidden_dim=ckpt["latent_size"],
                               num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])

    trainer = Untrainer_Loader(args)(args,
                               model=model,
                               dataloaders=dataloaders,
                               graph=graph)
    
    best_stats = {
        "state_dict": model.state_dict(),
        "test_acc": 0.0,
        "unlearn_acc": 0.0
    }

    #-----------------------------------

    setup_logging()
    log_device_info(args, model, dataloaders)

    #-----------------------------------
    total_start = time.time()

    for e in range(args.epochs):

        # train
        out_dict = trainer.fit(e)
        
        # evaluate
        test_acc, _ = trainer.evaluate()
        out_dict["test_acc"] = test_acc[0]
        out_dict["unlearn_acc"] = test_acc[1]

        # save
        if args.do_save:
            if test_acc[0] > test_acc[1]:
                if test_acc[0] > best_stats["test_acc"]:
                    best_stats["state_dict"] = trainer.model.state_dict()
                    best_stats["test_acc"] = test_acc[0]
                    best_stats["unlearn_acc"] = test_acc[1]
                    os.makedirs(os.path.join(save_path, str(e)), exist_ok=True)
                    torch.save(best_stats, os.path.join(save_path, str(e), "model.pt"))

        # stop_dict = trainer.Check_Stop(metric="avg")
        # out_dict.update(stop_dict)
            
        wandb.log(out_dict)

        print(out_dict)
        print()

        if args.do_save:
            os.makedirs(os.path.join(save_path, str(e)), exist_ok=True)
            torch.save(best_stats, os.path.join(save_path, str(e), "model.pt"))
    
    total_end = time.time()
    print(f"\n[TIME] Total Training Time: {total_end - total_start:.4f} seconds\n")
        
    if args.save_last:
        os.makedirs(os.path.join(save_path, "best"), exist_ok=True)
        torch.save(best_stats, os.path.join(save_path, "best", "model.pt"))

    _wandb.finish()

if __name__ == "__main__":
    parser = Untrain_Parser()
    args = parser.parse_args()
    run(args)

#每个epoch（检查点）的模型参数
#best->model.pt   最佳模型
#处理后的图数据