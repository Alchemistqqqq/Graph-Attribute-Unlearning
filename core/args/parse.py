from argparse import ArgumentParser

def _Base_Parser():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Seed for selecting unlearning samples")

    # 1. Path setup
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)

    # 2. GraphData setup
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed", "cs", "physics", "reddit", "citeseer", "karate", "ogbn_arxiv"])
    parser.add_argument("--split", type=str, default="inductive", choices=["default", "none", "transductive", "inductive"])
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # 3. Model setup
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "gin", "sgc"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batchnorm", default=False, action="store_true")
    parser.add_argument("--dropout", default=False, action="store_true")
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str,default="sgd")
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_known_args()[0]
    if args.optimizer == "sgd":
        parser.add_argument("--momentum", type=float, default=0.9)
    # if args.optimizer == "adam" or \
    #    args.optimizer == "adamw" or \
    #    args.optimizer == "adadelta" or \
    #    args.optimizer == "adamax" or \
    #    args.optimizer == "adagrad":

    parser.add_argument("--weight_decay", type=float, default=0.009)
    if args.model == "gat":
        parser.add_argument("--head", type=int, default=8)

    # 4. Logging & save setup
    parser.add_argument("--wandb_project", type=str, default="cgul_train")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_best", default=False, action="store_true")
    parser.add_argument("--save_last", default=False, action="store_true")
    parser.add_argument("--save_interval", type=int, default=20)

    return parser

def Train_Parser():
    parser = _Base_Parser()
    
    # Training specific setups
    parser.add_argument("--train_method", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=100)

    return parser

def Untrain_Parser():
    parser = _Base_Parser()

    # Unlearning specific setup
    parser.add_argument("--unlearn_method", type=str, default="contrastive")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--unlearn_type", type=str, default="random_node_contrastive")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--load_epoch", type=str, default="best")
    parser.add_argument("--num_unlearn", type=float, default=50)
    parser.add_argument("--no_reconstruct", default=False, action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--temperature", type=str, default=0.7)
    parser.add_argument("--do_save", default=False, action="store_true")
    parser.add_argument("--beta", type=float, default=4)

    # PPR related settings
    parser.add_argument("--use_ppr", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=1e-3)
    parser.add_argument("--stop_cond", type=str, default="dist", choices=["none", "acc", "dist"])    
    parser.add_argument("--stop_cond_metric", type=str, default="avg", choices=["min", "max", "avg"])

    #gradient ascent specific
    parser.add_argument("--ascent_scale", type=float, default=0.01)

    return parser

def Mia_Parser():
    parser = Untrain_Parser()

    # Setup MIA Parser
    parser.add_argument("--layer_dims", type=int, default=10)
    parser.add_argument("--attack_criterion", type=str, default="bce")
    parser.add_argument("--member_test_ratio", type=float, default=0.5)
    parser.add_argument("--victim_load_path", type=str)
    parser.add_argument("--victim_load_epoch", type=str, default="0")

    return parser

def LiRA_Parser():
    parser = _Base_Parser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--orig_load_path", type=str)
    parser.add_argument("--orig_load_epoch", type=str)
    parser.add_argument("--victim_name", action='append')
    parser.add_argument("--victim_color", action='append')
    parser.add_argument("--victim_load_path", action='append')
    parser.add_argument("--victim_load_epoch", action='append')
    parser.add_argument("--shadow_path", type=str)
    parser.add_argument("--n_queries", type=int, default=2)
    parser.add_argument("--use_same_size", action='store_true')

    return parser

def LiRA_Shadow_Parser():
    parser = Train_Parser()
       
    #setup LiRA shadow model params
    parser.add_argument("--shadow_type", type=str, choices=["in", "out"])
    parser.add_argument("--model_number", type=int)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--shadow_train_ratio", type=float, default=0.8) 
    parser.add_argument("--selection_strategy", type=str)
    parser.add_argument("--num_member", type=int, default=0)
    parser.add_argument("--num_shadow_models", type=int, default=16)
    parser.add_argument("--use_unused", default=False, action="store_true")
    return parser

def LiRA_Train_Parser():
    parser = _Base_Parser()
    
    # Training specific setups
    parser.add_argument("--train_method", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mia_train_ratio", type=float, default=0.5)
    parser.add_argument("--unlearn_type", type=str, default="random_node_contrastive")
    parser.add_argument("--num_unlearn", type=float, default=0.1)


    return parser