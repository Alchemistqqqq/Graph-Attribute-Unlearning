from core.trainer.learner.learn import NormalTrainer
from core.trainer.unlearner.unlearn import UnlearnTrainer
from core.trainer.unlearner.retrain import RetrainTrainer
from core.trainer.unlearner.unfeat import UnfeatTrainer


def Trainer_Loader(args):
    if args.train_method == "default":
        return NormalTrainer
    
def Untrainer_Loader(args):
    if args.unlearn_method == "contrastive":
        return UnlearnTrainer
    elif args.unlearn_method == "retrain":
        return RetrainTrainer
    elif args.unlearn_method == "unfeat":
        return UnfeatTrainer
    else:
        raise ValueError(f"Unknown unlearn method: {args.unlearn_method}")
    
