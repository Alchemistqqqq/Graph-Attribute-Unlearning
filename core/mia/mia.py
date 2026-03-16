import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.mia.dataset import AttackDataset
from core.model import get_mia_attack_model
from core.mia.utils.trainer import trainMiaAttackModel, evalMiaAttackModel

def get_optim(model, args):
    if args.optimizer == "sgd":
        _optim = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    elif args.optimizer == "adam":
        _optim = torch.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))
    else:
        raise ValueError("Undefined optimer")
    
    return _optim


class MembershipInferenceAttack:

    def __init__(self, num_classes, args):
        self.num_classes = num_classes
        self.batch_size = args.batch_size
        self.layer_dims = [64, 32, 16]
        self.args = args
        self.num_epochs = args.epochs

    """
    Layer dims, epoch, lr, optim
        cora: [64, 32], 25, 1e-3, adam -> 91.23
        pubmed: 
    
    """


    def train_attack_model_by_class(self,
                           attack_data_train,
                           attack_data_test,
                           device
                           ):
        
        attack_model_list = list()
        for victim_class in range(self.num_classes):
        
            attack_train = AttackDataset(data=attack_data_train[victim_class]["data"],
                                            labels=attack_data_train[victim_class]["labels"],
                                            transform=None)
            attack_test = AttackDataset(data=attack_data_test[victim_class]["data"],
                                        labels=attack_data_test[victim_class]["labels"],
                                        transform=None)
        
            attack_model = get_mia_attack_model(in_feat=self.num_classes,
                                                layer_dims=self.layer_dims,
                                                device=device)
            
            optimizer = get_optim(attack_model, self.args)

            if self.args.attack_criterion == "bce":
                attack_loss = torch.nn.BCELoss()
            elif self.args.attack_criterion == "ce":
                attack_loss = torch.nn.CrossEntropyLoss()
            else:
                raise ValueError("Unsupported attacker loss")

            attack_model = TrainMiaAttackModel(attack_model,
                                               self.num_epochs,
                                               attack_train,
                                               attack_test,
                                               optimizer,
                                               attack_loss,
                                               self.batch_size,
                                               device)

            attack_model_list.append(attack_model)

        return attack_model_list
    
    def train_attack_model(self,
                           attack_data_train,
                           attack_data_test,
                           device
                           ):

        
        attack_train = AttackDataset(data=attack_data_train["data"],
                                        labels=attack_data_train["labels"],
                                        transform=None)
        attack_test = AttackDataset(data=attack_data_test["data"],
                                    labels=attack_data_test["labels"],
                                    transform=None)
    
        attack_model = get_mia_attack_model(in_feat=self.num_classes,
                                            layer_dims=self.layer_dims,
                                            device=device)
        
        optimizer = get_optim(attack_model, self.args)

        if self.args.attack_criterion == "bce":
            attack_loss = torch.nn.BCELoss()
        elif self.args.attack_criterion == "ce":
            attack_loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported attacker loss")

        attack_model = TrainMiaAttackModel(attack_model,
                                            self.num_epochs,
                                            attack_train,
                                            attack_test,
                                            optimizer,
                                            attack_loss,
                                            self.batch_size,
                                            device)
        return attack_model

    
    def evaluate_by_class(self, attack_model, eval_data, device):

        out_dict = dict()
        for victim_class in range(self.num_classes):
            
            eval_test = AttackDataset(data=eval_data[victim_class]["data"],
                                      labels=eval_data[victim_class]["labels"],
                                      transform=None)
            
            eval_dict = evalMiaAttackModel(model=attack_model[victim_class],
                                           eval_data=eval_test,
                                           batch_size=self.batch_size, 
                                           device=device)
            
            for k, v in eval_dict.items():
                out_dict[str(victim_class)+"_"+k] = v
            
        return out_dict
    
    def evaluate(self, attack_model, eval_data, device):

        out_dict = dict()
            
        eval_test = AttackDataset(data=eval_data["data"],
                                    labels=eval_data["labels"],
                                    transform=None)
        
        eval_dict = evalMiaAttackModel(model=attack_model,
                                        eval_data=eval_test,
                                        batch_size=self.batch_size, 
                                        device=device)
            
        return eval_dict

    
def obtain_attack_data(target_data,
                       target_model,
                       device,
                       label):
    
    x_list = list()
    y_list = [label for _ in range(len(target_data))]

    target_model.eval()
    train_loader = DataLoader(target_data, batch_size=64)
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
    
        pred, _ = target_model(data)
        x_list.append(pred)

    assert(len(x_list) == len(y_list))

    return x_list, y_list        


def TrainMiaAttackModel(model,
                        num_epochs,
                        train_data,
                        test_data,
                        optimizer,
                        criterion,
                        batch_size,
                        device,
                        ):

    attack_model = trainMiaAttackModel(model, num_epochs, train_data, test_data,
                                       optimizer, criterion, batch_size, device)

    return attack_model
