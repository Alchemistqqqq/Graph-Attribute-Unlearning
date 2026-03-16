import copy
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.mia.utils.metrics import mia_metrics

def train_step(model,
               train_loader,
               optimizer,
               criterion,
               current_epoch,
               device
               ):
    
    model.train()
    model.to(device)
    total_targets = None
    total_preds = None
    total_loss = None

    for idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()

        if pred.size(1) == 1:
            preds = (pred > 0.5).float()
        else:
            _, preds = pred.max(dim=1)

        if total_targets is None:
            total_targets = target.clone().cpu()
        else:
            total_targets = torch.cat([total_targets, target.cpu()])
        if total_preds is None:
            total_preds = preds.clone().cpu()
        else:
            total_preds = torch.cat([total_preds, preds.cpu()])
        if total_loss is None:
            total_loss = loss.item()
        else:
            total_loss += loss.item()
        
        optimizer.step()
    out_dict = mia_metrics(pred=total_preds, true=total_targets, suffix="train")

    return out_dict, total_loss/len(train_loader)


def eval_step(model,
              test_loader,
              device,
              suffix="test"
              ):
    
    model.eval()
    total_targets = None
    total_preds = None
    
    for idx, (data, target) in enumerate(test_loader):

        data = data.to(device)
        target = target.to(device)

        pred = model(data)

        if pred.size(1) == 1:
            preds = (pred > 0.5).float()
        else:
            _, preds = pred.max(dim=1)

        if total_targets is None:
            total_targets = target.clone().cpu()
        else:
            total_targets = torch.cat([total_targets, target.cpu()])
        if total_preds is None:
            total_preds = preds.clone().cpu()
        else:
            total_preds = torch.cat([total_preds, preds.cpu()])
                
    out_dict = mia_metrics(pred=total_preds, true=total_targets, suffix=suffix)

    return out_dict


def trainMiaAttackModel(model,
                        num_epochs,
                        train_data,
                        test_data,
                        optimizer,
                        criterion,
                        batch_size,
                        device
                        ):

    
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
        
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)

    pbar = tqdm(total=num_epochs, desc="MIA_training")

    for epoch in range(num_epochs):
        model.train()
        out_dict = dict()
        train_out_dict, avg_loss = train_step(model, train_loader, optimizer, criterion, epoch, device)
        eval_out_dict = eval_step(model, test_loader, device)

        out_dict.update(train_out_dict)
        out_dict.update(eval_out_dict)

        wandb.log(out_dict)
        pbar.set_description(f"Loss: {avg_loss}")
        pbar.update()
    pbar.close()

    return model

def evalMiaAttackModel(model,
                       eval_data,
                       batch_size,
                       device
                       ):
    model.eval()
    test_loader = DataLoader(eval_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    
    eval_out_dict = eval_step(model, test_loader, device, suffix="test_unlearn")

    return eval_out_dict