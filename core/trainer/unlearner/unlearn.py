import dgl
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from core.trainer.base import BaseTrainer
from core.trainer.utils import get_optim, to_device, out_dict, clone_blocks
from core.trainer.unlearner.ppr_utils import Get_PPR, Get_RPPR, get_rppr, get_ppr_np, get_rppr_np
import logging
# Typing
from typing import List, Dict, Tuple
from torch.nn import Module

class UnlearnTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 model:Module,
                 dataloaders: Dict,
                 **kwargs):
        """
        ul : unlearning nodes
        rdc: randomly selected nodes with different class (pull ul noded towards)
        nb : neighbors of unlearning nodes
        lpnb: neighbors of unlearning nodes whose PPR is lower
        """
        
        # Model setting
        self.model = model
        #print("Dataloaders keys:", list(dataloaders.keys()))
        self.loader_rt = dataloaders["retain_train"]
        self.loader_ul = dataloaders["unlearn"]
        self.test_loaders = (dataloaders["retain_test"], dataloaders["unlearn_test"])
        self.device = args.device
        self.model.to(self.device)
        self.beta = args.beta
        # Optimizer setting
        self.optim = get_optim(args)(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, args.epochs)       

        # Unlearn hyparparameters setting
        self.repeat = args.repeat
        self.use_ppr = args.use_ppr
        self.temperature = args.temperature
        self.base_temperature = args.temperature
        self.no_reconstruct = args.no_reconstruct

        # PPR & RPPR hyperparameters
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.rho = args.rho


        self.rt_iter = iter(self.loader_rt)
        try:
            self.graph = kwargs["graph"]
            self.graph = self.graph.to(self.device)
        except:
            raise ValueError("graph not defined")
        
        # Stopping condition setting
        if args.stop_cond == "acc":
            self._stop_cond_fn = self._stop_cond_acc
        elif args.stop_cond == "dist":
            self._stop_cond_fn = self._stop_cond_dist
            try:
                self.loader_cnd_ul = dataloaders["stop_cond_ul"]
                self.loader_cnd_test = dataloaders["stop_cond_test"]
            except:
                raise ValueError("Distance based stopping condtion requires self.loader_cnd")
        else:
            self._stop_cond_fn = self._stop_cond_none

    def fit(self, epoch:int)->Dict:
        _out = out_dict()
        self.model.to(self.device)
        self.model.train()
        epoch_stats = dict()
        pbar = tqdm(total=len(self.loader_ul))

        for ul_idx, (ul_blocks, nb_blocks, ul_link, nb_link) in enumerate(self.loader_ul):
            logging.info(f"Unlearn_batch {ul_idx} from epochs {epoch}")            
            self.model.zero_grad()
            train_stats = dict()
            _nb_blocks = dict()
            _nb_data = dict()
            _nb_label = dict()
            _nb_ori_feat = dict()

            ul_blocks, ul_data, ul_label, ul_ori_feat = to_device(ul_blocks, self.device)

            for _key, blocks in nb_blocks.items():
                _nb_blocks[_key], _nb_data[_key], _nb_label[_key], _nb_ori_feat[_key] = to_device(blocks, self.device)
            for _key, _tensor in nb_link.items():
                nb_link[_key] = _tensor.to(self.device)
            for _key, _tensor in ul_link.items():
                ul_link[_key] = _tensor.to(self.device)

            ulnb_link = nb_link.pop(-1)
            if self.use_ppr:
                #PPR
                # ppr_vec = get_ppr(block=ul_blocks[-1], graph=self.graph, alpha=self.alpha, epsilon=self.epsilon,
                #                   rho=self.rho, ul_link=ul_link, device=self.device, topk=None, adj_data=None)
                # #RPPR
                # get_ppr_ = Get_PPR(graph=self.graph, alpha=self.alpha, epsilon=self.epsilon, rho=self.rho, device=self.device)
                # get_ppr_ = nn.parallel.DataParallel(get_ppr_)
                # ppr_vec = get_ppr_(ul_blocks[-1], ul_link)
                ppr_vec = get_ppr_np(ul_blocks[-1], self.graph, self.alpha, self.epsilon, self.rho, ul_link[0].T, self.device)
                ppr_vec = torch.mean(ppr_vec, dim=1) 
                # get_rppr_ = Get_RPPR(graph=self.graph, alpha=self.alpah, epsilon=self.epsilon, rho=self.rho, device=self.device)

                rppr_vec=dict()
                
                for _key in nb_link.keys():
                    _block = _nb_blocks[_key]
                    _rel = nb_link[_key].to(torch.float)
                    _nodes = _block[-1].srcdata["_ID"]
                    _target = ul_blocks[-1].dstdata["_ID"]
                    ulnb_link = ulnb_link.to(torch.int)
                    # rppr_vec[_key] = get_rppr_(block=_block, target=_target, ul_link=ul_link)
                    # _rppr_vec = get_rppr_np(block=_block[-1], graph=self.graph, target=_target, alpha=self.alpha, epsilon=self.epsilon,
                    #                              rho=self.rho, ul_link=ulnb_link, device=self.device)
                    # rppr_vec[_key] = torch.mean(_rppr_vec, dim=1)
                    rppr_vec[_key] = torch.ones_like(_block[-1].dstdata["_ID"], device=self.device)
                    ulnb_link = torch.mm(_rel.T, ulnb_link.to(torch.float)).to(torch.int)

            else:
                ppr_vec = torch.ones_like(ul_blocks[-1].dstdata["_ID"], device=self.device)
                rppr_vec = dict()
                for _key, _block in _nb_blocks.items():
                    rppr_vec[_key] = torch.ones_like(_block[-1].dstdata["_ID"], device=self.device)
                rppr_vec.pop(list(rppr_vec.keys())[-1]) # pop last element out

            # ul_pred, ul_feat = self.model(ul_blocks, ul_data)
            # print(ul_pred.shape, ul_label.shape)

            for _repeat in range(self.repeat):
                logging.info(f"    Repeat {_repeat} from epochs {epoch}")
                _ul_blocks, _ul_data, _ul_ori_feat = clone_blocks(ul_blocks, ul_data, ul_ori_feat)
                ul_pred, ul_feat = self.model(_ul_blocks, _ul_data)
                self.optim.zero_grad()
                try:
                    rt_in_nodes, rt_out_nodes, rt_blocks = next(self.rt_iter)
                except:
                    self.rt_iter = iter(self.loader_rt)
                    rt_in_nodes, rt_out_nodes, rt_blocks = next(self.rt_iter)
                rt_blocks, rt_data, rt_label, rt_ori_feat = to_device(rt_blocks, self.device)
                rt_pred, rt_feat = self.model(rt_blocks, rt_data)

                # Get pred, feat of first-hop neighbors
                nb1_pred, nb1_feat = self.model(_nb_blocks[0], _nb_data[0])
                logging.info("        pass")

                loss, loss_val = self.CT_Loss(ul_feat, rt_feat, nb1_feat,
                                              ul_pred, rt_pred,
                                              ul_label, rt_label, _nb_label[0],
                                              ppr_vec)
                loss.backward()
                self.optim.step()
                train_stats.update(loss_val)
                logging.info("        Update")

                if _repeat % 2 == 0 and self.no_reconstruct is False:        
                    self.optim.zero_grad()
                    _, _, _, nb_loss = self.recursive_reconstruct(copy.deepcopy(_nb_blocks),
                                                                copy.deepcopy(_nb_data),
                                                                copy.deepcopy(_nb_label),
                                                                copy.deepcopy(nb_link),
                                                                copy.deepcopy(rppr_vec))
                    train_stats.update(nb_loss)
                    logging.info("            Reconstruct")
            for key, value in train_stats.items():
                try:
                    epoch_stats[key].append(value)
                except:
                    epoch_stats[key] = list()
                    epoch_stats[key].append(value)
            pbar.update()
        pbar.close()

        for k in epoch_stats.keys():
            epoch_stats[k] = sum(epoch_stats[k])/len(epoch_stats[k])

        _out.update(train_stats)

        return _out

    def CT_Loss(self,
                ul_feat,
                rt_feat,
                nb1_feat,
                ul_pred,
                rt_pred,
                ul_label,
                rt_label,
                nb1_label,
                ppr_vec):
        
        loss_track = dict()

        ct_loss = self._CT_Loss(ul_feat, rt_feat, nb1_feat, ul_label, rt_label, nb1_label, ppr_vec)
        ce_loss = self._CE_Loss(rt_pred, rt_label)
        loss_track["ulct_loss"] = ct_loss.item()
        loss_track["ulce_loss"] = ce_loss.item()

        return  ct_loss + self.beta * ce_loss, loss_track
    
    def _CT_Loss(self, ul_feat, rt_feat, nb1_feat, ul_label, rt_label, nb1_label, ppr_vec):
        """
        Contrastive Loss

        Anchor: unlearning nodes (ul_pred)
        Positive: (Ones who attract) retain nodes with different labels
        Negative: (Ones who repulse)neighbors
        
        """

        ul_label = ul_label.contiguous().view(-1, 1)
        rt_label  = rt_label.contiguous().view(-1, 1)
        nb1_label = nb1_label.contiguous().view(-1, 1)

        mask = torch.eq(ul_label, rt_label.T)
        n_mask = (mask).clone().float()
        p_mask = (~mask).clone().float()

        p_logits = torch.matmul(ul_feat, rt_feat.T)
        # print("p_logits_init", p_logits)
        p_logits = torch.div(p_logits, self.temperature)
        p_logits_max, _ =  torch.max(p_logits, dim=1, keepdim=True)
        p_logits -= p_logits_max.detach()

        nb_mask = torch.eq(ul_label, nb1_label.T)
        n_logits = torch.matmul(ul_feat, nb1_feat.T)
        n_logits = torch.div(n_logits, self.temperature)
        n_logits_max, _ = torch.max(n_logits, dim=1, keepdim=True)
        n_logits -= n_logits_max.detach()
        n_logits = torch.exp(n_logits) * nb_mask

        # Add same classes as n_logits here
        n_logits_2 = torch.exp(p_logits) * n_mask
        p_logits = p_logits * p_mask

        # print("n_logits_2", n_logits_2)
        # print("p_logits", p_logits)

        # n_logits = n_logits.sum(1, keepdim=True) + n_logits_2.sum(1, keepdim=True)
        # n_logits = n_logits_2.sum(1, keepdim=True)
        n_logits = n_logits.sum(1, keepdim=True)

        # Apply PPR here
        # print(n_logits.shape, ppr_vec.unsqueeze(1).shape)
        n_logits = n_logits * ppr_vec.unsqueeze(1) + 1e-20
        # print("n_logits", n_logits)

        # print("n_logits after ppr", n_logits)

        log_prob = p_logits - torch.log(n_logits)
        # print("log_prob", log_prob)
        p_mask_sum = p_mask.sum(1)
        p_mask_sum_mask = p_mask_sum < 1.0
        if sum(p_mask_sum_mask > 0):
            mean_log_prob = log_prob.sum(1)[~p_mask_sum_mask] / p_mask.sum(1)[~p_mask_sum_mask]
            print(mean_log_prob)
        else:
            mean_log_prob = log_prob.sum(1) / p_mask.sum(1)
        # print("P-mask-sum", p_mask.sum(1))
        # print("p_mask.sum(1)", p_mask.sum(1))
        # print("mean_log_prob", mean_log_prob)
        # print("temperature", self.temperature, self.temperature/self.base_temperature)

        loss = -(self.temperature/self.base_temperature) * mean_log_prob
        loss = loss.mean()
        # print("ct_loss", loss.item())

        return loss

    def _CE_Loss(self, pred, label):
        loss = F.cross_entropy(pred, label)
        return loss

    def NB_loss(self, nb1_pred, nb2_pred,
                      nb1_feat, nb2_feat, 
                      nb1_label, nb2_label,
                      nb_rel, rppr_vec):
        
        loss_track = dict()

        nb_loss = self._NB_loss(nb1_feat, nb2_feat, nb_rel, rppr_vec)
        ce_loss = self._CE_Loss(nb2_pred, nb2_label)
        loss_track["nbct_loss"] = nb_loss.item()
        loss_track["nbce_loss"] = ce_loss.item()

        # print("nb_loss", nb_loss.item())
        
        return nb_loss + ce_loss, loss_track 

    def _NB_loss(self, nb1_feat, nb2_feat, nb_rel, rppr):

        # print(nb1_feat.shape, nb2_feat.shape)
        p_mask = nb_rel.float()

        logits = torch.matmul(nb1_feat, nb2_feat.T)
        logits = torch.div(logits, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        # print(logits.shape, p_mask.shape)
        p_logits = logits * p_mask
        mean_prob = p_logits.sum(1) / p_mask.sum(1)
        # print(mean_prob.shape, rppr.shape)
        loss = -(rppr / self.base_temperature) * mean_prob
        loss = loss.mean()

        return loss

    def recursive_reconstruct(self, nb_blocks:Dict, nb_data:Dict, nb_label:Dict, nb_rel:Dict, rppr_vec:Dict):
        # print(nb_blocks.keys())
        current_block = list(nb_blocks.keys())[0]
        _block = nb_blocks.pop(list(nb_blocks.keys())[0])
        _data = nb_data.pop(list(nb_data.keys())[0])
        _label = nb_label.pop(list(nb_label.keys())[0])

        if len(nb_blocks) == 0:
            loss_track = dict()
            pred, feat =  self.model(_block, _data)
            return pred, feat, _label, loss_track
        
        # Generic case
        _block1, _data1, _label1 = copy.deepcopy(_block), copy.deepcopy(_data), copy.deepcopy(_label)
        _rel  = nb_rel.pop(list(nb_rel.keys())[0])
        _rppr = rppr_vec.pop(list(rppr_vec.keys())[0])
        nb2_pred, nb2_feat, nb2_label, loss_track = self.recursive_reconstruct(nb_blocks, nb_data, nb_label, nb_rel, rppr_vec) # Reference
        nb1_pred, nb1_feat = self.model(_block, _data) # Target

        self.optim.zero_grad()
        loss, _loss_track = self.NB_loss(nb1_pred, nb2_pred, nb1_feat, nb2_feat,
                            nb1_label=_label, nb2_label=nb2_label, nb_rel=_rel, rppr_vec=_rppr)
        loss.backward()
        self.optim.step()
        for key , value in _loss_track.items():
            loss_track[str(current_block)+"_hop_"+str(key)] = value
        _pred, _feat = self.model(_block1, _data1)
        
        return _pred, _feat, _label1, loss_track

    def Check_Stop(self, **kwargs)->dict:
        return self._stop_cond_fn(**kwargs)
    
    def _stop_cond_none(self, **kwargs)->dict:
        # Never check stopping condition
        stat = {}
        stat["terminate"] = False
        return stat

    def _stop_cond_acc(self, **kwargs)->dict:
        stat = {}        
        test, _ = self.evaluate()
        unlearn_acc = test[0]
        test_acc = test[1]

        stat["unlearn_acc"] = unlearn_acc
        stat["test_acc"] = test_acc

        if unlearn_acc <= test_acc:
            stat["terminate"] = True
        else:
            stat["terminate"] = False
        
        return stat
    
    def _stop_cond_dist(self, **kwargs)->dict:
        stat = {}
        self.model.eval()
        ul_pos_sim = list()
        ul_neg_sim = list()
        tst_pos_dist = list()
        tst_neg_dist = list()

        # Set metric:
        # Metric could be min / max / avg (default)
        try:
            metric = kwargs.metric
        except:
            metric = "avg"

        # Obtain avg similarity among neighbors
        for ul_idx, (ul_blocks, nb_blocks, ul_link) in enumerate(self.loader_cnd_ul):
            
            self.model.zero_grad()
        
            _ul_blocks, _ul_data, _ul_label = to_device(ul_blocks, self.device)
            _nb_blocks, _nb_data, _nb_label = to_device(nb_blocks[0], self.device)
            _ul_link = ul_link[0].to(self.device).T # relationship between unlearn nodes & their 1st neighbor
            _ul_label = _ul_label.contiguous().view(-1, 1)
            _nb_label = _nb_label.contiguous().view(-1, 1)
            _p_mask = torch.eq(_ul_label, _nb_label.T)
            _n_mask = ~(_p_mask.clone())

            _ul_pred, _ul_feat = self.model(_ul_blocks, _ul_data)
            _nb_pred, _nb_feat = self.model(_nb_blocks, _nb_data)
            _logits = torch.matmul(_ul_feat, _nb_feat.T) * _ul_link
            _p_logits = _logits * _p_mask
            _n_logits = _logits * _n_mask
            # p_logits = p_logits.sum(dim=1)[mask] / _ul_link.sum(1)[mask]
            p_logits = _p_logits.sum(dim=1) / _ul_link.sum(1)
            n_logits = _n_logits.sum(dim=1) / _ul_link.sum(1)


            # if sum(mask) > 0:

            if metric == "min":
                ul_pos_sim.append(torch.min(p_logits).item())
            elif metric == "max":
                ul_pos_sim.append(torch.max(p_logits).item())
            else:
                ul_pos_sim.append(torch.mean(p_logits).item())
                ul_neg_sim.append(torch.mean(n_logits).item())

        
        # Obtain avg similarity among neighbors
        for tst_idx, (tst_blocks, nb_blocks, ul_link) in enumerate(self.loader_cnd_test):

            _tst_blocks, _tst_data, _tst_label = to_device(tst_blocks, self.device)
            _nb_blocks, _nb_data, _nb_label = to_device(nb_blocks[0], self.device)
            _ul_link = ul_link[0].to(self.device).T
            _tst_label = _tst_label.contiguous().view(-1, 1)
            _nb_label = _nb_label.contiguous().view(-1, 1)
            _p_mask = torch.eq(_tst_label, _nb_label.T)
            _n_mask = ~(_p_mask.clone()) 

            _tst_pred, _tst_feat = self.model(_tst_blocks, _tst_data)
            _nb_pred, _nb_feat = self.model(_nb_blocks, _nb_data)

            # print("tst_pred_shape, nb_pred_shape", _tst_pred.shape, _nb_pred.shape)
            # print("ul_link_sum", _ul_link.sum(1))
            _logits = torch.matmul(_tst_feat, _nb_feat.T) * _ul_link
            _p_logits = _logits * _p_mask
            _n_logits = _logits * _n_mask
            p_logits = _p_logits.sum(dim=1) / _ul_link.sum(1)
            n_logits = _n_logits.sum(dim=1) / _ul_link.sum(1)

            # if sum(mask) > 0:

            if metric == "min":
                tst_pos_dist.append(torch.min(p_logits).item())
            elif metric == "max":
                tst_pos_dist.append(torch.max(p_logits).item())
            else:
                tst_pos_dist.append(torch.mean(p_logits).item())
                tst_neg_dist.append(torch.mean(n_logits).item())

        ul_avg_pos_sim = sum(ul_pos_sim)/len(ul_pos_sim)
        ul_avg_neg_sim = sum(ul_neg_sim)/len(ul_neg_sim)
        tst_avg_pos_sim = sum(tst_pos_dist)/len(tst_pos_dist)
        tst_avg_neg_sim = sum(tst_neg_dist)/len(tst_neg_dist)

        stat["ul_avg_pos_sim"] = ul_avg_pos_sim
        stat["ul_avg_neg_sim"] = ul_avg_neg_sim
        stat["tst_avg_pos_sim"] = tst_avg_pos_sim
        stat["tst_avg_neg_sim"] = tst_avg_neg_sim

        # if tst_avg_sim >= ul_avg_sim:
        #     # Terminate when neighborhood similarity is no more significant than neighborhood similarities of test nodes
        #     stat["terminate"] = True
        # else:
        #     stat["terminate"] = False

        return stat