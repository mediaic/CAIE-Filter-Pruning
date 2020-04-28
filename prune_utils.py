import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import math
from tqdm import tqdm

from model_parser import get_mask_group, eval_model
from utils import AverageMeter, ExpMeter, print_info


# Hook functions for loss impact estimation
def te_BN2d(m, grad_in, grad_out):
    score = m.weight*grad_in[1]
    score += m.bias*grad_in[2]
    m.score = score

def te_BN2d_sq(m, grad_in, grad_out):
    score = (m.weight*grad_in[1])**2
    score += (m.bias*grad_in[2])**2
    m.score = score

def te_BN2d_abs(m, grad_in, grad_out):
    score = (m.weight*grad_in[1]).abs()
    score += (m.bias*grad_in[2]).abs()
    m.score = score

TE_hooks = {
    nn.BatchNorm2d: {
        "sum_of_sq": te_BN2d_sq,
        "sq_of_sum": te_BN2d,
        "sq_of_sum2": te_BN2d,
        "sum_of_abs": te_BN2d_abs,
        "abs_of_sum": te_BN2d,
        "abs_of_sum2": te_BN2d
    }
}

def set_te_hooks(model, te_method):
    te_hooks = []
    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m_type = type(m)
        b_hook = None
        # if isinstance(m, nn.BatchNorm2d):
        if m_type in TE_hooks.keys():
            b_hook = TE_hooks[m_type][te_method]
            assert m.score is not None
            handler = m.register_backward_hook(b_hook)
            # m.weight.register_hook(show_grad)
            te_hooks.append(handler)
    
    model.apply(add_hooks)
    return te_hooks

def rm_te_hooks(te_hooks):
    for handler in te_hooks:
        handler.remove()

class Prune_engine(object):
    def __init__(self, pack, masks, base_res, cfg):
                
        self.cfg = cfg
        self.pack = pack
        self.masks = masks

        self.base_res = base_res
        self.left_res = self.base_res.copy()
        self.res_cstr = cfg.prune.res_cstr # rescource constraints for pruning (maximum ratio lefted)

        # self.parser = Model_Parser(self.pack, self.BNs, self.cfg)
        self.mask_group = get_mask_group(masks)
        # print(self.mask_group)

        # pruning config
        self.num_to_pr = cfg.prune.num_to_pr # the number of channels to prune in one pruning step
        self.update_param = cfg.prune.update_param # update parameters or not when computing loss impact
        self.set_min_ch()

        # loss impact gathering
        self.loss_imp_dict = {}
        self.te_method = cfg.prune.te_method # method to calculate loss impact
        self.pr_step = cfg.prune.pr_step # iterations of gathering in one pruning step
        self.loss_imp_avg = cfg.prune.avg_type # the type of the average of loss impact

        # resource impact gathering
        self.caie = cfg.prune.caie # applying CAIE or not
        if self.caie:
            self.num_for_probe = cfg.prune.num_for_probe # the number of channel decreasing when evaluating resource impacts
            self.pr_objs = {res_type:max((base_res[res_type]-self.base_res[res_type]*cstr)/base_res[res_type], 0) for res_type, cstr in self.res_cstr.items()}
            self.res_imps = {}
            self.eff_res_imp = {}

    def set_min_ch(self):
        self.min_ch_dict = dict()
        for group_id, masks in self.mask_group.items():
            ch = int(masks[0].bn_mask.numel())
            self.min_ch_dict[group_id] = max(1, int(ch*self.cfg.prune.min_ch_ratio))

    # Model Information
    def update_left_res(self):
        self.left_res = eval_model(self.pack.net, self.pack.module_group, self.cfg)

    def reach_cstr(self):
        return all(self.left_res[res_type] < self.base_res[res_type]*cstr for res_type, cstr in self.res_cstr.items())

    # Loss Impact Estimation  
    def init_loss_impacts(self):
        for group_id, group in self.mask_group.items():
            ch = int(group[0].bn_mask.numel())
            self.loss_imp_dict[group_id] = ExpMeter(ch)

    def update_loss_impacts(self):
        for group_id in self.loss_imp_dict.keys():
            score_np = np.stack([m.get_score() for m in self.mask_group[group_id]]) # axis 0: different layers

            if self.te_method == "sum_of_sq" or self.te_method == "sum_of_abs":
                group_score = score_np.sum(axis=0)
            elif self.te_method == "sq_of_sum":
                group_score = (score_np**2).sum(axis=0)
            elif self.te_method == "sq_of_sum2":
                group_score = score_np.sum(axis=0)**2
            elif self.te_method == "abs_of_sum":
                group_score = score_np.abs().sum(axis=0)
            elif self.te_method == "abs_of_sum2":
                group_score = score_np.sum(axis=0).abs()

            self.loss_imp_dict[group_id].update(group_score, 1)

    # Resource Impacts Estimation
    def update_pr_objs(self):
        self.update_left_res()
        for res_type, cstr in self.res_cstr.items():
            self.pr_objs[res_type] = max((self.left_res[res_type]-self.base_res[res_type]*cstr)/self.left_res[res_type], 0)
            assert self.pr_objs[res_type] >= 0

    def probe_single_layer(self, group_id):
        
        # Temporarily set some (self.num_for_probe) alive channels of the masks in mask_group[group_id] as zero
        with torch.no_grad():
            orig_bn_mask = self.mask_group[group_id][0].bn_mask.clone().detach()

            nonzero_id = torch.nonzero(orig_bn_mask.view(-1) != 0).view(-1)
            if nonzero_id.numel() <= self.num_for_probe:
                return
            new_mask = torch.ones_like(orig_bn_mask)
            new_mask[:,nonzero_id[:self.num_for_probe],:,:] = 0

            for m in self.masks:
                if m.group_id == group_id:
                    m.bn_mask.set_(new_mask * m.bn_mask)

        # Check the resource consumption of the new model
        new_left_res = eval_model(self.pack.net, self.pack.module_group, self.cfg)

        # Resume the changed channels
        with torch.no_grad():
            for m in self.masks:
                if m.group_id == group_id:
                    m.bn_mask.set_(orig_bn_mask)
        
        # Calculate resource impacts for mask_group[group_id]
        if group_id not in self.res_imps.keys():
            self.res_imps[group_id] = {}
        for res_type in self.res_cstr.keys():
            self.res_imps[group_id][res_type] = (self.left_res[res_type]-new_left_res[res_type])/(self.left_res[res_type]*self.num_for_probe)
            assert self.res_imps[group_id][res_type] > 0, (group_id, res_type, self.left_res[res_type], new_left_res[res_type])
    
    def get_eff_res_imp(self):
        with tqdm(total=len(self.mask_group.keys()), disable=False, desc='Res imp') as pbar:
            for group_id in self.mask_group.keys():
                self.probe_single_layer(group_id)
                pr_obj_norm = math.sqrt(sum(pr_obj**2 for res_type, pr_obj in self.pr_objs.items()))
                self.eff_res_imp[group_id] = sum(pr_obj*self.res_imps[group_id][res_type] for res_type, pr_obj in self.pr_objs.items())/pr_obj_norm
                assert self.eff_res_imp[group_id] > 0
                pbar.update(1)

    # Importance Estimation
    def importance_estimation(self):
        importance_dict = {}
        for group_id, loss_imp in self.loss_imp_dict.items():
            if self.loss_imp_avg == 'exp':
                importance_dict[group_id] = loss_imp.exp_avg
            else:
                importance_dict[group_id] = loss_imp.mean_avg
                
        if self.caie:
            self.update_pr_objs()
            self.get_eff_res_imp()
            for group_id in importance_dict.keys():
                importance_dict[group_id] = importance_dict[group_id]/self.eff_res_imp[group_id]
        return importance_dict
    
    # Pruning
    def set_new_mask(self, importance_dict):
        filtered_imp_list = []
        for group_id, imp in importance_dict.items():

            sorted_imp = np.sort(imp)[:-self.min_ch_dict[group_id]]
            filtered_imp = sorted_imp[sorted_imp != 0] # collecting nonzero importance scores
            filtered_imp_list.append(filtered_imp)

        imps = np.concatenate(filtered_imp_list)
        global_threshold = np.sort(imps)[self.num_to_pr-1]

        new_masks = {}
        for group_id, imp in importance_dict.items():
            hard_threshold = float(np.sort(imp)[-self.min_ch_dict[group_id]])
            new_masks[group_id] = ((imp >= hard_threshold) + (imp > global_threshold))

        with torch.no_grad():
            for m in self.masks:
                mask = torch.from_numpy(new_masks[m.group_id].astype('float32')).to(m.device).view(1, -1, 1, 1)
                m.bn_mask.set_(mask * m.bn_mask)

    def prune(self, test=True):
        
        def set_optimizer():
            self.pack.optimizer = optim.SGD(
                self.pack.net.parameters(),
                lr=self.cfg.prune.lr,
                momentum=self.cfg.prune.momentum,
                weight_decay=self.cfg.prune.weight_decay,
                nesterov=self.cfg.prune.nesterov
            )
       
        if self.update_param:
            self.pack.net.train()
            set_optimizer()
        else:
            self.pack.net.eval()
        
        pr_logs = []

        losses = AverageMeter()
        self.init_loss_impacts()
        te_hooks = set_te_hooks(self.pack.net, self.te_method)
        step = 0
        pbar = tqdm(total=self.pr_step, desc='Loss imp')

        while True:
            for batch_idx, (inputs, targets) in enumerate(self.pack.train_loader):
                if self.cfg.base.cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                self.pack.net.zero_grad()
                outputs = self.pack.net(inputs)
                loss = self.pack.criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))

                loss.backward()
                self.update_loss_impacts()
                
                if self.update_param:
                    self.pack.optimizer.step()

                pbar.update()
                pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg)})

                # A step of pruning 
                if (batch_idx+1)%self.pr_step == 0:
                    pbar.close()
                    importance_dict = self.importance_estimation()
                    self.set_new_mask(importance_dict)
                    self.update_left_res()

                    # Storing pruning information
                    info = {'pr_step':'{:d}'.format(step+1)}
                    info.update(self.left_res)
                    info.update({'ch_ls':[int((group[0].bn_mask != 0).sum()) for group_id, group in self.mask_group.items()]})
                    if test:
                        info.update(self.pack.trainer.test(self.pack, mute=(self.cfg.data.type!="imagenet")))
                    pr_logs.append(info)

                    # Check if the model has reached the given constraints
                    if self.reach_cstr():
                        if not test:
                            info.update(self.pack.trainer.test(self.pack, mute=(self.cfg.data.type!="imagenet")))
                        rm_te_hooks(te_hooks)
                        return pr_logs, info

                    print_info(self.base_res, info, desc='Step %s'%info['pr_step'], logger=self.pack.logger)

                    # Preparing for next step of pruning
                    if self.update_param:
                        set_optimizer()
                    losses = AverageMeter()
                    self.init_loss_impacts()
                    step += 1
                    pbar = tqdm(total=self.pr_step, desc='Loss imp')
        

