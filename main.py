import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import math
import os

from loader import get_loader
from models import get_model
from trainer import get_trainer
from loss import get_criterion


from config import cfg
from utils import dotdict, print_info, Logger
from model_parser import BN2d_w_mask, set_group_id, build_module_group, eval_model, get_melted_net, save_model, load_pruned_model
from prune_utils import Prune_engine

# Preparing optimizer and lr scheduler for training
def _step_lr(epoch):
    r = 1.0
    for ep_milestone, ratio in cfg.train.steplr:
        r = ratio
        if epoch < ep_milestone:
            break
    return r

def get_lr_func():
    if cfg.train.steplr is not None:
        return _step_lr
    else:
        assert False

def set_optimizer(pack):
    if cfg.train.optim == 'sgd' or cfg.train.optim is None:
        pack.optimizer = optim.SGD(
            pack.net.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            nesterov=cfg.train.nesterov
        )
    else:
        print('WRONG OPTIM SETTING!')
        assert False
    pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())

# Preprocessing (preparing dataloaders, model, training function, etc.)
def set_seeds(cfg):
    if cfg.base.deterministic:
        torch.manual_seed(cfg.base.seed)
        if cfg.base.cuda:
            torch.cuda.manual_seed_all(cfg.base.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(cfg.base.seed)
        random.seed(cfg.base.seed)
    else:
        torch.backends.cudnn.benchmark = True

def get_pack(cfg):
    set_seeds(cfg)
    train_loader, test_loader = get_loader(cfg)
    pack = dotdict({
        'net': get_model(cfg),
        'train_loader': train_loader,
        'test_loader': test_loader,
        'trainer': get_trainer(cfg),
        'criterion': get_criterion(cfg),
        'optimizer': None,
        'lr_scheduler': None,
        'module_group': None,
        'logger': Logger(cfg)
    })
    set_optimizer(pack)

    build_module_group(pack, cfg)
    
    if os.path.isfile(cfg.model.pretrained_path) and cfg.model.pretrained:
        print('Loading pretrained model: ' + cfg.model.pretrained_path)
        pack.net.load_state_dict(torch.load(cfg.model.pretrained_path, map_location='cpu' if not cfg.base.cuda else 'cuda'))
    
    masks = BN2d_w_mask.transform(pack.net)
    set_group_id(pack.net, cfg.model.name)

    return pack, masks

# Fine-tuning process
def finetune(pack, cfg, base_info):
    best_acc = 0
    best_info = {}
    best_ckpt_path = os.path.join(pack.logger.base_path, 'best.ckpt')
    for epoch in range(cfg.train.max_epoch):
        info = pack.trainer.train(pack, desc='Ep {:d}'.format(epoch+1))
        info.update(pack.trainer.test(pack, mute=(cfg.data.type!="imagenet")))
        pack.lr_scheduler.step()
        print_info(base_info, info, desc='Epoch {:d}'.format(epoch+1), logger=pack.logger)
        if best_acc < info['acc@1']:
            print('Get the best model!!!')
            best_acc = info['acc@1']
            best_info.update(info)
            save_model(pack.net, info, best_ckpt_path)
    print_info(base_info, best_info, desc='Best', logger=pack.logger)


def main():
    pack, masks = get_pack(cfg)
    print('Pruning constraints:', cfg.prune.res_cstr, ', with CAIE:', cfg.prune.caie, '\n')
    
    base_info = {}
    base_res = eval_model(pack.net, pack.module_group, cfg)
    base_info.update(base_res)
    base_info.update(pack.trainer.test(pack, mute=False))

    print_info(base_info, base_info, desc='Original', logger=pack.logger)
    
    prune_engine = Prune_engine(pack, masks, base_res, cfg)
    LOGS, info = prune_engine.prune(test=False)

    print_info(base_info, info, desc='Pruned', logger=pack.logger)
    
    # Get pruned network
    get_melted_net(pack.net, pack.module_group, cfg, clone=False)
    set_optimizer(pack)
    
    ckpt_path = os.path.join(pack.logger.base_path, 'pruned.ckpt')
    save_model(pack.net, info, ckpt_path)

    finetune(pack, cfg, base_info)

if __name__ == '__main__':
    main()