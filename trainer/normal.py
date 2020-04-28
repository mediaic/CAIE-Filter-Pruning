# from time import time

import torch

from tqdm import tqdm
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NormalTrainer():
    def __init__(self, cfg):
        self.use_cuda = cfg.base.cuda

    def test(self, pack, topk=(1,), mute=True, desc='Test'):
        pack.net.eval()

        info_avgmeter = {'test_loss': AverageMeter()}
        for i, k in enumerate(topk):
            info_avgmeter.update({'acc@{:d}'.format(k): AverageMeter()})

        with tqdm(total=len(pack.test_loader), disable=mute, desc=desc) as pbar:
            for inputs, targets in pack.test_loader:
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                with torch.no_grad():
                    outputs = pack.net(inputs)
                    loss = pack.criterion(outputs, targets)
                    
                    prec = accuracy(outputs, targets, topk=topk)
                    info_avgmeter['test_loss'].update(loss.item(), inputs.size(0))
                    for i, k in enumerate(topk):
                        info_avgmeter['acc@{:d}'.format(k)].update(prec[i].item(), inputs.size(0))
                    
                    pbar.update(1)
                    pbar.set_postfix( {key:'{:.3f}'.format(value.avg) for key, value in info_avgmeter.items()})
        info = {key: value.avg for key, value in info_avgmeter.items()}
        return info


    def train(self, pack, update=True, mute=False, desc=None):
        pack.net.train()

        losses = AverageMeter()

        with tqdm(total=len(pack.train_loader), disable=mute, desc=desc) as pbar:
            for inputs, targets in pack.train_loader:
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                pack.optimizer.zero_grad()

                outputs = pack.net(inputs)
                loss = pack.criterion(outputs, targets)

                losses.update(loss.item(), inputs.size(0))

                loss.backward()
                if update:
                    pack.optimizer.step()

                pbar.update(1)
                pbar.set_postfix({'train_loss': '{:.4f}'.format(losses.avg)})
        info = {'train_loss': losses.avg}
        return info
