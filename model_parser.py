import torch
import torch.nn as nn

import numpy as np
import os, contextlib
from tqdm import tqdm
import time
import json
import copy

from models import get_model
from utils import dotdict
from thop.profile import profile

OBSERVE_TIMES = 2

class Meltable(nn.Module):
    def __init__(self):
        super(Meltable, self).__init__()

    @classmethod
    def melt_all(cls, net):
        def _melt(modules, parant_name):
            keys = modules.keys()
            for k in keys:
                module_name = k if parant_name == '' else  parant_name+'.'+k
                if len(modules[k]._modules) > 0:
                    _melt(modules[k]._modules, parant_name)
                if isinstance(modules[k], Meltable):
                    modules[k] = modules[k].melt()

        _melt(net._modules, '')

    @classmethod
    def observe(cls, pack, use_cuda=True, obs_time=2, eps=1e-3):
        tmp = pack.train_loader

        for m in pack.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.abs_().add_(eps)

        def replace_relu(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    replace_relu(modules[k]._modules)
                if isinstance(modules[k], nn.ReLU):
                    modules[k] = nn.LeakyReLU(inplace=True)
        replace_relu(pack.net._modules)

        pack.net.train()
        for m in pack.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for batch_idx, (inputs, targets) in enumerate(pack.train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                    
            pack.optimizer.zero_grad()
            outputs = pack.net(inputs)
            loss = pack.criterion(outputs, targets)
            loss.backward()
            if batch_idx+1 == obs_time:
                break

        def recover_relu(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    recover_relu(modules[k]._modules)
                if isinstance(modules[k], nn.LeakyReLU):
                    modules[k] = nn.ReLU(inplace=True)
        recover_relu(pack.net._modules)

        for m in pack.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.abs_().add_(-eps)

        pack.train_loader = tmp

    @classmethod
    def reset_mask(cls, net):
        def _reset_mask(modules, parant_name):
            keys = modules.keys()
            for k in keys:
                module_name = k if parant_name == '' else  parant_name+'.'+k
                if len(modules[k]._modules) > 0:
                    _reset_mask(modules[k]._modules, module_name)
                if isinstance(modules[k], Meltable) and not isinstance(modules[k], BN2d_w_mask):
                    modules[k].reset_mask()

        _reset_mask(net._modules, '')
    
    @classmethod
    def set_mask(cls, net, mask_dict):
        def _set_mask(modules, parant_name, mask_dict):
            keys = modules.keys()
            for k in keys:
                module_name = k if parant_name == '' else  parant_name+'.'+k
                if len(modules[k]._modules) > 0:
                    _set_mask(modules[k]._modules, module_name, mask_dict)
                if isinstance(modules[k], Meltable) and not isinstance(modules[k], BN2d_w_mask):
                    modules[k].set_mask(mask_dict[module_name])

        _set_mask(net._modules, '', mask_dict)
    
    @classmethod
    def get_mask_dict(cls, net):
        def _get_mask(modules, parant_name, mask_dict):
            keys = modules.keys()
            for k in keys:
                module_name = k if parant_name == '' else  parant_name+'.'+k
                if len(modules[k]._modules) > 0:
                    _get_mask(modules[k]._modules, module_name, mask_dict)
                if isinstance(modules[k], Meltable):
                    mask_dict[module_name] = modules[k].get_mask()
        mask_dict = dict()
        _get_mask(net._modules, '', mask_dict)
        return mask_dict

class BN2d_w_mask(Meltable):
    def __init__(self, bn):
        super(BN2d_w_mask, self).__init__()
        assert isinstance(bn, nn.BatchNorm2d)
        self.bn = bn
        self.group_id = 0

        self.channel_size = bn.weight.shape[0]
        self.device = bn.weight.device

        self.register_buffer('score', torch.zeros(self.channel_size).to(self.device))
        self.bn.register_buffer('score', torch.zeros(self.channel_size).to(self.device))
        self.register_buffer('bn_mask', torch.ones(1, self.channel_size, 1, 1).to(self.device))

    def set_groupid(self, new_id):
        self.group_id = new_id

    def extra_repr(self):
        return '%d -> %d | ID: %s' % (self.channel_size, int(self.bn_mask.sum()), self.group_id)

    def get_score(self):
        return (self.bn.score * self.bn_mask.view(-1)).cpu().data.numpy()

    def forward(self, x):
        x = self.bn(x)
        if self.bn_mask is not None:
            return x * self.bn_mask
        return x

    def melt(self):
        with torch.no_grad():
            mask = self.bn_mask.view(-1)
            replacer = nn.BatchNorm2d(int(self.bn_mask.sum())).to(self.bn.weight.device)
            replacer.running_var.set_(self.bn.running_var[mask != 0])
            replacer.running_mean.set_(self.bn.running_mean[mask != 0])
            replacer.weight.set_(self.bn.weight[mask != 0])
            replacer.bias.set_(self.bn.bias[mask != 0])
        return replacer
    
    def set_mask(self, mask):
        with torch.no_grad():
            self.bn_mask = mask['bn_mask'].view(1,-1,1,1).to(self.device)

    def get_mask(self):
        return {'bn_mask':self.bn_mask.view(-1)}

    @classmethod
    def transform(cls, net):
        r = []
        def _inject(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    _inject(modules[k]._modules)
                if isinstance(modules[k], nn.BatchNorm2d):
                    modules[k] = BN2d_w_mask(modules[k])
                    r.append(modules[k])
        _inject(net._modules)
        return r

class FinalLinearObserver(Meltable):
    ''' assert was in the last layer. only input was masked '''
    def __init__(self, linear):
        super(FinalLinearObserver, self).__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear
        # self.in_mask = torch.zeros(linear.weight.shape[1]).to('cpu')
        self.in_mask = torch.ones(linear.weight.shape[1]).to('cpu')
        self.f_hook = linear.register_forward_hook(self._forward_hook)
    
    def extra_repr(self):
        return '(%d, %d) -> (%d, %d)' % (
            int(self.linear.weight.shape[1]),
            int(self.linear.weight.shape[0]),
            int((self.in_mask != 0).sum()),
            int(self.linear.weight.shape[0]))

    def _forward_hook(self, m, _in, _out):
        x = _in[0]
        self.in_mask += x.data.abs().cpu().sum(0, keepdim=True).view(-1)

    def forward(self, x):
        return self.linear(x)

    def melt(self):
        with torch.no_grad():
            replacer = nn.Linear(int((self.in_mask != 0).sum()), self.linear.weight.shape[0]).to(self.linear.weight.device)
            replacer.weight.set_(self.linear.weight[:, self.in_mask != 0])
            replacer.bias.set_(self.linear.bias)
        return replacer

    def reset_mask(self):
        with torch.no_grad():
            self.in_mask = self.in_mask*0

    def set_mask(self, mask):
        with torch.no_grad():
            self.in_mask = mask['in_mask']
    
    def get_mask(self):
        return {'in_mask':self.in_mask}
    
    @classmethod
    def transform(cls, net):
        r = []
        def _inject(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    _inject(modules[k]._modules)
                if isinstance(modules[k], nn.Linear):
                    modules[k] = FinalLinearObserver(modules[k])
                    r.append(modules[k])
        _inject(net._modules)
        return r

class Conv2dObserver(Meltable):
    def __init__(self, conv):
        super(Conv2dObserver, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        self.conv = conv
        # self.in_mask = torch.zeros(conv.in_channels).to('cpu')
        # self.out_mask = torch.zeros(conv.out_channels).to('cpu')
        self.in_mask = torch.ones(conv.in_channels).to('cpu')
        self.out_mask = torch.ones(conv.out_channels).to('cpu')
        self.f_hook = conv.register_forward_hook(self._forward_hook)

    def extra_repr(self):
        return '(%d, %d) -> (%d, %d)' % (self.conv.in_channels, self.conv.out_channels, int((self.in_mask != 0).sum()), int((self.out_mask != 0).sum()))
    
    def _forward_hook(self, m, _in, _out):
        x = _in[0]
        self.in_mask += x.data.abs().sum(2, keepdim=True).sum(3, keepdim=True).cpu().sum(0, keepdim=True).view(-1)

    def _backward_hook(self, grad):
        self.out_mask += grad.data.abs().sum(2, keepdim=True).sum(3, keepdim=True).cpu().sum(0, keepdim=True).view(-1)
        new_grad = torch.ones_like(grad)
        return new_grad

    def forward(self, x):
        output = self.conv(x)
        noise = torch.zeros_like(output).normal_()
        output = output + noise
        if self.training:
            output.register_hook(self._backward_hook)
        return output

    def melt(self):
        if self.conv.groups == 1:
            groups = 1
        elif self.conv.groups == self.conv.out_channels:
            groups = int((self.out_mask != 0).sum())
        else:
            assert False

        replacer = nn.Conv2d(
            in_channels = int((self.in_mask != 0).sum()),
            out_channels = int((self.out_mask != 0).sum()),
            kernel_size = self.conv.kernel_size,
            stride = self.conv.stride,
            padding = self.conv.padding,
            dilation = self.conv.dilation,
            groups = groups,
            bias = (self.conv.bias is not None)
        ).to(self.conv.weight.device)

        with torch.no_grad():
            if self.conv.groups == 1:
                replacer.weight.set_(self.conv.weight[self.out_mask != 0][:, self.in_mask != 0])
            else:
                replacer.weight.set_(self.conv.weight[self.out_mask != 0])
            if self.conv.bias is not None:
                replacer.bias.set_(self.conv.bias[self.out_mask != 0])
        return replacer
    
    def reset_mask(self):
        with torch.no_grad():
            self.in_mask = self.in_mask*0
            self.out_mask = self.out_mask*0

    def set_mask(self, mask):
        with torch.no_grad():
            self.in_mask = mask['in_mask']
            self.out_mask = mask['out_mask']
    
    def get_mask(self):
        return {'in_mask':self.in_mask, 'out_mask':self.out_mask}

    @classmethod
    def transform(cls, net):
        r = []
        def _inject(modules):
            keys = modules.keys()
            for k in keys:
                if len(modules[k]._modules) > 0:
                    _inject(modules[k]._modules)
                if isinstance(modules[k], nn.Conv2d):
                    modules[k] = Conv2dObserver(modules[k])
                    r.append(modules[k])
        _inject(net._modules)
        return r


# Mask ID assignment (which BN layers should be pruned as a group)
def resnet_set_group_id(net):
    mask_dict = dict()
    prev_mask_nm = ''
    mask_nm_ls = []
    for name, module in net.named_modules():
        if isinstance(module, BN2d_w_mask):
            mask_dict[name] = {'module':module, 'single_layer': True}
            if prev_mask_nm != '':
                if 'bn1' in name or 'shortcut' in name or 'downsample' in name: # if previous mask is the last bn of the residual block or the bn of skip connection
                    mask_dict[prev_mask_nm]['single_layer'] = False
            else:
                mask_dict[name]['single_layer'] = False
            mask_nm_ls.append(name)
            prev_mask_nm = name
    mask_dict[prev_mask_nm]['single_layer'] = False

    cur_group_id = 0
    last_group_id = 0
    total_group = 0
    for i, name in enumerate(mask_nm_ls[:-1]):
        mask = mask_dict[name]
        if 'layer' not in name:
            mask['module'].set_groupid(cur_group_id)
            total_group += 1
        elif mask['single_layer']:
            cur_group_id = total_group
            mask['module'].set_groupid(cur_group_id)
            total_group += 1
        elif 'bn' not in mask_nm_ls[i+1]:
            cur_group_id = total_group
            last_group_id = cur_group_id
            mask['module'].set_groupid(cur_group_id)
            total_group += 1
        else:
            mask['module'].set_groupid(last_group_id)
    mask_dict[mask_nm_ls[-1]]['module'].set_groupid(last_group_id)

def vgg_set_group_id(net):
    masks = [m for m in net.modules() if isinstance(m, BN2d_w_mask)]
    cur_group_id = 0
    for m in masks:
        m.set_groupid(cur_group_id)
        cur_group_id += 1

def set_group_id(net, model_name):
    if 'resnet' in model_name:
        resnet_set_group_id(net)
    elif 'vgg' in model_name:
        vgg_set_group_id(net)
    else:
        assert False, 'Given model type is NOT supported!!!'

# Categorize the masks (on BN) with group_id
def get_mask_group(masks):
    mask_group = {}
    for m in masks:
        if m.group_id in mask_group.keys():
            mask_group[m.group_id].append(m)
        else:
            mask_group[m.group_id] = [m]
    return mask_group

# Observing the relation ("module_group") between the channels in masks (BN layers) and the channels in other modules (Conv layers, Linear layers)
def build_module_group(pack, cfg):
    if cfg.model.load_module_group and os.path.isfile(cfg.model.module_group_path):
        print('Loading module group information...')
        with open(cfg.model.module_group_path, 'r') as f:
            tmp = json.load(f)
        module_group = {}
        for key in tmp.keys():
            module_group[int(key)] = tmp[key]
    else:
        masks = BN2d_w_mask.transform(pack.net)
        set_group_id(pack.net, cfg.model.name)
        print('Building module group information...')
        mask_group = get_mask_group(masks)
        module_group = {}
        with tqdm(total=len(mask_group.items()), disable=False) as pbar:
            for group_id, group in mask_group.items():
                # Temporarily set the first channel of the masks in the specific group as zero
                with torch.no_grad():
                    orig_mask = group[0].bn_mask.clone().detach()
                    new_mask = torch.ones_like(group[0].bn_mask)
                    new_mask[:,0,:,:] = 0
                    for m in group:
                        m.bn_mask.set_(new_mask)
                
                # Check which masks in other modules are also influenced by the zeroed channel
                cloned_net, _ = clone_model(pack.net, cfg)
                _ = Conv2dObserver.transform(cloned_net)
                _ = FinalLinearObserver.transform(cloned_net)
                cloned_pack = dotdict(pack.copy())
                cloned_pack.net = cloned_net
                Meltable.reset_mask(cloned_pack.net)
                Meltable.observe(cloned_pack, use_cuda=cfg.base.cuda, obs_time=OBSERVE_TIMES, eps=1e-3)
                tmp_mask_dict = Meltable.get_mask_dict(cloned_pack.net)
                
                # Record the influenced masks for every modules
                module_group[group_id] = []
                for module_name, mask in tmp_mask_dict.items():
                    for mask_name in mask.keys():
                        if 'mask' in mask_name:
                            if int((mask[mask_name]==0).sum()) > 0:
                                start = int(torch.nonzero(mask[mask_name]==0)[0])
                                length = orig_mask.numel()
                                m_info = (module_name, mask_name, start, length)
                                module_group[group_id].append(m_info)

                # Check if the model is meltable and able to forward
                Meltable.melt_all(cloned_pack.net)
                res_dict = analyse_model(cloned_pack.net, cfg)

                del tmp_mask_dict
                del cloned_net
                del cloned_pack
                
                # Recover the masks
                with torch.no_grad():
                    for g in group:
                        g.bn_mask.set_(orig_mask)
                pbar.update(1)

        Meltable.melt_all(pack.net)

        if cfg.model.save_module_group:
            with open(cfg.model.module_group_path, 'w') as f:
                json.dump(module_group, f)

    pack.module_group = module_group

# Assign the masks (on BN) to other modules based on "module_group"
def set_module_mask(net, masks, module_group):
    mask_group = get_mask_group(masks)

    mask_dict = Meltable.get_mask_dict(net)
    for group_id, m_info_ls in module_group.items():
        mask = mask_group[group_id][0].bn_mask.clone().detach().view(-1).cpu()
        for module_name, mask_name, start, length in m_info_ls:
            mask_dict[module_name][mask_name][start:start+length] = mask
    Meltable.set_mask(net, mask_dict)

def clone_model(net, cfg):
    
    cloned_net = copy.deepcopy(net)
    cloned_masks = [m for m in cloned_net.modules() if isinstance(m, BN2d_w_mask)]
    
    return cloned_net, cloned_masks

# Get pruned model from model with masks
def get_melted_net(net, module_group, cfg, clone=True):
    if clone:
        cloned_net, cloned_masks = clone_model(net, cfg)
        _ = Conv2dObserver.transform(cloned_net)
        _ = FinalLinearObserver.transform(cloned_net)
        set_module_mask(cloned_net, cloned_masks, module_group)
        Meltable.melt_all(cloned_net)

        return cloned_net
    else:
        masks = [m for m in net.modules() if isinstance(m, BN2d_w_mask)]
        _ = Conv2dObserver.transform(net)
        _ = FinalLinearObserver.transform(net)
        set_module_mask(net, masks, module_group)
        Meltable.melt_all(net)

# Model evaluation (resource consumption)
def analyse_model(net, cfg):
    if 'cifar' in cfg.data.type:
        inputs = torch.randn(1, 3, 32, 32)
    elif 'imagenet' in cfg.data.type:
        inputs = torch.randn(1, 3, 224, 224)
    
    if cfg.base.cuda:
        inputs = inputs.cuda()
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            flops, params = profile(net, (inputs, ))
    res_dict = {'flops':flops, 'param':params}
    return res_dict

def eval_model(net, module_group, cfg):
    res_dict = {}
    masks = [m for m in net.modules() if isinstance(m, BN2d_w_mask)]

    # Calculate # of channels and # of filters
    mask_group = get_mask_group(masks)
    ch_num = sum(int((group[0].bn_mask != 0).sum()) for group in mask_group.values())
    fltr_num = sum(int((group[0].bn_mask != 0).sum())*len(group) for group in mask_group.values())
    res_dict.update({'ch_num':ch_num, 'fltr_num':fltr_num})
    
    # Compute the resource consumption
    melted_net = get_melted_net(net, module_group, cfg, clone=True)
    res_dict.update(analyse_model(melted_net, cfg))

    del melted_net
    return res_dict


def save_model(net, info, ckpt_path):
    checkpoint = {}
    checkpoint.update(info)
    checkpoint.update({'state_dict':net.state_dict()})
    torch.save(checkpoint, ckpt_path)

def load_pruned_model(pack, cfg, ckpt_path):
    # Re-initial a network
    net = get_model(cfg)
    masks = BN2d_w_mask.transform(net)
    set_group_id(net, cfg.model.name)
    mask_group = get_mask_group(masks)
    
    # Reshape the modules in the original model based on checkpoint['ch_ls']
    checkpoint = torch.load(ckpt_path)
    for group_id in mask_group.keys():
        with torch.no_grad():
            new_mask = torch.zeros_like(mask_group[group_id][0].bn_mask)
            if 'ch_ls' in checkpoint.keys():
                ch_ls = checkpoint['ch_ls']
                new_mask[:,:ch_ls[group_id],:,:] = 1

            for m in mask_group[group_id]:
                m.bn_mask.set_(new_mask * m.bn_mask)
    get_melted_net(net, pack.module_group, cfg, clone=False)
    
    # Loading state dict
    net.load_state_dict(checkpoint['state_dict'])
    pack.net = net
    info = {key:checkpoint[key] for key in checkpoint.keys() if key != 'state_dict'}
    return info



