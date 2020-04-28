
import numpy as np
import torch


import os
import json
import pickle

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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

class ExpMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, size, mom = 0.9):
        self.size = size
        self.mom = mom
        self.reset()
        

    def reset(self):
        self.val = np.zeros(self.size)
        self.sum = np.zeros(self.size)
        self.mean_avg = np.zeros(self.size)
        self.exp_avg = np.zeros(self.size)
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean_avg = self.sum / self.count
        self.exp_avg = self.mom*self.exp_avg + (1.0 - self.mom)*self.val
        if self.count-n == 0:
            self.exp_avg = self.val

def print_info(base_info, left_info, desc='', print_ch_ls=False, logger=None):
    info_str = {}
    _str = desc
    if 'train_loss' in left_info.keys():
        _str =  _str + '\t Train Loss: %.3f' % (left_info['train_loss'])
    if 'flops' in left_info.keys():
        info_str.update({
                'flops': '[%.2f%%] %.3f MFLOPs' % (left_info['flops']/base_info['flops'] * 100, left_info['flops'] / 1e6),
                'param': '[%.2f%%] %.3f M' % (left_info['param']/base_info['param'] * 100, left_info['param'] / 1e6),
                'fltr_num': '[%.2f%%] %d' % (left_info['fltr_num']/base_info['fltr_num'] * 100, left_info['fltr_num']),
                'ch_num': '[%.2f%%] %d' % (left_info['ch_num']/base_info['ch_num'] * 100, left_info['ch_num'])
        })
        _str = _str + '\t FLOPs: %s,\t Param: %s,\t # of ch: %s,\t # of F: %s' % (info_str['flops'], info_str['param'], info_str['ch_num'], info_str['fltr_num'])
    if 'test_loss' in left_info.keys():
        _str =  _str + '\t Test Loss: %.3f' % (left_info['test_loss'])
    if 'acc@1' in left_info.keys():
        if 'acc@1' in base_info.keys():
            info_str.update({'acc@1': '[%.2f%%\u2193] %.2f%%' % (base_info['acc@1']-left_info['acc@1'], left_info['acc@1'])})
        else:
            info_str.update({'acc@1': '%.2f%%' % (left_info['acc@1'])})
        _str =  _str + '\t Test Acc: ' + info_str['acc@1']

    print(_str)
    if logger is not None:
        logger.save_log(_str)

    if 'ch_ls' in left_info.keys() and print_ch_ls:
        print(left_info['ch_ls'])

class Logger():
    def __init__(self, cfg, overwrite=True):
        self.cfg = cfg
        cstr_str = '_'.join([res_type+'_'+str(cstr) for res_type, cstr in cfg.prune.res_cstr.items()])
        if not cfg.prune.caie:
            cstr_str = cstr_str+'_no_caie'
        self.base_path = os.path.join('./checkpoint', cfg.base.task_name, cstr_str)
        self.logfile = os.path.join(self.base_path, 'log.txt')
        self.cfgfile = os.path.join(self.base_path, 'cfg.json')

        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
            if not os.path.isfile(self.logfile) or overwrite:
                with open(self.logfile, 'w') as fp:
                    fp.write('')
            with open(self.cfgfile, 'w') as fp:
                json.dump(cfg.raw(), fp)

    def save_record(self, epoch, record):
        with open(self.logfile) as fp:
            log = json.load(fp)

        log[str(epoch)] = record
        with open(self.logfile, 'w') as fp:
            json.dump(log, fp)

    def save_info(self, info, info_fn):
        infofile = os.path.join(self.base_path, info_fn)
        with open(infofile, 'wb') as f:
            pickle.dump(info, f)
    
    def save_log(self, _str):
        with open(self.logfile, 'a') as fp:
            fp.write(_str+'\n')






