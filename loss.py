
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_criterion(cfg):
    pair = {
        'softmax': nn.CrossEntropyLoss()
    }

    assert (cfg.loss.criterion in pair)
    criterion = pair[cfg.loss.criterion]
    return criterion
