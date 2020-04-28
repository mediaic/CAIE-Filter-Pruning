import argparse
import os
import json
from utils import dotdict

def make_as_dotdict(obj):
    if type(obj) is dict:
        obj = dotdict(obj)
        for key in obj:
            if type(obj[key]) is dict:
                obj[key] = make_as_dotdict(obj[key])
    return obj

def parse_config():
    print('Parsing config file...')
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/base.json", help="Configuration file to use")
    parser.add_argument("--flops", type=float, default=0.0, help="Maximum proportion of FLOPs left")
    parser.add_argument("--param", type=float, default=0.0, help="Maximum proportion of param left")
    parser.add_argument("--no_caie", action='store_true', help="Applying Constraint-Aware Importance Estimation or not")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--show_cfg", action='store_true', help="Showing the configuration on screen or not")
    args = parser.parse_args()
    # print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    with open(args.config) as fp:
        config = make_as_dotdict(json.loads(fp.read()))
    
    # Overwrite the default configs
    config.prune.caie = (not args.no_caie) and config.prune.caie
    if args.flops > 0:
        config.prune.res_cstr.flops = args.flops
    if args.param > 0:
        config.prune.res_cstr.param = args.param

    if args.show_cfg:
        print(json.dumps(config, indent=4, sort_keys=True))

    return config

class Singleton(object):
    _instance = None
    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kw)  
        return cls._instance 

class Config(Singleton):
    def __init__(self):
        self._cfg = dotdict({})
        self._cfg = parse_config()

    def __getattr__(self, name):
        if name == '_cfg':
            super().__setattr__(name)
        else:
            return self._cfg.__getattr__(name)

    def __setattr__(self, name, val):
        if name == '_cfg':
            super().__setattr__(name, val)
        else:
            self._cfg.__setattr__(name, val)

    def __delattr__(self, name):
        return self._cfg.__delitem__(name)

    def copy(self, new_config):
        self._cfg = make_as_dotdict(new_config)

    def raw(self):
        return self._cfg

cfg = Config()
