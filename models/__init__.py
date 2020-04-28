import torch
import torchvision
import os

def get_vgg16_for_cifar(cfg):
    from models.cifar.vgg import VGG
    return VGG('VGG16', cfg.model.num_class)

def get_resnet34_for_cifar(cfg):
    from models.cifar.resnet import ResNet34
    return ResNet34(cfg.model.num_class)

def get_resnet50_for_cifar(cfg):
    from models.cifar.resnet import ResNet50
    return ResNet50(cfg.model.num_class)

def get_resnet34_for_imagenet(cfg):
    return torchvision.models.resnet34(pretrained=cfg.model.pretrained)

def get_resnet50_for_imagenet(cfg):
    return torchvision.models.resnet50(pretrained=cfg.model.pretrained)

def get_model(cfg):
    pair = {
        'cifar.vgg16': get_vgg16_for_cifar,
        'cifar.resnet34': get_resnet34_for_cifar,
        'cifar.resnet50': get_resnet50_for_cifar,
        'imagenet.resnet34': get_resnet34_for_imagenet,
        'imagenet.resnet50': get_resnet50_for_imagenet
    }

    model = pair[cfg.model.name](cfg)

    # if os.path.isfile(cfg.model.pretrained_path):
    #     print('Loading pretrained model: ' + cfg.model.pretrained_path)
    #     model.load_state_dict(torch.load(cfg.model.pretrained_path, map_location='cpu' if not cfg.base.cuda else 'cuda'))

    if cfg.base.cuda:
        model = model.cuda()
    
    return model
