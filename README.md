# Constraint-Aware Importance Estimation for Global Filter Pruning under Multiple Resource Constraints
This repo contains required scripts to reproduce results from paper:  
  
Constraint-Aware Importance Estimation for Global Filter Pruning under Multiple Resource Constraints  
Yu-Cheng Wu, Chih-Ting Liu, Bo-Ying Chen, Shao-Yi Chien.  
EDLCV 2020 (CVPRW 2020)

## Requirements
python 3.6+ and PyTorch 1.0+

## Usage
Run `main.py` to get a pruned model given the resource constraints (maximum proportion of FLOPs and params left), 
multiple constriants given is availiable:

    python3 main.py --config CONFIG_FILE [--options]
Options:
 - `--config`: the path of the configuration file, default: `./configs/ImageNet_resnet50_f50.json`
 - `--flops`: FLOPs cosntraint, it would be ignored if the value is invalid (≥ 1 or ≤ 0), default: `1.0`
 - `--param`: params constraint, it would be ignored if the value is invalid (≥ 1 or ≤ 0), default: `1.0`
 - `--no_caie`: add this option if not applying CAIE
 - `--gpu_id`: set the GPU id, default: `'0'`
 - `--show_cfg`: show the configuration on screen or not
 
 We provide several configuration files for different models (resnet50, resnet34, vgg16) in different datasets (ImageNet, CIFAR10).
 You can modify the config file is necessary.

## Citation
If you use this code for your research, please cite our papers:
```
@{
}
```
