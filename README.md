# Constraint-Aware Importance Estimation for Global Filter Pruning under Multiple Resource Constraints
This is the official repository of our [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Wu_Constraint-Aware_Importance_Estimation_for_Global_Filter_Pruning_Under_Multiple_Resource_CVPRW_2020_paper.pdf):  
  
Constraint-Aware Importance Estimation for Global Filter Pruning under Multiple Resource Constraints  
[Yu-Cheng Wu](https://github.com/ericwu2620), [Chih-Ting Liu](https://github.com/jackie840129), Bo-Ying Chen, Shao-Yi Chien.  
Joint Workshop on Efficient Deep Learning in Computer Vision (in conjunction with CVPR 2020) [[link]](https://workshop-edlcv.github.io/) 

## Notification

- **We will add more instructions of the usage soon !**

## Requirements
- Python 3.6+
- PyTorch 1.2+ (We test the code under version 1.2)

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
@inproceedings{wu2020constraint,
  title={Constraint-Aware Importance Estimation for Global Filter Pruning Under Multiple Resource Constraints},
  author={Wu, Yu-Cheng and Liu, Chih-Ting and Chen, Bo-Ying and Chien, Shao-Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={686--687},
  year={2020}
}
```
