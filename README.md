# Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack

## Licenses
You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper and indicating any changes that you've made.

## Requirements
see file `./sh/install.sh` for installing the PyTorch environment.
```
conda create -n reattack python=3.8
conda activate reattack
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

## Datasets.
We conduct experiments on CIFAR-10, GTSRB, and Tiny ImageNet datasets. For CIFAR-10, it can be downloaded automatically. For GTSRB and Tiny ImageNet datasets, please use `./dataset/xx.py` to download them.

## Method
### 1. Preparation for attack and defense
THe first step is to train the state-of-the-art backdoor attack and defense methods. At first, run the attack files to generate a backdoored model. For example, run `BadNets` attack on CIFAR-10 dataset:

```
python ./attack/badnet.py --save_folder_name cifar10_preactresnet18_badnet_0_1 --dataset cifar10
```
Then a defense model can be trained based on the backdoored model. For example, run `NAD` defense on CIFAR-10 dataset:


```
python ./defense/nad.py --result_file cifar10_preactresnet18_badnet_0_1 --dataset cifar10
```
The structure of this repository is heavily based on [BackdoorBench](https://github.com/SCLBD/BackdoorBench). We follow [BackdoorBench](https://github.com/SCLBD/BackdoorBench) on the implementation of SOTA attack and defense methods. For the implementation details of SOTA attacks and defenses, please refer to [BackdoorBench](https://github.com/SCLBD/BackdoorBench) for help.

### 2. Training our method
With the defense model, we can implement our backdoor re-activation attack method as follows:

For running the white-box method, run the following script:
```
python ./reattack/wb_attack.py --result_file cifar10_preactresnet18_badnet_0_1 --result_file_defense nad --dataset cifar10 --model preactresnet18 --norm 0.05 --norm_type L_inf --outer_steps 50 --inner_steps 5 
```

For running the black-box method, run the following script:
```
python ./reattack/bb_attack.py --result_file cifar10_preactresnet18_badnet_0_1 --result_file_defense nad --dataset cifar10 --model preactresnet18 --norm 0.05 --norm_type L_inf
```

For running the transfer attack method, you should have at least two defense models, such as nad, nc, and i-bau, then run the following script:
```
python ./reattack/ta_attack.py --result_file cifar10_preactresnet18_badnet_0_1 --result_file_defense i-bau --norm 0.05 --norm_type L_inf --attack_type avg --target_models nad nc
```

We wil release the implementation of our reactivation attack on CLIP model as long as our paper has been accepted.

## Acknowledgment
Our project references the codes in the following repos.
- [BackdoorBench](https://github.com/SCLBD/BackdoorBench)
