# Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack

This is the official repository for the paper "Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack" by Mingli Zhu, Siyuan Liang, and Baoyuan Wu. 

For more details, please refer to our Paper ([NeurIPS 2024](https://openreview.net/pdf?id=E2odGznGim)).

The structure of this repository is heavily based on [BackdoorBench](https://github.com/SCLBD/BackdoorBench). We follow [BackdoorBench](https://github.com/SCLBD/BackdoorBench) on the implementation of SOTA attack and defense methods, besides its code architecture and environment. For more implementation details of SOTA attacks and defenses, please refer to [BackdoorBench](https://github.com/SCLBD/BackdoorBench).

If you have any questions on this repo, please feel free to email the first author [Mingli Zhu](mailto:minglizhu@link.cuhk.edu.cn).

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
Our method requires a backdoored model that has undergone defense (referred to as a defense model). If you already have such a model, you can skip Steps 1 and 2 and proceed directly to our re-activation attack implementation.

### Step 1. Backdoor attack
First, train a backdoored model. For example, to run `BadNets` attack on CIFAR-10 dataset:

```
python ./attack/badnet.py --save_folder_name cifar10_preactresnet18_badnet_0_1 --dataset cifar10
```
### Step 2. Preparation for attack and defense
Next, apply defense methods to the backdoored model. For example, to run `NAD` defense on CIFAR-10 dataset:

```
python ./defense/nad.py --result_file cifar10_preactresnet18_badnet_0_1 --dataset cifar10
```

### Step 3. Training our method
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
python ./reattack/ta_attack.py --result_file cifar10_preactresnet18_badnet_0_1 --result_file_defense i-bau --norm 0.05 --norm_type L_inf --attack_type avg --target_models nad
```

You can customize all hyperparameters according to your needs or refer to the original paper for recommended settings.

## BEC Metric
This work proposes the Backdoor existence coefficient to measure the persistent existence of backdoor in models. The calculation includes three steps:

### Step 1. Backdoor neuron identification and the feature map of backdoor neuron.
```
python BEC/analy_backdoor_weight_acti.py --result_file_defense fst --result_file cifar10_preactresnet18_badnet_0_1 --device cuda:3 --sort_standard diff
```

### Step 2. Backdoor effect similarity metric by CKA
```
python BEC/analy_backdoor_cka.py --result_file_defense fst --result_file cifar10_preactresnet18_badnet_0_1 --device cuda:3 --sort_standard diff
```
### Step 3. Backdoor existence coefficient computation
```
python BEC/analy_backdoor_bec.py --device cuda:3 --sort_standard diff
```
Note: Before proceeding with this step, please ensure you have calculated all metrics for the attacks and defenses you are interested in.


If you use this paper/code in your research, please consider citing us:

```
@inproceedings{
zhu2024breaking,
title={Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack},
author={Mingli Zhu and Siyuan Liang and Baoyuan Wu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=E2odGznGim}
}
```

## Acknowledgment
Our project references the codes in the following repos.
- [BackdoorBench](https://github.com/SCLBD/BackdoorBench)

