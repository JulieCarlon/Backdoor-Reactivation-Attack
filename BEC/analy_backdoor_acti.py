
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from cgitb import handler
import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
from matplotlib.patches import Rectangle, Patch
import matplotlib
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import shap
import yaml
import torch.nn.functional as F
from cProfile import label
from analysis.visual_utils import *
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import load_attack_result

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def get_bd_indicator_by_original_from_bd_dataset(bd_dataset):
    dict_ = {}
    for idx,(img, label, *other_info) in enumerate(bd_dataset):
        dict_[other_info[0]] = idx
    return dict_

def choise_sample(clean_dataset,bd_dataset):
    bd_num = 2200

    visual_samples = []
    visual_labels = []

    origin2bd = get_bd_indicator_by_original_from_bd_dataset(bd_dataset)
    origin2bn = get_bd_indicator_by_original_from_bd_dataset(clean_dataset)

    origin_index_all = list(origin2bd.keys())
    selected_orig = np.random.choice(np.array(origin_index_all),bd_num, replace=False)
    selected_bd_idx = np.array([origin2bd[i] for i in selected_orig])
    selected_bn_idx = np.array([origin2bn[i] for i in selected_orig])
    bn_y, bd_y = [], []
    for bn,bd in zip(selected_bn_idx,selected_bd_idx):
        bn_y.append(clean_dataset[bn][1])
        bd_y.append(bd_dataset[bd][4])
    bd_y = np.array(bd_y)
    bn_y = np.array(bn_y)
    # print((bn_y == bd_y).sum())
    differ_indices = [index for index, (elem1, elem2) in enumerate(zip(bd_y, bn_y)) if elem1 != elem2]

    for i in differ_indices[::-1]:
        selected_bd_idx = np.delete(selected_bd_idx, i)  # 删除那些不相同的index
        selected_bn_idx =  np.delete(selected_bn_idx,i)

    return selected_bn_idx, selected_bd_idx

def main_conv():
    
    # 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args = preprocess_args(args)
    
    fix_random(int(args.random_seed))
    sort_standard = args.sort_standard
    save_path_attack = "./record/" + args.result_file
    if args.result_file_defense != "None":
        save_path_defense = "./record/" + args.result_file +'/defense/'+ args.result_file_defense
        result_defense = torch.load(save_path_defense+'/defense_result.pt')
    save_path = f"save/analy_backdoor_weight_acti/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load data
    result_attack = load_attack_result(save_path_attack + "/attack_result.pt")

    args.model = 'preactresnet18_CLP'
    model = generate_cls_model(args.model, args.num_classes)
    model.load_state_dict(result_attack["model"])
    args.model = 'preactresnet18'
    model_attack = generate_cls_model(args.model, args.num_classes)
    model_attack.load_state_dict(result_attack["model"])
    model_defense = generate_cls_model(args.model, args.num_classes)
    model_defense.load_state_dict(result_defense["model"])
    model.to(args.device)
    model.eval()
    model_attack.to(args.device)
    model_attack.eval()
    model_defense.to(args.device)
    model_defense.eval()

    ##### Plot for Attack #####
    tran = get_transform(
        args.dataset, *([args.input_height, args.input_width]), train=False
    )
    bd_test = result_attack["bd_test"]
    bn_test = result_attack["clean_test"]
    
    clean_test = prepro_cls_DatasetBD_v2(bn_test.wrapped_dataset)
    bn_index, bd_index = choise_sample(clean_test,bd_test)
    clean_test.subset(bn_index)
    bd_test.subset(bd_index)

    bn_test.wrapped_dataset = clean_test
    bn_test.wrap_img_transform = tran
    bd_test.wrap_img_transform = tran
    bd_loader = torch.utils.data.DataLoader(
        bd_test, batch_size=250, num_workers=args.num_workers, shuffle=False
    )
    bn_loader = torch.utils.data.DataLoader(
        bn_test, batch_size=250, num_workers=args.num_workers, shuffle=False
    )
    

    assert len(bd_test) == len(bn_test)
    for (x_bd,y_bd,_,_,_), (x_bn,y_bn,_,_,_) in zip(bd_loader,bn_loader):
        torch.cuda.empty_cache()
        x_bd = x_bd.to(args.device)
        x_bn = x_bn.to(args.device)
        x = torch.cat((x_bd,x_bn))
        x = x.to(args.device)
        output = model(x)
     
    sort_bar_list = []
    name_list = []
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) and ("shortcut" not in name) and ('layer' in name):
            if sort_standard == "diff_relative":
                sort_bar = np.argsort(m.diff_relative.detach().cpu().numpy())
            else:
                sort_bar = np.argsort(m.diff.detach().cpu().numpy())[::-1]
            print(len(sort_bar))
            sort_bar_list.append(sort_bar)
            name_list.append(name)
    
    features_a = []  # 这里拿到所有中间层的特征
    def hook_function(module, input, output):
        nonlocal features_a
        features_a.append(output.detach().cpu())
    for name, module in model_attack.named_modules():
        if isinstance(module, nn.Conv2d) and ("shortcut" not in name) and ('layer' in name):
            module.register_forward_hook(hook_function)
    
    features_d = []
    def hook_function1(module, input, output):
        nonlocal features_d
        features_d.append(output.detach().cpu())
    for name, module in model_defense.named_modules():
        if isinstance(module, nn.Conv2d) and ("shortcut" not in name) and ('layer' in name):
            module.register_forward_hook(hook_function1)
    
    features_a_pool = []
    features_d_pool = []
    bd_loader = torch.utils.data.DataLoader(
        bd_test, batch_size=250, num_workers=args.num_workers, shuffle=True
    )
    for i,(x, y, *other) in enumerate(bd_loader):
        if i == 2:
            break
        x,y = x.to(args.device),y.to(args.device)
        features_a = []
        features_d = []
        _ = model_attack(x)
        _ = model_defense(x)
        features_a_pool.append(features_a)
        features_d_pool.append(features_d)
    ## 得到的是11个batch的列表，每个列表里是一个列表，每个元素是一个中间层的特征，维度为bs*c*h*w
    print(len(sort_bar_list))
    print(len(features_a_pool))
    print(len(features_a_pool[0]))
    features_a_cat_list = []  # 得到的是每层的特征平均的list，每个元素为out*D
    for i,tensors in enumerate(zip(*features_a_pool)):
        # 使用 torch.cat 拼接
        cat_tensor = torch.cat(tensors, dim=0).cpu()
        bs, out, h, w = cat_tensor.size()
        # 将拼接后的张量加到结果列表
        cat_tensor = cat_tensor.view(bs, -1).numpy() # 对数量做平均，得到out*D
        # print(len(cat_tensor))
        # print(len(sort_bar_list[i]))
        # cat_tensor = cat_tensor[sort_bar_list[i]]
        features_a_cat_list.append(cat_tensor)

    features_d_cat_list = []
    for i,tensors in enumerate(zip(*features_d_pool)):
        # 使用 torch.cat 拼接
        cat_tensor = torch.cat(tensors, dim=0).cpu()
        bs, out, h, w = cat_tensor.size()
        # 将拼接后的张量加到结果列表
        cat_tensor = cat_tensor.view(bs, -1).numpy()# 对数量做平均，得到out*D
        # cat_tensor = cat_tensor[sort_bar_list[i]]
        features_d_cat_list.append(cat_tensor)
    for i,fmap in enumerate(features_a_cat_list):
        print(i, fmap.shape)
    # plt.style.use('default')
    # fig, axs = plt.subplots(4, 4, figsize=(11, 10))
    # axs = axs.reshape(-1)

    
    # for i,fmap in enumerate(features_a_cat_list):
    #     img = axs[i].imshow(fmap, cmap='magma', aspect='auto')
    #     axs[i].set_xticks([])
    #     axs[i].set_yticks([])
    #     axs[i].spines['top'].set_visible(False)
    #     axs[i].spines['right'].set_visible(False)
    #     axs[i].set_title(name_list[i])
    #     axs[i].grid()
           
    # # plt.colorbar(img, ax=axs.ravel().tolist(), orientation='vertical')
    # fig.colorbar(img, ax=[axs[i] for i in range(16)], fraction=0.1, pad=0.03)
    # plt.suptitle('defense vs attack for tac from large to small for Conv')
    # # plt.tight_layout()
    # plt.savefig(save_path+f"Plot_Conv_AvsD_sort_{sort_standard}_attack_{args.result_file}_map.png",bbox_inches='tight')
    # plt.close()

    # plt.style.use('default')
    # fig, axs = plt.subplots(4, 4, figsize=(11, 10))
    # axs = axs.reshape(-1)

    # for i,fmap in enumerate(features_d_cat_list):
    #     img = axs[i].imshow(fmap, cmap='magma', aspect='auto')
    #     axs[i].set_xticks([])
    #     axs[i].set_yticks([])
    #     axs[i].spines['top'].set_visible(False)
    #     axs[i].spines['right'].set_visible(False)
    #     axs[i].set_title(name_list[i])
    #     axs[i].grid()

    # # plt.colorbar(img, ax=axs.ravel().tolist(), orientation='vertical')
    # fig.colorbar(img, ax=[axs[i] for i in range(16)], fraction=0.1, pad=0.03)
    # plt.suptitle('defense vs attack for tac from large to small for Conv')
    # # plt.tight_layout()
    # plt.savefig(save_path+f"Plot_Conv_AvsD_sort_{sort_standard}_attack_{args.result_file}_defense_{args.result_file_defense}_map.png",bbox_inches='tight')
    # plt.close()

    torch.save(features_a_cat_list,save_path+f"Tensor_Conv_AvsD_sort_{sort_standard}_attack_{args.result_file}_map.pth")
    torch.save(features_d_cat_list,save_path+f"Tensor_Conv_AvsD_sort_{sort_standard}_attack_{args.result_file}_defense_{args.result_file_defense}_map.pth")


def main_conv_clean():
    args = get_args()
    with open(args.yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args = preprocess_args(args)
    fix_random(int(args.random_seed))
    sort_standard = args.sort_standard
    save_path_attack = "./record/" + args.result_file
    save_path = f"save/analy_backdoor_weight_acti/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load data
    result_attack = load_attack_result(save_path_attack + "/attack_result.pt")

    args.model = 'preactresnet18_CLP'
    model = generate_cls_model(args.model, args.num_classes)
    model.load_state_dict(result_attack["model"])
    args.model = 'preactresnet18'
    model_attack = generate_cls_model(args.model, args.num_classes)
    model_attack.load_state_dict(result_attack["model"])
    model_defense = generate_cls_model(args.model, args.num_classes)
    ckpt = torch.load("resource/clean_model/cifar10_preactresnet18/clean_model.pth") # !!!! clean model 的ckpt
    model_defense.load_state_dict(ckpt)
    model.to(args.device)
    model.eval()
    model_attack.to(args.device)
    model_attack.eval()
    model_defense.to(args.device)
    model_defense.eval()

    ##### Plot for Attack #####
    tran = get_transform(
        args.dataset, *([args.input_height, args.input_width]), train=False
    )
    bd_test = result_attack["bd_test"]
    bn_test = result_attack["clean_test"]
    
    clean_test = prepro_cls_DatasetBD_v2(bn_test.wrapped_dataset)
    bn_index, bd_index = choise_sample(clean_test,bd_test)
    clean_test.subset(bn_index)
    bd_test.subset(bd_index)

    bn_test.wrapped_dataset = clean_test
    bn_test.wrap_img_transform = tran
    bd_test.wrap_img_transform = tran
    bd_loader = torch.utils.data.DataLoader(
        bd_test, batch_size=250, num_workers=args.num_workers, shuffle=False
    )
    bn_loader = torch.utils.data.DataLoader(
        bn_test, batch_size=250, num_workers=args.num_workers, shuffle=False
    )
    assert len(bd_test) == len(bn_test)
    for (x_bd,y_bd,_,_,_), (x_bn,y_bn,_,_,_) in zip(bd_loader,bn_loader):
        torch.cuda.empty_cache()
        x_bd = x_bd.to(args.device)
        x_bn = x_bn.to(args.device)
        x = torch.cat((x_bd,x_bn))
        x = x.to(args.device)
        output = model(x)

    sort_bar_list = []
    name_list = []
    tac_list = []
    for (name, m) in model.named_modules():
        if isinstance(m, nn.Conv2d) and ("shortcut" not in name) and ('layer' in name):
            if sort_standard == "diff_relative":
                sort_bar = np.argsort(m.diff_relative.detach().cpu().numpy())
                tac_list.append(m.diff_relative.detach().cpu().numpy())
            else:
                sort_bar = np.argsort(m.diff.detach().cpu().numpy())[::-1]
                tac_list.append(m.diff.detach().cpu().numpy())
            print(len(sort_bar))
            sort_bar_list.append(sort_bar)
            name_list.append(name)
    
    save_dict = {"sort_bar_list":sort_bar_list,"tac_list":tac_list,"name_list":name_list}
    torch.save(save_dict, save_path+f"Z_TAC_Conv_AvsD_sort_{sort_standard}_{args.result_file}_map.pth")
    del save_dict

    features_a = []  # 这里可以拿到所有中间层的特征
    def hook_function(module, input, output):
        nonlocal features_a
        features_a.append(output.detach().cpu())
    for name, module in model_attack.named_modules():
        if isinstance(module, nn.Conv2d) and ("shortcut" not in name) and ('layer' in name):
            module.register_forward_hook(hook_function)
    
    features_d = []
    def hook_function1(module, input, output):
        nonlocal features_d
        features_d.append(output.detach().cpu())
    for name, module in model_defense.named_modules():
        if isinstance(module, nn.Conv2d) and ("shortcut" not in name) and ('layer' in name):
            module.register_forward_hook(hook_function1)
    
    features_a_pool = []
    features_d_pool = []
    bd_loader = torch.utils.data.DataLoader(
        bd_test, batch_size=250, num_workers=args.num_workers, shuffle=True
    )
    for i,(x, y, *other) in enumerate(bd_loader):
        if i ==2:
            break
        x,y = x.to(args.device),y.to(args.device)
        features_a = []
        features_d = []
        _ = model_attack(x)
        _ = model_defense(x)
        features_a_pool.append(features_a)
        features_d_pool.append(features_d)
    ## 得到的是11个batch的列表，每个列表里是一个列表，每个元素是一个中间层的特征，维度为bs*c*h*w

    features_d_cat_list = []
    for i,tensors in enumerate(zip(*features_d_pool)):
        # 使用 torch.cat 拼接
        cat_tensor = torch.cat(tensors, dim=0).cpu()
        bs, out, h, w = cat_tensor.size()
        # 将拼接后的张量加到结果列表
        cat_tensor = cat_tensor.view(bs, -1).numpy()# 对数量做平均，得到out*D
        features_d_cat_list.append(cat_tensor)
    for a in features_d_cat_list:
        print(a.shape)
    torch.save(features_d_cat_list,save_path+f"Tensor_Conv_AvsD_sort_{sort_standard}_model_clean_data_{args.result_file}_map.pth")

def analyze_features():
    args = get_args()
    with open(args.yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args = preprocess_args(args)
    fix_random(int(args.random_seed))
    sort_standard = args.sort_standard
    save_path_attack = "./record/" + args.result_file
    save_path = f"defense/attackD/visualization/analy_backdoor_weight_acti_new/"

    args.result_file
    args.result_file_defense
    if args.result_file_defense == "None":
        defense_tensor = torch.load(save_path+f"Tensor_BN_AvsD_sort_{sort_standard}_model_clean_data_{args.result_file}_map.pth")
    else:
        defense_tensor = torch.load(save_path+f"Tensor_Conv_AvsD_sort_{sort_standard}_attack_{args.result_file}_defense_{args.result_file_defense}_map.pth")

    attack_tensor = torch.load(save_path+f"Tensor_Conv_AvsD_sort_{sort_standard}_attack_{args.result_file}_map.pth")

    tac_list = torch.load(save_path+f"Z_TAC_Conv_AvsD_sort_{sort_standard}_{args.result_file}_map.pth")
    sort_bar_list = tac_list["sort_bar_list"]
    name_list = tac_list["name_list"]
    tac_list = tac_list["tac_list"] 

    

if __name__ == "__main__":
    main_conv() ## 对防御和攻击模型，计算并画图，并保存计算出来的tensor
    main_conv_clean()  ## 对干净模型，计算并画图，并保存计算出来的tensor


