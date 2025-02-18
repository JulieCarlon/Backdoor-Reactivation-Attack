'''
这个实验可视化tac排序之下的，神经元weight的相似性，这个相似性map的横轴为attack model，纵轴为defense model
'''
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
from analysis.CKA_similarity_torch import linear_CKA
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

  
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit


def similarity_features():
    args = get_args()
    with open(args.yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args = preprocess_args(args)
    fix_random(int(args.random_seed))
    sort_standard = args.sort_standard
    save_path = f"save/analy_backdoor_weight_acti/"

    layer = "Conv" 
    attack_tensor = torch.load(save_path+f"Tensor_{layer}_AvsD_sort_{sort_standard}_attack_{args.result_file}_map.pth")
    tac_list = torch.load(save_path+f"Z_TAC_{layer}_AvsD_sort_{sort_standard}_{args.result_file}_map.pth")
    
    sort_bar_list = tac_list["sort_bar_list"]
    name_list = tac_list["name_list"]
    value_list = tac_list["tac_list"]

    defense_list = ['ft','nc','i-bau', 'ft-sam', 'sau', 'nad','fst', 'None']
    defense_tensor_list = []

    for defense in defense_list:
        if defense == "None":
            defense_tensor = torch.load(save_path+f"Tensor_{layer}_AvsD_sort_{sort_standard}_model_clean_data_{args.result_file}_map.pth")
        else:
            defense_tensor = torch.load(save_path+f"Tensor_{layer}_AvsD_sort_{sort_standard}_attack_{args.result_file}_defense_{defense}_map.pth")

        defense_tensor_list.append(copy.deepcopy(defense_tensor))
    del defense_tensor

    print(f'attack: {args.result_file}')
    print(f'layer: {layer} and sort standard: {sort_standard}')

    total_sim_result = dict()
    
    
    for ref_defense in defense_list:
        if os.path.exists(save_path+f"Tensor_{layer}_AvsD_sort_{sort_standard}_attack_{args.result_file}_defense_{ref_defense}_map.pth"):
            ref_shape_tensors = torch.load(save_path+f"Tensor_{layer}_AvsD_sort_{sort_standard}_attack_{args.result_file}_defense_{ref_defense}_map.pth")
        else:
            raise ValueError("None of defense models exists.")
    
    # 计算10%或者20%的神经元的相似性
    for i in range(len(defense_list)):
        defense_name = defense_list[i]
        defense_tensor = defense_tensor_list[i]
        temp_top5_cka_sim_list = []
        temp_top10_cka_sim_list = []
        temp_top20_cka_sim_list = []
        for i_layer,(attack, defense) in enumerate(zip(attack_tensor, defense_tensor)):
            c = ref_shape_tensors[i_layer].shape[0]
            attack = torch.tensor(attack).reshape(500,c,-1).transpose(0,1)
            defense = torch.tensor(defense).reshape(500,c,-1).transpose(0,1)
            temp_tac = torch.tensor(value_list[i_layer])
            top_values, top_indices5 = torch.topk(temp_tac, k=int(len(temp_tac)*0.05))
            top_values, top_indices10 = torch.topk(temp_tac, k=int(len(temp_tac)*0.1))
            top_values, top_indices20 = torch.topk(temp_tac, k=int(len(temp_tac)*0.2))
            temp_top5_cka_sim_list.append(linear_CKA(attack[top_indices5].transpose(0,1).reshape(500,-1).to(args.device), defense[top_indices5].transpose(0,1).reshape(500,-1).to(args.device)).item())
            temp_top10_cka_sim_list.append(linear_CKA(attack[top_indices10].transpose(0,1).reshape(500,-1).to(args.device), defense[top_indices10].transpose(0,1).reshape(500,-1).to(args.device)).item())
            temp_top20_cka_sim_list.append(linear_CKA(attack[top_indices20].transpose(0,1).reshape(500,-1).to(args.device), defense[top_indices20].transpose(0,1).reshape(500,-1).to(args.device)).item())

        total_sim_result[defense_name+"_top5_cka"] = copy.deepcopy(temp_top5_cka_sim_list)
        total_sim_result[defense_name+"_top10_cka"] = copy.deepcopy(temp_top10_cka_sim_list)
        total_sim_result[defense_name+"_top20_cka"] = copy.deepcopy(temp_top20_cka_sim_list)

    with open(save_path+f"zz_similarity_features_{args.result_file}.txt", "a+") as f:
        for key,value in total_sim_result.items():
            f.write(f'{key}: {value}\n')
        f.write("\n")
    
    total_sim_result['attack'] = args.result_file
    total_sim_result['layer'] = layer
    total_sim_result['tac'] = sort_standard
    torch.save(total_sim_result, save_path+f"save_similarity_features_{args.result_file}_{layer}_{sort_standard}.pth")

if __name__ == "__main__":
    similarity_features()  # 计算到一个值并统计相似度； 这里是对每个攻击单独求，放在一个.pth里
 

