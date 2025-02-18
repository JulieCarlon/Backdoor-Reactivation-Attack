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

def analysis_similarity():
    ## 读取所有的结果
    save_path = f"save/analy_backdoor_weight_acti/"
    attacks = ["badnet","blended",'inputaware','lf', 'ssba',"trojannn",'wanet']
    attack_list = [f"cifar10_preactresnet18_{attack}_0_1" for attack in attacks]
    defense_list = ['ft', 'nc','i-bau', 'ft-sam', 'sau', 'nad','fst', 'None']

    total_result_list = [[] for _ in range(len(attacks))]
    layer = "Conv"
    sort_standard = "diff"
    
    for idx, attack in enumerate(attack_list):
        total_sim_result = torch.load(save_path+f"save_similarity_features_{attack}_{layer}_{sort_standard}.pth")
        total_result_list[idx].append(total_sim_result)

    metric_dict = {}
    for idx_a,attack in enumerate(attack_list):
        total_sim_result = total_result_list[idx_a][0]
        metric_dict[attack] = {}
        for idx_d,defense in enumerate(defense_list):
            metric_dict[attack][defense] = {}
            # metric_dict[attack][defense]["CKA"] = np.array(total_sim_result[f"{defense}_CKA"]).mean()
            metric_dict[attack][defense]["top5_cka"] = np.array(total_sim_result[f"{defense}_top5_cka"]).mean()
            metric_dict[attack][defense]["top10_cka"] = np.array(total_sim_result[f"{defense}_top10_cka"]).mean()
            metric_dict[attack][defense]["top20_cka"] = np.array(total_sim_result[f"{defense}_top20_cka"]).mean()
    
    # 将metric写入txt并保存为dict
    with open(save_path+f"zz_all_similarity_features_metric.txt", "a+") as f:
        f.write(f'layer: {layer} and sort standard: {sort_standard}\n')
        for key,value in metric_dict.items():
            f.write(f'{key}: {value}\n')
        f.write("\n")


    torch.save(metric_dict, save_path+ f"zz_all_save_all_similarity_features_metric_{layer}_{sort_standard}.pth")
    print(f"save to  {save_path}zz_all_save_all_similarity_features_metric_{layer}_{sort_standard}.pth")
    return metric_dict


if __name__ == "__main__":
    analysis_similarity() # 统计相似度；这里是对所有攻击放一起求bec
 
