
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


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

  
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def sortxy(x,y,z,cmap):
    sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
    x_sorted = [x[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]
    z_sorted = [z[i] for i in sorted_indices]
    cmap_sorted = [cmap[i] for i in sorted_indices]
    return x_sorted, y_sorted, z_sorted,cmap_sorted

def plot_2dim():
    save_path = f"save/analy_backdoor_weight_acti/"
    sort_standard = "diff" 
    layer = "Conv"
    metric_dict = torch.load(save_path+f"zz_all_save_all_similarity_features_metric_{layer}_{sort_standard}.pth")
    
    import pandas as pd
    # 从csv文件中读取数据
    data = pd.read_csv(save_path+'cifar10_attack_defense.csv')

    read_data = {}
    for defense in ['attack','ft', 'nc', 'nad', 'i-bau', 'ft-sam', 'sau', 'fst']:
        read_data[defense] = data[defense].values

    attacks = ["badnet","blended",'inputaware','lf', 'ssba',"trojannn",'wanet']
    attack_list = [f"{attack}_attack_cifar10_default" for attack in attacks]
    defense_list = ['None', 'ft','nc','nad', 'i-bau', 'ft-sam', 'sau', 'fst']
    
    from matplotlib import pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    markersize=8
    plt.rc('font',family='Times New Roman')
    plt.style.use('default')
    
    cmap=["coral","silver","pink","yellowgreen","mediumturquoise","thistle","skyblue","goldenrod"]
    cmap=["#F18072","#8CD0C3","#BCB9D8","#80B1D2","#F9B063","#BA7FB5","#FAF5B5","#F7CBDF"]
    cmap=["C3","#8CD0C3","#BCB9D8","#80B1D2","#F9B063","#BA7FB5","#FAF5B5","#F7CBDF"]

    mark_list = ['o','>',r'$\clubsuit$',(5,0),(5,1),'s','^']

    key_name = "top10_cka"

    plt.figure(figsize=(6, 5))
    plt.subplot(1,1,1)
    for idx,attack in enumerate(attacks):
        simi_list = []
        for defense in defense_list:
            simi_list.append(metric_dict[attack_list[idx]][defense][key_name])

        simi_list = np.array(simi_list)
        norm_simi_list = (simi_list - simi_list.min())/(simi_list.max()-simi_list.min())
        norm_simi_list[0] = 1 # 这里是因为，dict中的第一个是clean,但我们要的是attack，所以改成1
        asr_list = data.iloc[idx][1:].values.astype(float)
        asr_list,norm_simi_list,defense_name,cmap_sorted = sortxy(asr_list,norm_simi_list,defense_list,cmap)
        # print(asr_list)
        # print(norm_simi_list)
        # print(defense_name)
        for i in reversed(range(len(asr_list))):
            # print(idx,i)
            if cmap_sorted[i] == cmap[0]:
                plt.scatter(asr_list[i], norm_simi_list[i],color=cmap_sorted[i], s=100,marker=mark_list[idx], edgecolor='black',linewidths=1)
            else:
                plt.scatter(asr_list[i], norm_simi_list[i],color=cmap_sorted[i], s=100,marker=mark_list[idx], edgecolor='black',linewidths=0.5)
            # plt.text(asr_list[i], norm_simi_list[i],defense_name[i], fontsize=9,ha='center', va='bottom')
            # ax.text(x[i], y[i], label, fontsize=8, ha='center', va='bottom')

        # plt.scatter(asr_list, norm_simi_list,color=cmap[idx], s=100, edgecolor='black',label=f'{attack}',linewidths=0.5)
        # plt.plot(asr_list, norm_simi_list, color=cmap[idx], linestyle='--',  markersize=markersize,linewidth=1, alpha=0.6)
    
    

    # Creating custom legend for markers (column-wise)
    attacks = ["Badnet","Blended",'InputAware','LF', 'SSBA',"Trojannn",'WaNet']
    defense_list = ['Attack', 'FT','NC','NaD', 'i-BAU', 'FT-SAM', 'SAU', 'FST']
    for i, color in enumerate(cmap):
        plt.scatter([], [], color=color, label=f'{defense_list[i]}', s=60)

    for j, marker in enumerate(mark_list):
        plt.scatter([], [], color='none', edgecolors='black', marker=marker, label=f'{attacks[j]}', s=60)
        
    plt.xticks((np.array([0,0.2,0.4,0.6,0.8,1.0])).tolist())
    plt.yticks((np.array([0.3,0.4,0.6,0.8,1.0])).tolist())
    # plt.xticks(rotation=0)
    plt.xlabel('Backdoor Activation Rate\n(a)',fontsize=16)
    plt.ylabel('Backdoor Existence Coefficient',fontsize=16)
    plt.grid()

    plt.legend(bbox_to_anchor=(0.2, 0.35),ncol=3)
    # plt.legend(bbox_to_anchor=(0.6, 0.8),ncol=2)
    plt.savefig(save_path+f"{key_name}.png",bbox_inches='tight')
    plt.savefig(save_path+f"{key_name}.pdf",bbox_inches='tight')
    

if __name__ == "__main__":

    plot_2dim()
 

# python defense/attackD/analysis/analy_backdoor_weight_acti.py --result_file_defense fst --result_file cifar10_preactresnet18_badnet_0_1 --device cuda:3
