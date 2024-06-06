

import argparse
import os,sys
import numpy as np
import torch

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
import pandas as pd
from defense.base import defense
from utils.trainer_cls import Metric_Aggregator
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from tqdm import tqdm
import torch.nn.functional as F

import torchvision.transforms as transforms
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization
from utils.reattack_utils.utils import AverageMeter
from utils.reattack_utils.square_attack import *

# global conditional_target
def dense_to_onehot(y_test, n_cls):
    y_test_onehot = torch.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[torch.arange(len(y_test)), y_test] = True
    return y_test_onehot


def random_classes_except_current(y_test, n_cls):
    y_test_new = torch.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])
        y_test_new[i_img] = torch.random.choice(lst_classes)
    return y_test_new


class BB_attack(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--result_file_defense', type=str,default='None', help='the location of result')
    
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/reattack/config.yaml", help='the path of yaml')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--print_freq', type=int, help='index of clean data')

        #set the parameter for the ft defense
        parser.add_argument("--ratio", type=float, help="ratio of clean samples, used for mix_dataset and legend")
        parser.add_argument("--lr", type=float, help="lr for defense")
        parser.add_argument("--epochs",type=int, help="epochs for defense")
        parser.add_argument("--norm_type", default="L2", type=str,choices=["L_inf","L2","L1"], help="the norm type of the bound")
        parser.add_argument("--save_path", default=None, type=str, help="save path name") 
        parser.add_argument("--clean", action='store_true' , help="use clean model or not")
        
        parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
        parser.add_argument('--p', type=float, default=0.1,
                            help='Probability of changing a coordinate. Note: check the paper for the best values. '
                                'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
        parser.add_argument('--n_iter', type=int, default=10000)
        parser.add_argument('--targeted', action='store_false',default=True, help='Targeted or untargeted attack.')
        parser.add_argument('--target_label', type=int, default=0, help='Targeted label for the attack.')
        parser.add_argument('--max_query', type=int, default=10000, help='Targeted label for the attack.')
        parser.add_argument("--norm", type=float, default=1.0, help='Radius of the Lp ball.') 
        parser.add_argument("--attack_type", type=str, default="score",choices=["score","decision"], help='Radius of the Lp ball.') 
        parser.add_argument("--poison_nums", default=1000, type=int, help="for ablation study")

    def set_result(self, result_file):

        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/reattack/bb_attack/defense_{args.result_file_defense}_Norm_{args.norm_type}_{args.norm}/'
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')
        # exit()

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

    
    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
            ) if torch.cuda.is_available() else "cpu"
        )

    def projection(self, pert, args): 
        if args.norm_type == 'L_inf':
            pert.data = torch.clamp(pert.data, -args.norm , args.norm)
        elif args.norm_type == 'L1':
            norm = torch.sum(torch.abs(pert), dim=(1, 2, 3), keepdim=True)
            for i in range(pert.shape[0]):
                if norm[i] > args.norm:
                    pert.data[i] = pert.data[i] * args.norm / norm[i].item()
        elif args.norm_type == 'L2':
            if len(pert.shape) == 4:
                norm = torch.sum(pert ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
                for i in range(pert.shape[0]):
                    if norm[i] > args.norm:
                        pert.data[i] = pert.data[i] * args.norm / norm[i].item()
            elif len(pert.shape) == 3:
                norm = torch.sum(pert ** 2) ** 0.5
                if norm > args.norm:
                    pert.data = pert.data * args.norm / norm.item()
        else:
            raise NotImplementedError
        return pert

    def train(self, model, train_dataloader,clean_test_dataloader,data_bd_loader):
        device = args.device
        model.eval()
        n_cls = args.num_classes
        if args.attack_type=="score":
            square_attack = square_attack_linf if args.norm_type == 'L_inf' else square_attack_l2
        else:
            raise NotImplementedError
        
        args.loss = 'cross_entropy'
        total_query = 0
        train_asr = 0
        for i_batch,(x,y,*other) in enumerate(train_dataloader):
            x, y = x.to(args.device), y.to(args.device)
            y_target = torch.ones_like(y)*args.target_label
            y_target_onehot = dense_to_onehot(y_target, n_cls=n_cls).to(args.device)
            n_queries, pert,best_asr, train_asr = square_attack(model, x,args.target_label, y_target_onehot, args.norm, args.n_iter,
                                        args.p, args.targeted, args.loss, args.device,self.normalization, self.denormalization)
            total_query = n_queries
            uap_pert = pert
            uap_pert = self.projection(uap_pert,args)
            # cat and save
    
            test_asr_all = AverageMeter()
            test_acc_all = AverageMeter()
            with torch.no_grad():
                uap_v = self.projection(uap_pert,args)
                # compute asr on test dataloader
                for (x,y,*other) in data_bd_loader:
                    x,y = x.to(device), y.to(device)
                    x = self.denormalization(x)
                    delta = uap_v.unsqueeze(0).repeat(len(x), 1, 1, 1).to(device)
                    x_adv = x + delta
                    x_adv = torch.clamp(x_adv, 0, 1)
                    x_adv = self.normalization(x_adv)
                    temp_asr = (model(x_adv).argmax(1) == args.target_label).sum().item()/len(x)
                    test_asr_all.update(temp_asr, x.size(0))
                    temp_acc = (model(x_adv).argmax(1).cpu() == other[2]).sum().item()/len(x)
                    test_acc_all.update(temp_acc, x.size(0))
            logging.info(f"total_query: {total_query}, train_asr: {train_asr.item()}, best_train_asr: {best_asr.item()}, test_asr_all: {test_asr_all.avg}, test_acc_all: {test_acc_all.avg}")
            attack_result = {"uap_pert": uap_pert, "total_query": total_query, "train_asr": train_asr.item(),"best_train_asr":best_asr.item(),"test_asr_all":test_asr_all.avg,"test_acc_all": test_acc_all.avg}
            torch.save(attack_result, args.log+f'n_queries_batch_{i_batch}.pt')

    def reattack(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        args=self.args
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        if args.result_file_defense != "None":
            save_path_defense = "record/" + args.result_file +'/defense/'+ args.result_file_defense
            result_defense = torch.load(save_path_defense+'/defense_result.pt')
            model.load_state_dict(result_defense["model"])
            logging.info(f'successfully load from {save_path_defense}')
        else:
            model.load_state_dict(self.result["model"])
            logging.info("Load attack model")

        if "," in self.args.device:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
        else:
            model.to(self.args.device)
        model.eval()

        train_set = self.result["bd_train"]
        poison_indices = np.where(train_set.poison_indicator == 1)[0]
        poison_select = np.random.choice(poison_indices, args.poison_nums, replace=False)
        train_set.subset(poison_select)
        train_set.wrap_img_transform = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=min(500,args.poison_nums), num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=True)
         
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=256, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)
        for trans_t in test_tran.transforms:
            if isinstance(trans_t, transforms.Normalize):
                denormalizer = get_dataset_denormalization(trans_t)
        self.normalization = trans_t
        self.denormalization = denormalizer

        ## 2. train
        self.train(
            model,
            trainloader,
            data_clean_loader,
            data_bd_loader,
    
        )

        result = {}
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    BB_attack.add_arguments(parser)
    args = parser.parse_args()
    ft_method = BB_attack(args)
    result = ft_method.reattack(args.result_file)

