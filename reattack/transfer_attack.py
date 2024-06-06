

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
from defense.base import defense
from utils.trainer_cls import Metric_Aggregator
from utils.trainer_cls import BackdoorModelTrainer, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
import torch.nn.functional as F
import pandas as pd
import torchvision.transforms as transforms
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization
from utils.reattack_utils.utils import AverageMeter
import copy, math

class TA_attack(defense):

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

        parser.add_argument("--ratio", type=float, help="ratio of clean samples, used for mix_dataset and legend")
        parser.add_argument("--lr", type=float, help="lr for defense")
        parser.add_argument("--epochs",type=int, help="epochs for defense")
        parser.add_argument("--norm", type=float,default=1.0, help="the norm bound of the perturbation")
        parser.add_argument("--norm_type", default="L2", type=str,choices=["L_inf","L2","L1"], help="the norm type of the bound")
        parser.add_argument("--out_step", default=100, type=int,help="the step for generate adversarial examples")
        parser.add_argument("--inner_step", default=1, type=int,help="the step for generate adversarial examples")
        parser.add_argument("--max_init", action='store_true', default=False, help="the norm of the bound")
        parser.add_argument("--save_path", default=None, type=str, help="save path name") 
        parser.add_argument("--clean", action='store_true' , help="use clean model or not")
        parser.add_argument('--result_file_defense', nargs='+', type=str, help='the location of result')
        parser.add_argument('--target_models', nargs='+', type=str, help='the location of result')
        parser.add_argument('--attack_type', type=str,default="avg", choices=["avg","mom","step"], help='avg is avg of grad, mom is momentum, step is use diff model in diff step')
        parser.add_argument('--target_label', type=int,default=0, help='avg is avg of grad, mom is momentum, step is use diff model in diff step')
        
    def set_result(self, result_file):
        result_file_defense_cat = "_".join(args.result_file_defense)
        target_models_cat = "_".join(args.target_models)
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/reattack/ta_attack/defense_{args.result_file_defense}_Norm_{args.norm_type}_{args.norm}/'
        self.args.save_path = save_path
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

    def eval_dataloader(self,pert, dataloader):
        model = self.model
        total_bd_acc, total_bd_asr = AverageMeter(), AverageMeter()
        pert = pert.detach()
        total_bd_test, bd_acc, bd_asr = 0, 0, 0
        for i, (inputs, labels,*other) in enumerate(dataloader):
            inputs, labels = inputs.to(args.device), other[2].to(args.device)
            batch_pert_imgs = self.get_perturbed_image(inputs, pert).detach()
            outputs = model.forward(batch_pert_imgs)
            bd_acc += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            bd_asr += torch.sum(torch.argmax(outputs[:], dim=1) == args.target_label)
            total_bd_test += inputs.shape[0]
        bd_acc = bd_acc / total_bd_test
        bd_asr = bd_asr / total_bd_test
    
        total_bd_acc.update(bd_acc.item())
        total_bd_asr.update(bd_asr.item())
        return total_bd_asr.avg,total_bd_acc.avg


    def get_perturbed_image(self, images, pert):
        images_wo_trans = self.denormalization(images) + pert
        images_wo_trans = images_wo_trans.clamp(0, 1)
        images_with_trans = self.normalization(images_wo_trans)
        return images_with_trans

    def projection(self, pert, args): 
        if args.norm_type == 'L_inf':
            pert.data = torch.clamp(pert.data, -args.trigger_norm , args.trigger_norm)
        elif args.norm_type == 'L1':
            norm = torch.sum(torch.abs(pert), dim=(1, 2, 3), keepdim=True)
            for i in range(pert.shape[0]):
                if norm[i] > args.trigger_norm:
                    pert.data[i] = pert.data[i] * args.trigger_norm / norm[i].item()
        elif args.norm_type == 'L2':
            norm = torch.sum(pert ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
            for i in range(pert.shape[0]):
                if norm[i] > args.trigger_norm:
                    pert.data[i] = pert.data[i] * args.trigger_norm / norm[i].item()
        else:
            raise NotImplementedError
        return pert


    def train(self, model, train_dataloader,clean_test_dataloader,data_bd_loader):
        args.rand_init = True
        args.max_init = False
        args.adv_lr = 0.05
        if args.rand_init:
            batch_pert = torch.rand(size=[1,args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)
            batch_pert.data = batch_pert.data * 2 * args.norm - args.norm
            batch_pert.data = self.projection(batch_pert.data, args)
        elif args.max_init:
            batch_pert = torch.zeros([1,args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)
            batch_pert.data += args.norm
        else:
            batch_pert = torch.zeros([1,args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)
        batch_opt = torch.optim.SGD(params=[batch_pert], lr=args.adv_lr)
        agg = Metric_Aggregator()
        if args.attack_type == "avg":
            for out in range(1, args.out_step+1):
                losses = AverageMeter()
                top1 = AverageMeter()
                for batch_idx, (batch_x, batch_y, *other) in enumerate(train_dataloader):
                    for inner in range(1,args.inner_step+1):
                        images = batch_x.to(args.device)
                        labels = batch_y.to(args.device) 
                        bsz = batch_y.shape[0]
                        target_lab = torch.ones_like(labels)*args.target_label
                        pert_images = self.get_perturbed_image(images, batch_pert)
                        loss = 0
                        train_asr = 0
                        for model in self.models:
                            per_logits = model.forward(pert_images)
                            loss +=  F.cross_entropy(per_logits, target_lab)
                            train_asr += torch.sum(torch.argmax(per_logits[:], dim=1) == args.target_label)/bsz
                        train_asr /= len(self.models)
                        batch_opt.zero_grad()
                        loss.backward()
                        batch_opt.step() 
                        batch_pert = self.projection(batch_pert, args)
                    losses.update(loss.item(), bsz)
                    top1.update(train_asr.item(), bsz)
                if out % 20 == 0 or out == args.out_step:
                    for ii,target in enumerate(args.target_models):
                        self.model = generate_cls_model(self.args.model,self.args.num_classes)
                        result_defense = torch.load("record/" + args.result_file +'/defense/'+ target +'/defense_result.pt')
                        self.model.load_state_dict(result_defense["model"])
                        self.model.eval()
                        self.model.to(args.device)
                        logging.info(f'successfully load target defense model {target}')

                        total_bd_asr,total_bd_acc = self.eval_dataloader(batch_pert.detach().clone(),data_bd_loader)
                        logging.info(f"out_step:{out}, target model: {target} train_loss:{losses.avg}, train_acc:{top1.avg}, bd_acc:{total_bd_acc}, bd_asr:{total_bd_asr}")
                        agg(
                            {   "out_step": out,
                                "target_model": ii,
                                "train_loss": losses.avg,
                                "train_acc": top1.avg,
                                "bd_acc" : total_bd_acc,
                                "bd_asr" : total_bd_asr,
                            }
                        )
                        agg.to_dataframe().to_csv(f"{args.save_path}result_df.csv")
 
        elif args.attack_type == "step":
            for out in range(1, args.out_step+1):
                losses = AverageMeter()
                top1 = AverageMeter()
                for batch_idx, (batch_x, batch_y,  *other) in enumerate(train_dataloader):
                    images = batch_x.to(args.device)
                    labels = batch_y.to(args.device) 
                    bsz = batch_y.shape[0]
                    target_lab = torch.ones_like(labels)*args.target_label
                    for inner in range(1,args.inner_step+1):
                        for model in self.models:
                            pert_images = self.get_perturbed_image(images, batch_pert)
                            per_logits = model.forward(pert_images)
                            train_asr = torch.sum(torch.argmax(per_logits[:], dim=1) == args.target_label)/bsz
                            loss =  F.cross_entropy(per_logits, target_lab)
                            batch_opt.zero_grad()
                            loss.backward()
                            batch_opt.step() 
                            batch_pert = self.projection(batch_pert, args)
                            losses.update(loss.item(), bsz)
                            top1.update(train_asr.item(), bsz)
                if out % 20 == 0 or out == args.out_step:
                    for ii,target in enumerate(args.target_models):
                        self.model = generate_cls_model(self.args.model,self.args.num_classes)
                        result_defense = torch.load("record/" + args.result_file +'/defense/'+ target +'/defense_result.pt')
                        self.model.load_state_dict(result_defense["model"])
                        self.model.eval()
                        self.model.to(args.device)
                        logging.info(f'successfully load target defense model {target}')

                        total_bd_asr,total_bd_acc = self.eval_dataloader(batch_pert.detach().clone(),data_bd_loader)
                        logging.info(f"out_step:{out}, target model: {target} train_loss:{losses.avg}, train_acc:{top1.avg}, bd_acc:{total_bd_acc}, bd_asr:{total_bd_asr}")
                        agg(
                            {   "out_step": out,
                                "target_model": ii,
                                "train_loss": losses.avg,
                                "train_acc": top1.avg,
                                "bd_acc" : total_bd_acc,
                                "bd_asr" : total_bd_asr,
                            }
                        )
                        agg.to_dataframe().to_csv(f"{args.save_path}result_df.csv")

        elif args.attack_type == "mom":
            self.mu = 0.9
            inner_momentum = torch.zeros_like(batch_pert).to(args.device)
            for out in range(1, args.out_step+1):
                losses = AverageMeter()
                top1 = AverageMeter()
                for batch_idx, (batch_x, batch_y,  *other) in enumerate(train_dataloader):
                    images = batch_x.to(args.device)
                    labels = batch_y.to(args.device) 
                    bsz = batch_y.shape[0]
                    for inner in range(1,args.inner_step+1):
                        target_lab = torch.ones_like(labels)*args.target_label
                        for model in self.models:
                            batch_pert.requires_grad = True
                            pert_images = self.get_perturbed_image(images, batch_pert)
                            per_logits = model.forward(pert_images)
                            train_asr = torch.sum(torch.argmax(per_logits[:], dim=1) == args.target_label)/bsz
                            loss =  F.cross_entropy(per_logits, target_lab)
                            batch_opt.zero_grad()
                            loss.backward()
                            grad = batch_pert.grad
                            inner_momentum = self.mu * inner_momentum - grad / torch.sum(grad ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
                            batch_pert.requires_grad = False
                            batch_pert += args.adv_lr * inner_momentum
                            batch_pert = self.projection(batch_pert, args)
                            losses.update(loss.item(), bsz)
                            top1.update(train_asr.item(), bsz)
                if out % 20 == 0 or out == args.out_step:
                    for ii,target in enumerate(args.target_models):
                        self.model = generate_cls_model(self.args.model,self.args.num_classes)
                        result_defense = torch.load("record/" + args.result_file +'/defense/'+ target +'/defense_result.pt')
                        self.model.load_state_dict(result_defense["model"])
                        self.model.eval()
                        self.model.to(args.device)
                        logging.info(f'successfully load target defense model {target}')

                        total_bd_asr,total_bd_acc = self.eval_dataloader(batch_pert.detach().clone(),data_bd_loader)
                        logging.info(f"out_step:{out}, target model: {target} train_loss:{losses.avg}, train_acc:{top1.avg}, bd_acc:{total_bd_acc}, bd_asr:{total_bd_asr}")
                        agg(
                            {   "out_step": out,
                                "target_model": ii,
                                "train_loss": losses.avg,
                                "train_acc": top1.avg,
                                "bd_acc" : total_bd_acc,
                                "bd_asr" : total_bd_asr,
                            }
                        )
                        agg.to_dataframe().to_csv(f"{args.save_path}result_df.csv")
        
        agg.summary().to_csv(f"{args.save_path}result_df_summary.csv")
        reattack_result = {"target_label": args.target_label, "bd_acc": total_bd_acc, "bd_asr": total_bd_asr, "uap_pert":batch_pert.detach().cpu()}
        torch.save(reattack_result, f"{args.log}reattack_result.pth")

    def reattack(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        args=self.args
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        args.num_defenses = len(args.result_file_defense)
        model = generate_cls_model(self.args.model,self.args.num_classes)
        self.models = []
        for defense in args.result_file_defense:
            result_defense = torch.load("record/" + args.result_file +'/defense/'+ defense +'/defense_result.pt')
            model.load_state_dict(result_defense["model"])
            model.eval()
            self.models.append(copy.deepcopy(model).to(self.args.device))
            logging.info(f'successfully load defense {defense}')
        
        train_set = self.result["bd_train"]
        poison_indices = np.where(train_set.poison_indicator == 1)[0]
        poison_select = np.random.choice(poison_indices, min(5000,len(poison_indices)), replace=False)
        train_set.subset(poison_select)
        train_set.wrap_img_transform = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        if args.dataset == "tiny":
            batch_size = 256
        else:
            batch_size = 500
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=True)
        trainloader = data_loader 
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

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
            data_bd_loader)

        result = {}
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    TA_attack.add_arguments(parser)
    args = parser.parse_args()
    ft_method = TA_attack(args)
    result = ft_method.reattack(args.result_file)


