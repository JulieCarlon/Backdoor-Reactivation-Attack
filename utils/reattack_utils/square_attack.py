import argparse
import time
import numpy as np
import models
import os
import utils
from datetime import datetime
import logging
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import Tensor as t
import torch.nn.functional as F
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def pseudo_gaussian_pert_rectangles(x, y):
    delta = torch.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= torch.sqrt(torch.sum(delta ** 2, dim=(0,1), keepdim=True))

    return delta

def meta_pseudo_gaussian_pert(s):
    delta = torch.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= torch.sqrt(torch.sum(delta ** 2,dim=(0,1), keepdims=True))
        if np.random.rand(1) > 0.5: delta = torch.transpose(delta,0,1)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= torch.sqrt(torch.sum(delta ** 2, dim=(0,1), keepdim=True))
    return delta

def projection_img(x,pert,normalization,denormalization,device):
    x = denormalization(x)
    delta = pert.unsqueeze(0).repeat(len(x), 1, 1, 1).to(device)
    x_adv = x + delta
    x_adv = torch.clamp(x_adv, 0, 1)
    x_adv = normalization(x_adv)
    return x_adv

def comp_loss( y, logits, targeted=False, loss_type='margin_loss'):
    """ Implements the margin loss (difference between the correct and 2nd best class). """
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits 
        diff[y] = torch.inf 
        margin = diff.min(1, keepdims=True)[0]
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = F.softmax(logits, dim=1)
        loss = -torch.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')

    return loss.reshape(-1)

def square_attack_l2(model, x,target_label, y, eps, n_iters, p_init, targeted, loss_type,device,normalization,denormalization):
    """ The L2 square attack """
    np.random.seed(0)
    c, h, w = x.shape[1:]
    n_features = c * h * w
    delta_init = torch.zeros(x.shape[1:])
    s = h // 5
    logging.info('Initial square side={} for bumps'.format(s))
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, center_h:center_h + s, center_w:center_w + s] += (meta_pseudo_gaussian_pert(s).reshape(
                [1, s, s]) * torch.Tensor(np.random.choice([-1, 1], size=[c, 1, 1]))) 
            center_w += s
        center_h += s
    delta_init = delta_init/(torch.sum(delta_init ** 2) ** 0.5) * eps
    x_best = projection_img(x,delta_init,normalization,denormalization,device=device)
    with torch.no_grad():
        logits = model(x_best)
    best_asr = (logits.argmax(1) == target_label).sum()/x_best.size(0)
    loss_min_curr = comp_loss(y, logits, targeted, loss_type=loss_type)

    n_queries = 1

    time_start = time.time()
    delta_curr = copy.deepcopy(delta_init)
    for i_iter in range(n_iters):
        delta_curr_back = copy.deepcopy(delta_curr)

        x_curr = x 
        y_curr = y

        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = torch.zeros(delta_curr.shape)
        new_deltas_mask[:, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = torch.zeros(delta_curr.shape)
        new_deltas_mask_2[:, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0

        ### compute total norm available
        curr_norms_window = torch.sqrt(
            torch.sum((delta_curr * new_deltas_mask) ** 2, axis=(1,2), keepdims=True))
        curr_norms_image = torch.sqrt(torch.sum(delta_curr ** 2, axis=(0,1, 2), keepdims=True))
        mask_2 = torch.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = torch.sqrt(torch.sum((delta_curr * mask_2) ** 2, axis=(1,2), keepdims=True))

        ### create the updates
        new_deltas = torch.ones([c, s, s]) 
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, s, s])
        new_deltas *= torch.Tensor(np.random.choice([-1, 1], size=[c, 1, 1]))
        old_deltas = delta_curr[ :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window) 
        new_deltas += old_deltas

        new_deltas = new_deltas / torch.sqrt(torch.sum(new_deltas ** 2, axis=(1,2), keepdims=True)) * (
            torch.max(eps ** 2 - curr_norms_image ** 2, torch.zeros_like(curr_norms_image)) / c + norms_windows ** 2) ** 0.5

        delta_curr[:, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        delta_curr = delta_curr/(torch.sum(delta_curr ** 2) ** 0.5) * eps
        x_new = projection_img(x_curr,delta_curr,normalization,denormalization,device=device)
        with torch.no_grad():
            logits = model(x_new)
        loss = comp_loss(y_curr, logits, targeted, loss_type=loss_type)

        curr_asr = (logits.argmax(1) == target_label).sum()/x_best.size(0)
        time_total = time.time() - time_start
        if loss.sum() < loss_min_curr.sum():
            if curr_asr > best_asr:
                best_asr = curr_asr
            logging.info('I_iter {:.0f}, Time {:.0f}s, {:.0f}, found better uap with loss update {:.4f}, asr={:.4f}, best asr={:.4f}'.format(i_iter,time_total,i_iter, loss.sum() - loss_min_curr.sum(),curr_asr,best_asr))
            loss_min_curr = loss
        else:
            delta_curr = delta_curr_back
        n_queries += 1
        if best_asr >0.99:
            break 

    return n_queries, delta_curr, best_asr, curr_asr


def square_attack_linf(model, x, target_label, y, eps, n_iters, p_init, targeted, loss_type,device,normalization,denormalization):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    c, h, w = x.shape[1:]
    n_features = c*h*w

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    delta_init = torch.Tensor(np.random.choice([-eps, eps], size=[c, 1, w])).to(device)
    x_best = projection_img(x,delta_init,normalization,denormalization,device)
    with torch.no_grad():
        logits = model(x_best)
    best_asr = (logits.argmax(1) == target_label).sum()/x_best.size(0)
    loss_min_curr = comp_loss(y, logits, targeted, loss_type=loss_type)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=50'
    n_queries = 1
    time_start = time.time()
    delta_curr = copy.deepcopy((x_best-x)[0])
    for i_iter in range(n_iters - 1):
        delta_curr_back = copy.deepcopy(delta_curr)
        # delta_curr_back = delta_curr.detach()
        x_curr,y_curr = x, y
        x_best = projection_img(x_curr,delta_curr,normalization,denormalization,device)
    
        p = p_selection(p_init, i_iter, n_iters)
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        x_curr_window = x_curr[1, :, center_h:center_h+s, center_w:center_w+s]
        x_best_window = (x_curr+delta_curr)[1, :, center_h:center_h+s, center_w:center_w+s]
        # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
        while torch.sum(torch.abs(x_curr_window + delta_curr[:, center_h:center_h+s, center_w:center_w+s] - x_best_window) < 10**-6) == c*s*s:
            delta_curr[:, center_h:center_h+s, center_w:center_w+s] = t(np.random.choice([-eps, eps], size=[c, 1, 1]))
        # logging.info('Found')
        x_new = projection_img(x_curr,delta_curr,normalization,denormalization,device=device)
        with torch.no_grad():
            logits = model(x_new)
        loss = comp_loss(y_curr, logits, targeted, loss_type=loss_type)
        curr_asr = (logits.argmax(1) == target_label).sum()/x_curr.size(0)
        time_total = time.time() - time_start
        if loss.sum() < loss_min_curr.sum():
            if curr_asr > best_asr:
                best_asr = curr_asr
            logging.info('I_iter {:.0f}, Time {:.0f}s, {:.0f}, found better uap with loss update {:.4f}, asr={:.4f}, best asr={:.4f}'.format(i_iter,time_total,i_iter, loss.sum() - loss_min_curr.sum(),curr_asr,best_asr))
            loss_min_curr = loss
        else:
            delta_curr = delta_curr_back
        
        n_queries += 1
        if best_asr >0.99:
            break 

    return n_queries, delta_curr, best_asr, curr_asr
