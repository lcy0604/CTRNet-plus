# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import sys
import cv2
import math
import json
import torch
import numpy as np
import util.misc as utils

from typing import Iterable
from tqdm import tqdm
from util.visualize import tensor_to_cv2image
from skimage import io

def save_img(path, img):
    # img (H,W,C) or (H,W) np.uint8
    io.imsave(path+'/'+name+'.png', img)

def train_one_epoch(model: torch.nn.Module, discriminator: torch.nn.Module,
                    criterion: torch.nn.Module, data_loader: Iterable, optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    lr_scheduler: list = [0], print_freq: int = 10, debug: bool = False,
                    optim_with_mask: bool = False):
    model.train()
    criterion.train()

    # criterion_adversial.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 10
    optimizer['D'].param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer['G'].param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer['G'].param_groups[1]['lr'] = lr_scheduler[epoch] * 0.1

    if debug:
        count = 0
        save_folder = '../debug/20220112-vis-scutens-train-textmask'
        os.makedirs(save_folder, exist_ok=True)
        for data in tqdm(data_loader):
            for image, label, mask, mask_gt in zip(data['image'], data['label'], data['mask'], data['mask_gt']):
                image = tensor_to_cv2image(image, False)
                label = tensor_to_cv2image(label, False)
                mask = tensor_to_cv2image(mask, False)
                mask_gt = tensor_to_cv2image(mask_gt, False)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-image.jpg'), image)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-label.jpg'), label)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-mask.jpg'), mask)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-mask_gt.jpg'), mask_gt)
                count += 1
        return

    for data in metric_logger.log_every(data_loader, print_freq, header):
        
        images = data['image'].to(device); labels = data['label'].to(device); mask_gts = data['mask_gt'].to(device)

        structure_im = data['structure_im'].to(device)
        structure_lbl = data['structure_lbl'].to(device)

        soft_mask = data['soft_mask'].to(device)
         

        # soft_mask = 1 - soft_mask   ### for syn
        # mask_gts = 1 - mask_gts

        outputs = model(images, mask_gts, labels, structure_im, structure_lbl, soft_mask)

        real_prob = discriminator(labels, mask_gts)

        fake_prob_D = discriminator(outputs['output'][-1].contiguous().detach(), mask_gts)

        D_loss = criterion.discriminator_loss(real_prob, fake_prob_D)
        optimizer['D'].zero_grad()
        D_loss.backward()
        optimizer['D'].step()

        fake_prob_G = discriminator(outputs['output'][-1], mask_gts)

        outputs['real_prob'] = real_prob
        outputs['fake_prob_D'] = fake_prob_D
        outputs['fake_prob_G'] = fake_prob_G

        loss_dict = criterion(outputs, mask_gts, labels, structure_lbl)
        loss_dict['D_loss'] = D_loss

        weight_dict = {'MSR_loss': 1.5, 'prc_loss': 0.1, 'style_loss': 120, 'D_fake': 1, 'D_loss': 1, 'FM_loss': 1, 'structure_loss': 1}   

        for k in loss_dict.keys():
            loss_dict[k] *= weight_dict[k]

        G_loss = sum([loss_dict[k] for k in loss_dict.keys() if k != 'D_loss'])
        optimizer['G'].zero_grad()
        G_loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer['G'].step()
    # optimizer['D'].step()

    # D_loss = loss_dict['D_loss']
    # G_loss = sum([loss_dict[k] for k in loss_dict.keys() if k != 'D_loss'])
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()} 
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
    
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer['G'].param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir, chars, start_index, visualize=False):
    model.eval()
    criterion.eval()
    chars = list(chars)

    for data in tqdm(data_loader):
        images = data['image'].to(device)
        mask_gts = data['mask_gt'].to(device)
        labels = data['label'].to(device)

        soft_mask = data['soft_mask'].to(device)
        # import pdb;pdb.set_trace()
        soft_mask =  torch.mean(soft_mask, 1, keepdim=True)
        mask_gts = torch.mean(mask_gts, 1, keepdim=True)

        structure_im = data['structure_im'].to(device)
        structure_lbl = data['structure_lbl'].to(device)

        # outputs = model(images, mask_gts, None)
        # soft_mask = mask_gts
        # import pdb;pdb.set_trace()
        outputs = model(images, mask_gts, labels, structure_im, structure_lbl, soft_mask)
        # import pdb;pdb.set_trace()
        str_output = outputs[-1]
        str_output = str_output.cpu().clamp(min=0, max=1)
        
        output = outputs * (1 - soft_mask) + images * soft_mask
        output = output[-1].cpu().clamp(min=0, max=1)

        output = tensor_to_cv2image(output,False)

        str_output = tensor_to_cv2image(str_output,False)

        stroke_gt = mask_gts.cpu()
        stroke_mask = torch.cat((stroke_gt[0], stroke_gt[0], stroke_gt[0]), 0)
        mask = tensor_to_cv2image(stroke_mask,False)

        image_path = data['image_path'][0]
        if 'scut-syn' in image_path.lower():
            dataset_name_str = 'SCUT-Syn'
            dataset_name_paste = 'SCUT_Syn_com'
        elif 'scut-ens' in image_path.lower():
            dataset_name_str = 'SCUT-ENS'
            dataset_name_paste = 'SCUT_ENS_com'
        save_folder = os.path.join(output_dir, dataset_name_str)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, os.path.basename(image_path).replace('jpg', 'png'))
        # import pdb;pdb.set_trace()
        cv2.imwrite(save_path, str_output)

        save_folder_com = os.path.join(output_dir, dataset_name_paste)
        os.makedirs(save_folder_com, exist_ok=True)
        save_path_com = os.path.join(save_folder_com, os.path.basename(image_path).replace('jpg', 'png'))
        cv2.imwrite(save_path_com, output)

        # cv2.imwrite(os.path.join(save_folder, 'mask_' + os.path.basename(image_path).replace('jpg', 'png')), mask) 
        # save_img(save_path, output)

