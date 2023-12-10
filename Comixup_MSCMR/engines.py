import math
import sys
import random
import time
import datetime
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from torch.autograd import Variable
from utils import to_one_hot, distance
from mixup import mixup_process
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from match import mix_input
from math import ceil
import torch.nn.functional as Func
from cutout import Cutout, rotate_back, rotate_invariant
from inference import keep_largest_connected_components


class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, originals, inputs, outputs, ori_labels, labels, epoch, writer):
        self.save_image(originals, 'inputs_original', epoch, writer)
        self.save_image(inputs, 'inputs_train', epoch, writer)
        self.save_image(outputs.float(), 'outputs_train', epoch, writer)
        self.save_image(ori_labels.float(), 'labels_original', epoch, writer)
        self.save_image(labels.float(), 'labels_train', epoch, writer)


def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, writer):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = {k: len(v) for k, v in dataloader_dict.items()}
    iterats = {k: iter(v) for k, v in dataloader_dict.items()}
    tasks = dataloader_dict.keys()
    counts = {k: 0 for k in tasks}
    total_steps = sum(numbers.values())
    start_time = time.time()
    original_list, sample_list, output_list, target_list, target_ori_list = [], [], [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [t for t in tasks if counts[t] < numbers[t]]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task: counts[task] + 1})
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
        targets_onehot = convert_targets(targets, device)
        samples_var = Variable(samples.tensors, requires_grad=True)

        # Co-mixup
        A_dist = None

        # calculate saliency (unary)
        model.train()
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        losses.backward(retain_graph=True)

        unary = torch.sqrt(torch.mean(samples_var.grad ** 2, dim=1))
        unary = F.pad(unary, (22, 22, 22, 22, 0, 0), 'constant')

        # calculate compatibility
        with torch.no_grad():
            z = F.avg_pool2d(unary, kernel_size=8, stride=1)
            z_reshape = z.reshape(args.batch_size, -1)
            z_idx_1d = torch.argmax(z_reshape, dim=1)
            z_idx_2d = torch.zeros((args.batch_size, 2), device=z.device)
            z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
            z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
            A_dist = distance(z_idx_2d, dist_type='l1')

        # pad to 256 * 256
        samples_var_256 = F.pad(samples_var, (22, 22, 22, 22, 0, 0, 0, 0), 'constant')
        targets_onehot_256 = F.pad(targets_onehot, (22, 22, 22, 22, 0, 0, 0, 0), 'constant')

        out, reweighted_target, mask_list = mixup_process(samples_var_256,
                                                          targets_onehot_256,
                                                          args=args,
                                                          sc=unary,
                                                          A_dist=A_dist)
        out = out[:, :, 22:-22, 22:-22]
        reweighted_target = reweighted_target[:, :, 22:-22, 22:-22]
        ###

        # cutout_loss
        outputs_mixed = model(out, task)
        loss_dict_mixed = criterion(outputs_mixed, reweighted_target)
        losses_mixed = sum(
            loss_dict_mixed[k] * weight_dict[k] for k in loss_dict_mixed.keys() if k in ['loss_CrossEntropy'])
        if step == 0:
            print("mixed loss:", losses_mixed.item())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_mixed)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                    k in ['loss_CrossEntropy']}
        optimizer.zero_grad()

        losses_final = losses_mixed
        losses_final.backward()

        optimizer.step()
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, epoch, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    print_freq = 10
    numbers = {k: len(v) for k, v in dataloader_dict.items()}
    iterats = {k: iter(v) for k, v in dataloader_dict.items()}
    tasks = dataloader_dict.keys()
    counts = {k: 0 for k in tasks}
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [t for t in tasks if counts[t] < numbers[t]]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task: counts[task] + 1})
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot = convert_targets(targets, device)

        # comixup
        samples_var = Variable(samples.tensors, requires_grad=True)
        ###
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        # metric_logger.update(loss_multiDice=loss_dict_reduced['loss_multiDice'])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:
            ##
            ##mixup
            sample_list.append(samples_var[0])
            ##

            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)

            ##mixup
            target_list.append(targets_onehot.argmax(1, keepdim=True)[0])
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)

    return stats


def infer(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    print_freq = 10
    numbers = {k: len(v) for k, v in dataloader_dict.items()}
    iterats = {k: iter(v) for k, v in dataloader_dict.items()}
    tasks = dataloader_dict.keys()
    counts = {k: 0 for k in tasks}
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    # loss_recorded = []
    for step in range(total_steps):
        start = time.time()
        tasks = [t for t in tasks if counts[t] < numbers[t]]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task: counts[task] + 1})
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        outputs = model(samples, task)
        loss_dict = criterion(outputs, targets_mixed)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled)
        print(loss_dict_reduced.keys())
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:
            sample_list.append(samples.tensors[0])
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            target_list.append(targets[0]['masks'])
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    epoch = 0
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)

    return stats
