# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:05:07 2021

@author: catpl
"""

from prototypical_batch_sampler import PrototypicalBatchSampler
from args import get_parser
from mini_imagenet_FS import MiniImagenetDataset

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os

device = 'cuda'
opt = get_parser().parse_args()

'''
Initialize the datasets, samplers and dataloaders
'''
if opt.dataset == 'mini_imagenet':
    train_dataset = MiniImagenetDataset(mode='train')
    val_dataset = MiniImagenetDataset(mode='val')
    test_dataset = MiniImagenetDataset(mode='test')
    

'''
tr_train_dataset, val_train_dataset = torch.utils.data.random_split(train_dataset, [30720, 7680])

tr_sampler = PrototypicalBatchSampler(labels=tr_train_dataset.dataset.y,
                                      classes_per_it=opt.classes_per_it_tr,
                                      num_samples=opt.num_support_tr + opt.num_query_tr,
                                      iterations=opt.iterations)

trainval_sampler = PrototypicalBatchSampler(labels=val_train_dataset.dataset.y,
                                      classes_per_it=opt.classes_per_it_tr,
                                      num_samples=opt.num_support_tr + opt.num_query_tr,
                                      iterations=opt.iterations)

val_sampler = PrototypicalBatchSampler(labels=val_dataset.y,
                                        classes_per_it=opt.classes_per_it_val,
                                        num_samples=opt.num_support_val + opt.num_query_val,
                                        iterations=opt.iterations)

test_sampler = PrototypicalBatchSampler(labels=test_dataset.y,
                                        classes_per_it=opt.classes_per_it_val,
                                        num_samples=opt.num_support_val + opt.num_query_val,
                                        iterations=opt.iterations)


tr_dataloader = torch.utils.data.DataLoader(tr_train_dataset.dataset, batch_size=128, shuffle=True)
                                            # batch_sampler=tr_sampler)

tr_val_dataloader = torch.utils.data.DataLoader(val_train_dataset.dataset, batch_size=240, shuffle=True)
                                                    # batch_sampler=trainval_sampler)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_sampler=val_sampler)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_sampler=test_sampler)

'''