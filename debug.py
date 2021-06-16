# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:23:24 2021

@author: catpl
"""

from prototypical_batch_sampler import PrototypicalBatchSampler
from args import get_parser
from mini_imagenet import MiniImagenetDataset

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os

opt = get_parser().parse_args()
train_dataset = MiniImagenetDataset(mode='train')
val_dataset = MiniImagenetDataset(mode='val')
test_dataset = MiniImagenetDataset(mode='test')

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

tr_dataloader = torch.utils.data.DataLoader(tr_train_dataset.dataset,
                                            batch_sampler=tr_sampler)

trainval_dataloader = torch.utils.data.DataLoader(val_train_dataset.dataset,
                                                    batch_sampler=trainval_sampler)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_sampler=val_sampler)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_sampler=test_sampler)


device = 'cuda'
for batch_idx, (inputs, targets) in enumerate(tr_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        print(targets)