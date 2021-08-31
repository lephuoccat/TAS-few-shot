# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:49:19 2021

@author: catpl
"""

import argparse
import os
import yaml

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from copy import deepcopy


def fisher_distance(config, origin_label, fix_label):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    # utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    
    #### TRAIN Dataset ####

    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    
    # index of train data with original selected labels
    train_idx = []
    for A in origin_label:
        idx = [i for i, x in enumerate(train_dataset.label) if x == A]
        train_idx.extend(idx)
    
    # update the label to matched_label
    train_dataset.label = torch.tensor(train_dataset.label)
    for i in range(len(origin_label)):
        train_dataset.label[train_dataset.label == origin_label[i]] = - fix_label[i]
        
    train_dataset.label[train_idx] = - train_dataset.label[train_idx]
    train_dataset.label = train_dataset.label.tolist()
        
    train_data_index = Subset(train_dataset, train_idx)
    train_loader = DataLoader(train_data_index, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    
    utils.log('train dataset: {} (x{})'.format(
            train_data_index[0][0].shape, len(train_data_index)))
    
    
    
    #### TEST Dataset ####
    
    fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
    
    utils.log('fs dataset: {} (x{}), {}'.format(
            fs_dataset[0][0].shape, len(fs_dataset),
            fs_dataset.n_classes))
    
    # load n datapoints for each label in testing
    test_idx = []
    for A in range(fs_dataset.n_classes):
        idx = [i for i, x in enumerate(fs_dataset.label) if x == A]
        
        # select n sample (n-shot) for each label
        n = config['n_shot']
        idx = random.sample(idx, n)
        test_idx.extend(idx)
    
    
    fs_data_index = Subset(fs_dataset, test_idx)
    fs_loader = DataLoader(fs_data_index, len(test_idx), shuffle=True,
                               num_workers=8, pin_memory=True)
    utils.log('fs dataset: {} (x{})'.format(
            fs_data_index[0][0].shape, len(fs_data_index)))
        
    


    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
    
    # replace the classifier layer (linear 512, 64) with (linear 512, 20)
    model.classifier = nn.Linear(512, 20)
    # print(model)
    model.cuda()
    
    ########
    
    max_epoch = 9

    for epoch in range(1, max_epoch + 1 + 1):
        print('Epoch: {}'.format(epoch))
        
        # if epoch == max_epoch + 1:
        #     if not config.get('epoch_ex'):
        #         break
        #     train_dataset.transform = train_dataset.default_transform
        #     train_data_index = Subset(train_dataset, train_idx)
        #     train_loader = DataLoader(train_data_index, config['batch_size'], shuffle=True,
        #                       num_workers=8, pin_memory=True)
            
        #     utils.log('train dataset: {} (x{})'.format(
        #     train_data_index[0][0].shape, len(train_data_index)))
            

        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}
        
        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            data = data.cuda()
            label = label.cuda()
            model.cuda()
            
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None
            
        
        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)
            
        # save model
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        if epoch == max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'task_epoch-last.pth'))

        else:
            torch.save(save_obj, os.path.join(save_path, 'task_epoch-ex.pth'))

        writer.flush()
            
    
    # calculate Fisher matrices for TRAIN
    fisher_matrix_source = diag_fisher(model, train_loader)
    
    total_source = 0
    for n, p in model.named_parameters():
        total_source += np.sum(fisher_matrix_source[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in model.named_parameters():
        fisher_matrix_source[n] = fisher_matrix_source[n]/total_source
    
    # calculate fisher matrix for TEST
    fisher_matrix_target = diag_fisher(model, fs_loader)
    
    total_target = 0
    for n, p in model.named_parameters():
        total_target += np.sum(fisher_matrix_target[n].cpu().numpy())
    
    # normalize the entire network
    for n, p in model.named_parameters():
        fisher_matrix_target[n] = fisher_matrix_target[n]/total_target

    
    # Frechet distance
    distance = 0
    for n, p in model.named_parameters():
        distance += 0.5 * np.sum(((fisher_matrix_source[n]**0.5 - fisher_matrix_target[n]**0.5)**2).cpu().numpy())
        
    return distance
    


def matching_feature(label_list):
    N = len(label_list)
    # load N selected center points for train
    train_center_point = np.loadtxt('train_center.csv', delimiter=',')
    selected_center_point = train_center_point[label_list]
    
    # load N center points for test
    test_center_point = np.loadtxt('test_center_full.csv', delimiter=',')
    
    # weight matrix of 20 test to 20 selected train points
    weight_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            weight_mat[i][j] = np.linalg.norm(selected_center_point[i] - test_center_point[j])
            
    # find the min weight matching
    biadjacency_matrix = csr_matrix(weight_mat)
    
    return min_weight_full_bipartite_matching(biadjacency_matrix)[1]


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def diag_fisher(model, data):
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data)

    model.eval()
    error = nn.CrossEntropyLoss()
    for inputs, labels in data:
        inputs, labels = inputs.cuda(), labels.cuda()
        model.zero_grad()
        output = model(inputs)
        # print(output.shape)

        loss = error(output, labels)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    
    return precision_matrices



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config = yaml.load(open('configs/center_feature.yaml'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    # utils.set_gpu(args.gpu)
    
    N = 20  # number of label in test
    d1 = 1
    d2 = 1
    L1 = random.sample(range(0, 64), N)
    L2 = random.sample(range(0, 64), N)
    
    for i in range(100):
        # randomly select 20 labels from 64 train labels
        select_label = random.sample(range(0, 64), N)
        
        # obtain the matching from 20 selected labels to 20 test labels (0,1,..19)
        matched_label = matching_feature(select_label)
        
        # Fisher distance
        d = fisher_distance(config, select_label, matched_label)
        print(d)
        print(select_label)
        
        if (d < d1):
            d2 = d1
            L2 = L1
            
            d1 = d
            L1 = select_label
            
        elif (d < d2):
            d2 = d
            L2 = select_label
    
        print('1st distance: {}'.format(d1))
        print(L1)
        print('2nd distance: {}'.format(d2))
        print(L2)
    
    L1 = np.array(L1)
    L2 = np.array(L2)
    top2 = np.vstack((L1, L2))
    
    # save top-2 labels
    np.savetxt('top2.csv', top2, delimiter=',')
    
    # load labels
    # top2_label = np.loadtxt('top2.csv', delimiter=',')
    # print(top2_label.shape)
    
    
    