# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:34:08 2021

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
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    

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

    # optimizer, lr_scheduler = utils.make_optimizer(
    #         model.parameters(),
    #         config['optimizer'], **config['optimizer_args'])
    
    # remove the classifier layer (linear 512, 64), the last layer feature is 512
    model.classifier = Identity()
    # print(model)
    
    '''
    #### TEST Dataset ####
    fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
    
    utils.log('fs dataset: {} (x{}), {}'.format(
            fs_dataset[0][0].shape, len(fs_dataset),
            fs_dataset.n_classes))
    
    # load data for each label in training
    list_center = np.empty((512), int)
    for A in range(fs_dataset.n_classes):
        idx = [i for i, x in enumerate(fs_dataset.label) if x == A]
        
        # select n sample (n-shot) for each label
        n = config['n_shot']
        # idx = idx[0:n]
        idx = random.sample(idx, n)
        
        data_index = Subset(fs_dataset, idx)
        fs_loader = DataLoader(data_index, config['batch_size'], shuffle=False,
                                  num_workers=8, pin_memory=True)
        # fs_loader = DataLoader(data_index, 5, shuffle=False,
                                  # num_workers=8, pin_memory=True)
        utils.log('fs dataset: {} (x{})'.format(
                data_index[0][0].shape, len(data_index)))
        
        # feed the data to the trained model to obtain feature
        batch_id = 0
        model.eval()
        for data,_ in tqdm(fs_loader, desc='train', leave=False):
            data = data.cuda()
            with torch.no_grad():
                feature = model(data)
            if (batch_id == 0):
                features = feature.cpu().detach().numpy()
                batch_id = 1
            else:
                features = np.concatenate((features, feature.cpu().detach().numpy()), axis=0)
        
        print(features.shape)
        center = np.mean(features, axis=0)
        print(center.shape)
        
        
        if (A==0):
            list_center = center
        else:
            list_center = np.vstack((list_center, center)) 
        print(list_center.shape)
        
    print(list_center.shape)
    
    # save center points to file
    np.savetxt('test_center.csv', list_center, delimiter=',')
    
    '''
    #### TRAIN Dataset ####

    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    
    # load data for each label in training
    list_center = np.empty((512), int)
    for A in range(train_dataset.n_classes):
        idx = [i for i, x in enumerate(train_dataset.label) if x == A]
        data_index = Subset(train_dataset, idx)
        train_loader = DataLoader(data_index, config['batch_size'], shuffle=False,
                                  num_workers=8, pin_memory=True)
        
        utils.log('train dataset: {} (x{})'.format(
                data_index[0][0].shape, len(data_index)))
    
    
        # feed the data to the trained model to obtain feature
        batch_id = 0
        model.eval()
        for data,_ in tqdm(train_loader, desc='train', leave=False):
            data = data.cuda()
            with torch.no_grad():
                feature = model(data)
            if (batch_id == 0):
                features = feature.cpu().detach().numpy()
                batch_id = 1
            else:
                features = np.concatenate((features, feature.cpu().detach().numpy()), axis=0)
        
        print(features.shape)
        center = np.mean(features, axis=0)
        print(center.shape)
        
        
        if (A==0):
            list_center = center
        else:
            list_center = np.vstack((list_center, center)) 
        print(list_center.shape)
        
    print(list_center.shape)
    
    # save center points to file
    np.savetxt('train_center.csv', list_center, delimiter=',')
    
    # load center points
    # center_point = np.loadtxt('train_center.csv', delimiter=',')
    # print(center_point.shape)

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

    utils.set_gpu(args.gpu)
    main(config)
    
    