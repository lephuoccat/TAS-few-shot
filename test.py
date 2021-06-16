# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:01:52 2021

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

device = 'cuda'

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    if opt.dataset == 'mini_imagenet':
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


    tr_dataloader = torch.utils.data.DataLoader(tr_train_dataset.dataset, batch_size=128, shuffle=True)
                                                # batch_sampler=tr_sampler)

    tr_val_dataloader = torch.utils.data.DataLoader(val_train_dataset.dataset, batch_size=240, shuffle=True)
                                                        # batch_sampler=trainval_sampler)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_sampler=val_sampler)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_sampler=test_sampler)
    
    return tr_dataloader, tr_val_dataloader, val_dataloader, test_dataloader



def init_protonet(opt):
    '''
    Initialize the network
    '''
    torch.manual_seed(0)
    if opt.dataset == 'mini_imagenet':
        model = models.mobilenet_v2()
    model = model.cuda() if opt.cuda else model
    torch.manual_seed(0)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                      nn.Linear(1280, 64))
    # print(model)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, tr_val_dataloader):
    '''
    Train the model with the learning algorithm
    '''

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    # best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    # last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(1):
        print('=== Epoch: {} ==='.format(epoch))
        # tr_iter = iter(tr_dataloader)
        model.train()
        correct = 0
        val_correct = 0
        
        # tr_len = 0
        # for batch_idx, (inputs, targets) in enumerate(tr_dataloader):
        #     inputs = inputs.to(device)
        #     targets = targets.to(device)
            
        #     optimizer.zero_grad()
        #     output = model(inputs)
        #     loss = error(output, targets)
        #     loss.backward()
        #     optimizer.step()
            
        #     # Total correct predictions
        #     predicted = torch.max(output.data, 1)[1] 
        #     correct += (predicted == targets).sum()
        #     tr_len += len(targets)
        #     if batch_idx % 50 == 0:
        #         print('Epoch : {} ({:.0f}%) \t\t Accuracy:{:.3f}%'.format(
        #             epoch, 100.*batch_idx / len(tr_dataloader), float(correct * 100) / tr_len))
            
        
        val_len = 0
        for batch_idx, (inputs, targets) in enumerate(tr_val_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            output = model(inputs)
            predicted = torch.max(output,1)[1]
            val_correct += (predicted == targets).sum()
            val_len += len(targets)
        print("Test accuracy:{:.3f}% \n".format( float(val_correct * 100) / val_len))
            
        
def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, trainval_dataloader, val_dataloader, test_dataloader = init_dataset(options)
    model = init_protonet(options).to(device)
    checkpoint = torch.load('./checkpoint/pretrain.t2')
    model.load_state_dict(checkpoint['net'])
    print(model)
    
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    
    train(opt=options,
            tr_dataloader=tr_dataloader,
            tr_val_dataloader=trainval_dataloader,
            model=model,
            optim=optim,
            lr_scheduler=lr_scheduler)
    
if __name__ == '__main__':
    main()