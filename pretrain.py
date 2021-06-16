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
import sys
import logging

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

    tr_val_dataloader = torch.utils.data.DataLoader(val_train_dataset.dataset, batch_size=64, shuffle=False)
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
    print(model)
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


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader):
    '''
    Train the model with the learning algorithm
    '''
    a_logger = logging.getLogger()
    a_logger.setLevel(logging.DEBUG)
    
    output_file_handler = logging.FileHandler("output.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    
    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)
    
    
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
    
    # train the network
    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        # tr_iter = iter(tr_dataloader)
        model.train()
        correct = 0
        val_correct = 0
        
        tr_len = 0
        for batch_idx, (inputs, targets) in enumerate(tr_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == targets).sum()
            tr_len += len(targets)
            if batch_idx % 50 == 0:
                a_logger.debug('Epoch : {} ({:.0f}%) \t\t Accuracy:{:.3f}%'.format(
                    epoch, 100.*batch_idx / len(tr_dataloader), float(correct * 100) / tr_len))
            
    # test the trained network on validation set
    val_len = 0
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        output = model(inputs)
        predicted = torch.max(output,1)[1]
        val_correct += (predicted == targets).sum()
        val_len += len(targets)
    a_logger.debug("Test accuracy:{:.3f}% \n".format( float(val_correct * 100) / val_len))
            
            
    # Save the pretrained network
    a_logger.debug('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': val_correct,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/pretrain.t2')
            
            
    #     for batch in tqdm(tr_iter):
    #         optim.zero_grad()
    #         x, y = batch
    #         x, y = Variable(x), Variable(y)
    #         if opt.cuda:
    #             x, y = x.cuda(), y.cuda()
    #         model_output = model(x)
    #         l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
    #         l.backward()
    #         optim.step()
    #         train_loss.append(l.data[0])
    #         train_acc.append(acc.data[0])
    #     avg_loss = np.mean(train_loss[-opt.iterations:])
    #     avg_acc = np.mean(train_acc[-opt.iterations:])
    #     print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
    #     lr_scheduler.step()
    #     if val_dataloader is None:
    #         continue
    #     val_iter = iter(val_dataloader)
    #     model.eval()
    #     for batch in val_iter:
    #         x, y = batch
    #         x, y = Variable(x), Variable(y)
    #         if opt.cuda:
    #             x, y = x.cuda(), y.cuda()
    #         model_output = model(x)
    #         l, acc = loss(model_output, target=y, n_support=opt.num_support_val) 
    #         val_loss.append(l.data[0])
    #         val_acc.append(acc.data[0])
    #     avg_loss = np.mean(val_loss[-opt.iterations:])
    #     avg_acc = np.mean(val_acc[-opt.iterations:])
    #     postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
    #         best_acc)
    #     print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
    #         avg_loss, avg_acc, postfix))
    #     if avg_acc >= best_acc:
    #         torch.save(model.state_dict(), best_model_path)
    #         best_acc = avg_acc
    #         best_state = model.state_dict()

    # torch.save(model.state_dict(), last_model_path)

    # for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
    #     save_list_to_file(os.path.join(opt.experiment_root, name + '.txt'), locals()[name])
        
    
    # optimizer = torch.optim.Adam(model.parameters())
    # error = nn.CrossEntropyLoss()
    # EPOCHS = args.num_epoch
    # model.train()
    # for epoch in range(EPOCHS):
    #     correct = 0
    #     for batch_idx, (inputs, targets) in enumerate(train_loader):
    #         inputs = inputs.to(device)
    #         targets = targets.long().to(device)
            
    #         optimizer.zero_grad()
    #         output = model(inputs)
    #         loss = error(output, targets)
    #         loss.backward()
    #         optimizer.step()
            
    #         # Total correct predictions
    #         predicted = torch.max(output.data, 1)[1] 
    #         correct += (predicted == targets).sum()
    #         #print(correct)
    #         if batch_idx % 50 == 0:
    #             print('Epoch : {} ({:.0f}%) \t\t Accuracy:{:.3f}%'.format(
    #                 epoch, 100.*batch_idx / len(train_loader), float(correct*100) / float(args.batch_size_train*(batch_idx+1))))

    
    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            avg_acc.append(acc.data[0])
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


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
    tr_dataloader, trainval_dataloader, val_dataloader, test_dataloader = init_dataset(
        options)
    model = init_protonet(options).to(device)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=trainval_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()