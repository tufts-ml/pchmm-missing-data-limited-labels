import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join('../utils')))
sys.path.append(os.path.abspath(os.path.join('../utils', 'MixMatch', 'MixMatch-pytorch')))
from dataset_loader import TidySequentialDataCSVLoader
from utils_preproc import parse_id_cols, parse_output_cols, parse_feature_cols, parse_id_cols, parse_time_cols, load_data_dict_json
from numpy.random import seed
import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, make_scorer)
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from simple_LSTM import SimpleLSTM
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

# import torch.utils as utils
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from utils import Bar, Logger, AverageMeter
import time
from sklearn.metrics import (roc_auc_score, average_precision_score)

parser = argparse.ArgumentParser(description='GRU-D fitting')
parser.add_argument('--outcome_col_name', type=str, required=True)
parser.add_argument('--train_csv_files', type=str, required=True)
parser.add_argument('--valid_csv_files', type=str, default=None)
parser.add_argument('--test_csv_files', type=str, required=True)
parser.add_argument('--data_dict_files', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of sequences per minibatch')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate for the optimizer')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--validation_size', type=float, default=0.15,
                    help='validation split size')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout for GRUD')
parser.add_argument('--output_dir', type=str, default=None, 
                    help='directory where trained model and loss curves over epochs are saved')
parser.add_argument('--output_filename_prefix', type=str, default=None, 
                    help='prefix for the training history jsons and trained classifier')
parser.add_argument('--l2_penalty', type=float, default=1e-4,
                    help='weight decay l2')
parser.add_argument('--train-iteration', type=int, default=4,
                        help='Number of iteration per epoch')



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

def data_dataloader(train_data, train_outcomes, valid_data, valid_outcomes, test_data, test_outcomes, batch_size=512):
    
    
    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_outcomes[:, np.newaxis])
    dev_data, dev_label = torch.Tensor(valid_data), torch.Tensor(valid_outcomes[:, np.newaxis])
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_outcomes[:, np.newaxis])
    
    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    # dataset to dataloader 
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = utils.DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size)
    
    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))
    
    return train_dataloader, dev_dataloader, test_dataloader
    
    
def main():
    x_train_csv_filename, y_train_csv_filename = args.train_csv_files.split(',')
    x_valid_csv_filename, y_valid_csv_filename = args.valid_csv_files.split(',')
    x_test_csv_filename, y_test_csv_filename = args.test_csv_files.split(',')
    x_dict, y_dict = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_dict)

    # get the id and feature columns
    id_cols = parse_id_cols(x_data_dict)
    feature_cols = parse_feature_cols(x_data_dict)
    # extract data
    train_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_train_csv_filename,
        y_csv_path=y_train_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_sequence'
    )

    test_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_test_csv_filename,
        y_csv_path=y_test_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_sequence'
    )
    

    valid_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_valid_csv_filename,
        y_csv_path=y_valid_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=args.outcome_col_name,
        y_label_type='per_sequence')
    
    
    train_x_NTD, train_y = train_vitals.get_batch_data(batch_id=0)
    valid_x_NTD, valid_y = valid_vitals.get_batch_data(batch_id=0)
    test_x_NTD, test_y = test_vitals.get_batch_data(batch_id=0)
    
    
    train_x_NTD[np.isnan(train_x_NTD)]=0
    valid_x_NTD[np.isnan(valid_x_NTD)]=0
    test_x_NTD[np.isnan(test_x_NTD)]=0
    
    # get them into the dataloader
    train_data, train_label = torch.Tensor(train_x_NTD), torch.Tensor(train_y[:, np.newaxis])
    train_labeled_set = data.TensorDataset(train_data, train_label)
    
    valid_data, valid_label = torch.Tensor(valid_x_NTD), torch.Tensor(valid_y[:, np.newaxis])
    valid_labeled_set = data.TensorDataset(valid_data, valid_label)    

    test_data, test_label = torch.Tensor(test_x_NTD), torch.Tensor(test_y[:, np.newaxis])
    test_labeled_set = data.TensorDataset(test_data, test_label)  
    
    
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    if len(labeled_trainloader)==0:
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=len(train_labeled_set), shuffle=True, num_workers=0, drop_last=True)
    
    full_train_loader = data.DataLoader(train_labeled_set, batch_size=len(train_labeled_set), shuffle=False, num_workers=0)
    val_loader = data.DataLoader(valid_labeled_set, batch_size=len(valid_labeled_set), shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_labeled_set, batch_size=len(test_labeled_set), shuffle=False, num_workers=0)
    
    
    N, T, D = train_x_NTD.shape
    model=SimpleLSTM(input_size=D, hidden_size=16, 
                output_size=2, num_layers=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('Total params: %d' % (sum(p.numel() for p in model.parameters())))
    
    start_epoch = 0
    early_stopping_epochs=5#10
    min_epochs_before_early_stopping=20#50
    step = 0
    perf_dict_list = []
    use_cuda=False
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = train_with_bce(labeled_trainloader, model, optimizer, criterion)
            
        train_loss, train_auprc, train_auroc = validate(full_train_loader, model, criterion, 
                                                        epoch, use_cuda, mode='Train Stats')
        valid_loss, valid_auprc, valid_auroc = validate(val_loader, model, criterion, 
                                                        epoch, use_cuda, mode='Valid Stats')
        test_loss, test_auprc, test_auroc = validate(test_loader, model, criterion, 
                                                     epoch, use_cuda, mode='Test Stats ')

        step = args.train_iteration * (epoch + 1)
        
        # early stopping
        if epoch>=early_stopping_epochs+min_epochs_before_early_stopping:
            prev_auprc = perf_dict_list[epoch-early_stopping_epochs]['valid_auprc']
            if valid_auprc<=prev_auprc:
                print('Validation AUPRC did not improve in last %d epochs.. Early stopping at epoch %d..'%(early_stopping_epochs, epoch))
                break
        
        
        # append logger file
        perf_dict = {'epoch': epoch,
                    'train_loss' : train_loss,
#                     'labelled_train_loss' : train_loss_x,
#                     'unlabelled_train_loss' : train_loss_u,
                    'valid_loss' : valid_loss,
                    'test_loss' : test_loss,
                    'train_auprc' : train_auprc,
                    'valid_auprc' : valid_auprc,
                    'test_auprc' : test_auprc,
                    'train_auroc' : train_auroc,
                    'valid_auroc' : valid_auroc,
                    'test_auroc' : test_auroc,
                    }
        print('-----------------------------Performance after epoch %s ------------------'%epoch)
        print(perf_dict)
        print('--------------------------------------------------------------------------')
        perf_dict_list.append(perf_dict)
        
    # save some additional performance 
    perf_save_file = os.path.join(args.output_dir, 'final_perf_'+args.output_filename_prefix+'.csv')
    model_perf_df = pd.DataFrame([{'train_AUC' : train_auroc, 
                                   'valid_AUC' : valid_auroc, 
                                   'test_AUC' : test_auroc}])
    
    
    model_perf_df.to_csv(perf_save_file, index=False)
    
def train_with_bce(labeled_trainloader, model, optimizer, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

#     bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
#     unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(len(labeled_trainloader)):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()
        
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 2).scatter_(1, targets_x.view(-1,1).long(), 1)
        outputs = model(inputs_x)
        loss = criterion(outputs, targets_x.long()[:, 1])
         

#         roc_auc = roc_auc_score(targets, outputs)
#         auprc = average_precision_score(targets[:, 1], outputs[:, 1])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    return (float(loss))


def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    valid_iter = iter(valloader)
    with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(valloader):
        for batch_idx in range(len(valloader)):
            inputs, targets = valid_iter.next()
            
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            
            batch_size = inputs.size(0)

            # Transform label to one-hot
            targets = torch.zeros(batch_size, 2).scatter_(1, targets.view(-1,1).long(), 1)
            loss = criterion(outputs, targets.long()[:, 1]) 
            
            roc_auc_valid = roc_auc_score(targets, outputs)
            auprc_valid = average_precision_score(targets[:, 1], outputs[:, 1])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
    return (float(loss), auprc_valid, roc_auc_valid)


if __name__ == '__main__':
    main()
