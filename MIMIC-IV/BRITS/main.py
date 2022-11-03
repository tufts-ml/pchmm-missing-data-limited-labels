import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
import sys
sys.path.append('models')
import models
import argparse
# import data_loader
import pandas as pd
# import ujson as json
import os
import torch.utils.data as data
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--train_np_files', type=str, required=True)
parser.add_argument('--test_np_files', type=str, required=True)
parser.add_argument('--valid_np_files', type=str, default=None)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=512)
parser.add_argument('--model', type=str, default='brits')
parser.add_argument('--hid_size', type=int, default=64)
parser.add_argument('--impute_weight', type=float, default=0.3)
parser.add_argument('--label_weight', type=float, default=1.0)
parser.add_argument('--perc_labelled', default='100')
parser.add_argument('--output_dir', default='result',
                        help='output directory')
parser.add_argument('--output_filename_prefix', default='result',
                        help='prefix of outcome file')


args = parser.parse_args()

def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


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


def train(model, data_iter, tr_loader, val_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     data_iter = data_loader.get_loader(batch_size=args.batch_size)
    
    perf_dict_list = []
    
    
    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
#             data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            print('Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

        tr_auc, tr_auprc = evaluate(model, tr_loader)
        val_auc, val_auprc = evaluate(model, val_loader)
        test_auc, test_auprc = evaluate(model, test_loader)
        print('========================================================================')
        print('Epoch : %d, \nAUC_train : %.3f, AUC_valid : %.3f, AUC_test : %.3f, \nAUPRC_train : %.3f, AUPRC_valid : %.3f, AUPRC_test : %.3f'%(epoch, tr_auc, val_auc, test_auc, tr_auprc, val_auprc, test_auprc))
        print('========================================================================')
        
        curr_perf_dict = {'loss': run_loss / (idx + 1.0),
                         'AUC_train' : tr_auc,
                         'AUC_valid' : val_auc,
                         'AUC_test' : test_auc,
                         'AUPRC_train' : tr_auprc,
                         'AUPRC_valid' : val_auprc,
                         'AUPRC_test' : test_auprc}
        perf_dict_list.append(curr_perf_dict)
        
    perf_df = pd.DataFrame(perf_dict_list)
    return perf_df
        
def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter):
#         data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        preds = ret['predictions'].data.cpu().numpy()
        labels = ret['labels'].data.cpu().numpy()
#         is_train = ret['is_train'].data.cpu().numpy()

#         eval_masks = ret['eval_masks'].data.cpu().numpy()
#         eval_ = ret['evals'].data.cpu().numpy()
        imputations = ret['imputations'].data.cpu().numpy()

#         evals += eval_[np.where(eval_masks == 1)].tolist()
#         imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
#         pred = pred[np.where(is_train == 0)]
#         label = label[np.where(is_train == 0)]

#         labels += label.tolist()
#         preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)
    auc_score = metrics.roc_auc_score(labels, preds)
    auprc_score = metrics.average_precision_score(labels, preds)
    
    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    
    return auc_score, auprc_score 

def run():
    
    np.random.seed(args.seed)
    
    x_train_np_filename, y_train_np_filename = args.train_np_files.split(',')
    x_test_np_filename, y_test_np_filename = args.test_np_files.split(',')
    x_valid_np_filename, y_valid_np_filename = args.valid_np_files.split(',')
    
#     data_save_dir = '/cluster/tufts/hugheslab/prath01/datasets/mimic4_ssl/percentage_labelled_sequnces=100/'
#     x_train_np_filename = os.path.join(data_save_dir, 'X_train.npy')
#     x_valid_np_filename = os.path.join(data_save_dir, 'X_valid.npy')
#     x_test_np_filename = os.path.join(data_save_dir, 'X_test.npy')
#     y_train_np_filename = os.path.join(data_save_dir, 'y_train.npy')
#     y_valid_np_filename = os.path.join(data_save_dir, 'y_valid.npy')
#     y_test_np_filename = os.path.join(data_save_dir, 'y_test.npy')  
    
    X_train = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_valid = np.load(x_valid_np_filename)
    y_valid = np.load(y_valid_np_filename)
    
    # get only the labelled data
    labelled_inds_tr = ~np.isnan(y_train)
    labelled_inds_va = ~np.isnan(y_valid)
    labelled_inds_te = ~np.isnan(y_test)
    
    X_train_labelled = X_train[labelled_inds_tr].copy()
    y_train_labelled = y_train[labelled_inds_tr]
    
    X_valid_labelled = X_valid[labelled_inds_va]
    y_valid_labelled = y_valid[labelled_inds_va]  
    
    
    X_test_labelled = X_test[labelled_inds_te]
    y_test_labelled = y_test[labelled_inds_te]    
    
    
    train_x_mask_NTD = 1-np.isnan(X_train_labelled).astype(int)
    valid_x_mask_NTD = 1-np.isnan(X_valid_labelled).astype(int)
    test_x_mask_NTD = 1-np.isnan(X_test_labelled).astype(int)    
    
    train_x_delta_NTD = 8*train_x_mask_NTD  
    train_x_delta_NTD[train_x_delta_NTD==0]=16
    
    
    valid_x_delta_NTD = 8*valid_x_mask_NTD
    valid_x_delta_NTD[valid_x_delta_NTD==0]=16
    
    
    test_x_delta_NTD = 8*test_x_mask_NTD
    test_x_delta_NTD[test_x_delta_NTD==0]=16
    
    
    _, T, D = X_test_labelled.shape
    for d in range(D):
        # zscore normalization
        den = 1#np.nanstd(X_train[:, :, d])
    
        X_train_labelled[:, :, d] = (X_train_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_valid_labelled[:, :, d] = (X_valid_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        X_test_labelled[:, :, d] = (X_test_labelled[:, :, d] - np.nanmean(X_train[:, :, d]))/den
        
        # impute missing values by mean filling using estimates from full training set
        fill_vals = 0
        X_train_labelled[:, :, d] = np.nan_to_num(X_train_labelled[:, :, d], nan=fill_vals)
        X_valid_labelled[:, :, d] = np.nan_to_num(X_valid_labelled[:, :, d], nan=fill_vals)
        X_test_labelled[:, :, d] = np.nan_to_num(X_test_labelled[:, :, d], nan=fill_vals)


    # Mask = 1 if present
    train_x_NTD = X_train_labelled.copy()
    valid_x_NTD = X_valid_labelled.copy()
    test_x_NTD = X_test_labelled.copy()
    train_y = y_train_labelled.copy()
    valid_y = y_valid_labelled.copy()
    test_y = y_test_labelled.copy()
    
    train_data, train_label = torch.Tensor(X_train_labelled), torch.Tensor(y_train_labelled[:, np.newaxis])
    train_mask, train_delta = torch.Tensor(train_x_mask_NTD), torch.Tensor(train_x_delta_NTD)
    train_labeled_set = data.TensorDataset(train_data, train_label, train_mask, train_delta)
    
    valid_data, valid_label = torch.Tensor(X_valid_labelled), torch.Tensor(y_valid_labelled[:, np.newaxis])
    valid_mask, valid_delta = torch.Tensor(valid_x_mask_NTD), torch.Tensor(valid_x_delta_NTD)
    valid_labeled_set = data.TensorDataset(valid_data, valid_label, valid_mask, valid_delta)    

    test_data, test_label = torch.Tensor(X_test_labelled), torch.Tensor(y_test_labelled[:, np.newaxis])
    test_mask, test_delta = torch.Tensor(test_x_mask_NTD), torch.Tensor(test_x_delta_NTD)
    test_labeled_set = data.TensorDataset(test_data, test_label, test_mask, test_delta) 
    
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    full_train_loader = data.DataLoader(train_labeled_set, batch_size=len(train_labeled_set), shuffle=False, num_workers=0)
    val_loader = data.DataLoader(valid_labeled_set, batch_size=len(valid_labeled_set), shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_labeled_set, batch_size=len(test_labeled_set), shuffle=False, num_workers=0)    
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    perf_df = train(model, labeled_trainloader, full_train_loader, val_loader, test_loader)
    final_perf_save_file = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    
    print('Save final performance file to : \n%s'%final_perf_save_file)
    perf_df.to_csv(final_perf_save_file, index=False)
    
    
if __name__ == '__main__':
    run()

