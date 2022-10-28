import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer)
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('../utils')))

from dataset_loader import TidySequentialDataCSVLoader
from utils_preproc import (parse_id_cols, parse_feature_cols, load_data_dict_json)
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import glob
from numpy.random import RandomState
from pcvae.datasets.toy import toy_line, custom_dataset
from pcvae.models.hmm import HMM
from pcvae.datasets.base import dataset, real_dataset, classification_dataset, make_dataset
from sklearn.model_selection import train_test_split
from pcvae.util.optimizers import get_optimizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy.ma as ma
from sklearn.cluster import KMeans

def get_sequence_lengths(X_NTF, pad_val):
    N = X_NTF.shape[0]
    seq_lens_N = np.zeros(N, dtype=np.int64)
    for n in range(N):
        bmask_T = np.all(X_NTF[n] == 0, axis=-1)
        seq_lens_N[n] = np.searchsorted(bmask_T, 1)
    return seq_lens_N

# def standardize_data_for_pchmm(X_train, y_train, X_test, y_test):
def convert_to_categorical(y):
    C = len(np.unique(y))
    N = len(y)
    y_cat = np.zeros((N, C))
    y_cat[:, 0] = (y==0)*1.0
    y_cat[:, 1] = (y==1)*1.0
    
    return y_cat

def save_loss_plots(model, save_file, data_dict):
    model_hist = model.history.history
    epochs = range(len(model_hist['loss']))
    hmm_model_loss = model_hist['hmm_model_loss']
    predictor_loss = model_hist['predictor_loss']
    model_hist['epochs'] = epochs
    model_hist['n_train'] = len(data_dict['train'][1])
    model_hist['n_valid'] = len(data_dict['valid'][1])
    model_hist['n_test'] = len(data_dict['test'][1])
    model_hist_df = pd.DataFrame(model_hist)
    
    # save to file
    model_hist_df.to_csv(save_file, index=False)
    print('Training plots saved to %s'%save_file)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pchmm fitting')
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
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')
    parser.add_argument('--lamb', type=int, default=1111,
                        help='langrange multiplier')
    parser.add_argument('--missing_handling', type=str, default="no_imp",
                        help='how to handle missing observations')
    parser.add_argument('--perc_missing', type=str, default=20,
                        help='percentage missing observations')
    
    args = parser.parse_args()

    rs = RandomState(args.seed)

    x_train_csv_filename, y_train_csv_filename = args.train_csv_files.split(',')
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

    X_train, y_train = train_vitals.get_batch_data(batch_id=0)
    X_test, y_test = test_vitals.get_batch_data(batch_id=0)
    _,T,F = X_train.shape
    
    print('number of time points : %s\nnumber of features : %s\n'%(T,F))
    
    
    if args.valid_csv_files is None:
        # split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_size, random_state=213)
    else:
        x_valid_csv_filename, y_valid_csv_filename = args.valid_csv_files.split(',')
        valid_vitals = TidySequentialDataCSVLoader(
            x_csv_path=x_valid_csv_filename,
            y_csv_path=y_valid_csv_filename,
            x_col_names=feature_cols,
            idx_col_names=id_cols,
            y_col_name=args.outcome_col_name,
            y_label_type='per_sequence'
        )
        X_val, y_val = valid_vitals.get_batch_data(batch_id=0)
    
    # get the init means and covariances of the observation distribution by clustering
    print('Initializing cluster means with kmeans...')
    prng = np.random.RandomState(args.seed)
    N = len(X_train)
    init_inds = prng.permutation(N)# select random samples from training set
    X_flat = X_train.reshape(N*T, X_train.shape[-1])# flatten across time
    X_flat = np.where(np.isnan(X_flat), ma.array(X_flat, mask=np.isnan(X_flat)).mean(axis=0), X_flat)
        
    # get cluster means
    n_states = 6
    init_means = np.array([15., 1.]) * prng.randn(6, 2) + np.array([10, -0.5]) 
    
    
    # get cluster covariance in each cluster
    init_covs = np.stack([np.array([1., 1.]) for i in range(n_states)])
    
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_train, y_train)
    data_dict['valid'] = (X_val, y_val)
    data_dict['test'] = (X_test, y_test)
    
    
    # train model
    spacing = 8
    optimizer = get_optimizer('adam', lr = args.lr)
    
    init_state_probas = 0.124*np.ones(n_states)

    model = HMM(
            states=n_states,                                     
            lam=args.lamb,                                      
            prior_weight=0.01,
            observation_dist='NormalWithMissing',
            observation_initializer=dict(loc=init_means, 
                scale=init_covs),
            initial_state_alpha=1.,
            initial_state_initializer=np.log(init_state_probas),
            transition_alpha=1.,
            transition_initializer=None, optimizer=optimizer)
    
    
    data = custom_dataset(data_dict=data_dict)
    
    if args.batch_size==-1:
        data.batch_size = data.batch_size=X_train.shape[0]
    else:
        data.batch_size = args.batch_size
    model.build(data) 
    
    # set the regression coefficients of the model
    eta_weights = np.zeros((n_states, 2))  
    
    # set the initial etas as the weights from logistic regression classifier with average beliefs as features 
    x_train,y_train = data.train().numpy()
    
    pos_inds = np.where(y_train[:,1]==1)[0]
    neg_inds = np.where(y_train[:,0]==1)[0]
        
    x_pos = x_train[pos_inds]
    y_pos = y_train[pos_inds]
    x_neg = x_train[neg_inds]
    y_neg = y_train[neg_inds]
    
    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)
    
    # print the average belief states across time
    beliefs_pos = z_pos.mean(axis=1)
    beliefs_neg = z_neg.mean(axis=1)
    
    
    # perform logistic regression with belief states as features for these positive and negative samples
    print('Performing Logistic Regression on initial belief states to get the initial eta coefficients...')
    logistic = LogisticRegression(solver='lbfgs', random_state = 42)
    penalty = ['l2']
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4, 1e5]
    hyperparameters = dict(C=C, penalty=penalty)
    classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = 'roc_auc')
    
    X_tr = np.vstack([beliefs_pos, beliefs_neg])
    y_tr = np.vstack([y_pos, y_neg])
    cv = classifier.fit(X_tr, y_tr[:,1])
    
    # get the logistic regression coefficients. These are the optimal eta coefficients.
    lr_weights = cv.best_estimator_.coef_
    
    # set the K logistic regression weights as K x 2 eta coefficients
    opt_eta_weights = np.vstack([np.zeros_like(lr_weights), lr_weights]).T
    
    # set the intercept of the eta coefficients
    opt_eta_intercept = np.asarray([0, cv.best_estimator_.intercept_[0]])
    init_etas = [opt_eta_weights, opt_eta_intercept]
    model._predictor.set_weights(init_etas)
    
    model.fit(data, steps_per_epoch=10, epochs=150, batch_size=args.batch_size, lr=args.lr,
              initial_weights=model.model.get_weights())
    
    
    # train just the predictor after training HMM if lambda=0
    if args.lamb==0:
        print('Training the predictor only since lambda=0')
        z_train = model.hmm_model.predict(x_train)
        z_train_mean = np.mean(z_train, axis=1)
        
        logistic = LogisticRegression(solver='lbfgs', random_state = 42)
        penalty = ['l2']
        C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4, 1e5]
        hyperparameters = dict(C=C, penalty=penalty)
        classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = 'average_precision')
        
        labeled_inds = ~np.isnan(y_train[:,1])
        cv = classifier.fit(z_train_mean[labeled_inds], y_train[labeled_inds,1])

        # get the logistic regression coefficients. These are the optimal eta coefficients.
        lr_weights = cv.best_estimator_.coef_

        # set the K logistic regression weights as K x 2 eta coefficients
        opt_eta_weights = np.vstack([np.zeros_like(lr_weights), lr_weights]).T

        # set the intercept of the eta coefficients
        opt_eta_intercept = np.asarray([0, cv.best_estimator_.intercept_[0]])
        final_etas = [opt_eta_weights, opt_eta_intercept]
        model._predictor.set_weights(final_etas)
    
    
    z_train = model.hmm_model.predict(x_train)
    y_train_pred_proba = model._predictor.predict(z_train)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    print('ROC AUC on train : %.5f'%train_roc_auc)
    
    # evaluate on test set
    x_valid, y_valid = data.valid().numpy()
    z_valid = model.hmm_model.predict(x_valid)
    y_valid_pred_proba = model._predictor.predict(z_valid)
    valid_roc_auc = roc_auc_score(y_valid, y_valid_pred_proba)
    print('ROC AUC on valid : %.5f'%valid_roc_auc)    
    
    # evaluate on test set
    x_test, y_test = data.test().numpy()
    z_test = model.hmm_model.predict(x_test)
    y_test_pred_proba = model._predictor.predict(z_test)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    print('ROC AUC on test : %.5f'%test_roc_auc)
    
    # get the parameters of the fit distribution
#     x,y = data.train().numpy()
    mu_all = model.hmm_model(x_train).observation_distribution.distribution.mean().numpy()
    try:
        cov_all = model.hmm_model(x_train).observation_distribution.distribution.covariance().numpy()
    except:
        cov_all = model.hmm_model(x_train).observation_distribution.distribution.scale.numpy()
        
    eta_all = np.vstack(model._predictor.get_weights())
    
    mu_params_file = os.path.join(args.output_dir, args.output_filename_prefix+'-fit-mu.npy')
    cov_params_file = os.path.join(args.output_dir, args.output_filename_prefix+'-fit-cov.npy')
    eta_params_file = os.path.join(args.output_dir, args.output_filename_prefix+'-fit-eta.npy')
    np.save(mu_params_file, mu_all)
    np.save(cov_params_file, cov_all)
    np.save(eta_params_file, eta_all)
    
    # save the loss plots of the model
    save_file = os.path.join(args.output_dir, args.output_filename_prefix+'.csv')
    save_loss_plots(model, save_file, data_dict)
    
    # save the model
    model_filename = os.path.join(args.output_dir, args.output_filename_prefix+'-weights.h5')
    model.model.save_weights(model_filename, save_format='h5')
    
    # save some additional performance 
    perf_save_file = os.path.join(args.output_dir, 'final_perf_'+args.output_filename_prefix+'.csv')
    model_perf_df = pd.DataFrame([{'train_AUC' : train_roc_auc, 
                                   'valid_AUC' : valid_roc_auc, 
                                   'test_AUC' : test_roc_auc}])
    
    
    model_perf_df.to_csv(perf_save_file, index=False)
        
 