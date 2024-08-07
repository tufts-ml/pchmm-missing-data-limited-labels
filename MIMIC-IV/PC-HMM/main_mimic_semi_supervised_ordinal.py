import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer, average_precision_score)
from yattag import Doc
import matplotlib.pyplot as plt

PROJECT_REPO_DIR = os.path.abspath('../utils')
sys.path.append(PROJECT_REPO_DIR)
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'pcvae'))

# sys.path.append('pcvae/util/')
# from dataset_loader import TidySequentialDataCSVLoader
# from feature_transformation import (parse_id_cols, parse_feature_cols)
# from utils import load_data_dict_json
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import glob
from numpy.random import RandomState
from pcvae.datasets.toy import toy_line, custom_dataset, custom_regression_dataset
from pcvae.models.hmm import HMM
from pcvae.datasets.base import dataset, real_dataset, classification_dataset, make_dataset
from sklearn.model_selection import train_test_split
from pcvae.util.optimizers import get_optimizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy.ma as ma
from sklearn.cluster import KMeans
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

class OrdinalLoss(tf.keras.losses.Loss):
    """Ordinal Loss class"""
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ordinal_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)


def compute_binary_classification_perf(y, y_pred_proba_NC):
    # Get the predicted probabilities of los>=3 days
    y_pred_proba_N3 = np.sum(y_pred_proba_NC[:, 1:], axis=1)
    
    # get the true labels for los>=3 days
    y_N3 = (y>=1)*1
    
    # Get the predicted probabilities of los>=7 days
    y_pred_proba_N7 = np.sum(y_pred_proba_NC[:, 2:], axis=1)
    
    # get the true labels for los>=7 days
    y_N7 = (y>=2)*1
    
    # Get the predicted probabilities of los>=11 days
    y_pred_proba_N11 = y_pred_proba_NC[:, 3]
    
    # get the true labels for los>=7 days
    y_N11 = (y==3)*1
    
    labelled_inds = ~np.isnan(y)
    auprc_3 = average_precision_score(y_N3[labelled_inds], y_pred_proba_N3[labelled_inds])
    roc_auc_3 = roc_auc_score(y_N3[labelled_inds], y_pred_proba_N3[labelled_inds])
    
    auprc_7 = average_precision_score(y_N7[labelled_inds], y_pred_proba_N7[labelled_inds])
    roc_auc_7 = roc_auc_score(y_N7[labelled_inds], y_pred_proba_N7[labelled_inds])
    
    auprc_11 = average_precision_score(y_N11[labelled_inds], y_pred_proba_N11[labelled_inds])
    roc_auc_11 = roc_auc_score(y_N11[labelled_inds], y_pred_proba_N11[labelled_inds])
    
    
    return y_N3, y_pred_proba_N3, auprc_3, roc_auc_3, y_N7, y_pred_proba_N7, auprc_7, roc_auc_7, y_N11, y_pred_proba_N11, auprc_11, roc_auc_11
    

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
#     epochs = range(len(model_hist['loss']))
#     hmm_model_loss = model_hist['hmm_model_loss']
#     predictor_loss = model_hist['predictor_loss']
#     model_hist['epochs'] = epochs
#     model_hist['n_train'] = len(data_dict['train'][1])
#     model_hist['n_valid'] = len(data_dict['valid'][1])
#     model_hist['n_test'] = len(data_dict['test'][1])
    model_hist_df = pd.DataFrame(model_hist)
    
    # save to file
    model_hist_df.to_csv(save_file, index=False)
    print('Training plots saved to %s'%save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pchmm fitting')
    parser.add_argument('--train_np_files', type=str, required=True)
    parser.add_argument('--test_np_files', type=str, required=True)
#     parser.add_argument('--data_dict_files', type=str, required=True)
    parser.add_argument('--valid_np_files', type=str, default=None)
#     parser.add_argument('--imputation_strategy', type=str, default="no_imp")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of sequences per minibatch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--predictor_l2_penalty', type=float, default=0.,
                        help='l2 penalty for predictor weights')
    parser.add_argument('--init_strategy', type=str, default='kmeans')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--lamb', type=int, default=100,
                        help='Langrange multiplier')
    parser.add_argument('--n_states', type=int, default=10,
                        help='number of HMM states')
    parser.add_argument('--perc_labelled', type=str, default="100",
                        help='percentage of labelled examples')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='directory where trained model and loss curves over epochs are saved')
    parser.add_argument('--output_filename_prefix', type=str, default=None, 
                        help='prefix for the training history jsons and trained classifier')
    args = parser.parse_args()

    rs = RandomState(args.seed)

    x_train_np_filename, y_train_np_filename = args.train_np_files.split(',')
    x_test_np_filename, y_test_np_filename = args.test_np_files.split(',')
    x_valid_np_filename, y_valid_np_filename = args.valid_np_files.split(',')
    
    
    X_train = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename).astype(np.float32)
    y_test = np.load(y_test_np_filename).astype(np.float32)
    X_val = np.load(x_valid_np_filename)
    y_val = np.load(y_valid_np_filename).astype(np.float32)
    
    
    '''
    ts_feat_inds = [False, True, True, True, True, True, True, False, False, True, True, True, True]
    static_feat_inds = np.logical_not(ts_feat_inds)
    static_feat_idx = np.flatnonzero(static_feat_inds)
    
    # make sure the static features are known only at time 0
    for feat_idx in static_feat_idx:
        X_train[:, 1:, feat_idx]=np.nan
        X_val[:, 1:, feat_idx]=np.nan
        X_test[:, 1:, feat_idx]=np.nan
    '''
    
    N,T,F = X_train.shape
    n_classes = int(np.nanmax(y_train))+1
    
    print('number of data points : %d\nnumber of time points : %s\nnumber of features : %s\n'%(N,T,F))
    
    
    
    # mask labels in training set and validation set as per user provided %perc_labelled
    N_tr = len(X_train)
    N_va = len(X_val)
    N_te = len(X_test)
    
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_train, y_train)
    data_dict['valid'] = (X_val, y_val)
    data_dict['test'] = (X_test, y_test)
    
    
    # get the init means and covariances of the observation distribution by clustering
#     print('Initializing cluster means with kmeans...')
    prng = np.random.RandomState(args.seed)
    n_init = 15000
    init_inds = prng.permutation(N)[:n_init]# select random samples from training set
    X_flat = X_train[:n_init].reshape(n_init*T, X_train.shape[-1])# flatten across time
    X_flat = np.where(np.isnan(X_flat), ma.array(X_flat, mask=np.isnan(X_flat)).mean(axis=0), X_flat)
        
    # get cluster means
    n_states = args.n_states
    if args.init_strategy=='kmeans':
        print('Initializing cluster means with kmeans...')
        # get cluster means
        kmeans = KMeans(n_clusters=n_states, n_init=10).fit(X_flat)
        init_means = kmeans.cluster_centers_

        # get cluster covariance in each cluster
        init_covs = np.stack([np.zeros(F) for i in range(n_states)])
    if args.init_strategy=='per_class_kmeans_max_diversity':
        print('Initializing cluster means with per class kmeans with max diversity...')
        per_class_init_means = []
        per_class_kmeans_models = []

        for jj in range(n_classes):
            X_train_jj = X_train[y_train==jj]
            N = len(X_train_jj)
            
            # get cluster means
            kmeans = KMeans(n_clusters=n_states, n_init=5, random_state=args.seed).fit(X_flat)
    #         sorted_cluster_centers = kmeans.cluster_centers_[np.argsort(kmeans.cluster_centers_[:, 0]), :]

            per_class_kmeans_models.append(kmeans)
            per_class_init_means.append(kmeans.cluster_centers_)
        
        
        # aggregate the clusters such that there's max diversity amongst the learned clusters
        # Get the number of clusters
        num_clusters = per_class_kmeans_models[0].n_clusters

        # Initialize an array to store the aggregated cluster centers
        init_means = np.zeros((num_clusters, F))
        
        # Iterate over each cluster index
        for cluster_index in range(num_clusters):
            # Get the cluster centers for each class
            cluster_centers = []
            for model in per_class_kmeans_models:
                cluster_centers.append(model.cluster_centers_[cluster_index])


            # Calculate the pairwise distances between cluster centers
            pairwise_distances = np.zeros((len(per_class_kmeans_models), len(per_class_kmeans_models)))
            for i in range(len(per_class_kmeans_models)):
                for j in range(i + 1, len(per_class_kmeans_models)):
                    pairwise_distances[i, j] = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    pairwise_distances[j, i] = pairwise_distances[i, j]

            # Find the cluster center with maximum average distance to other centers
            average_distances = np.mean(pairwise_distances, axis=1)
            most_diverse_index = np.argmax(average_distances)

            # Store the most diverse cluster center
            init_means[cluster_index] = cluster_centers[most_diverse_index]


#         init_means = init_means[np.argsort(init_means[:, 0]), :]
        
        # get cluster covariance in each cluster
        init_covs = np.stack([np.zeros(F) for i in range(n_states)])
    elif args.init_strategy=='uniform':
        print('Initializing cluster means with random init...')
        percentile_ranges_F2 = np.percentile(X_flat, [1, 99], axis=0).T
        init_means = np.zeros((n_states, F))
        for f in range(F):
            low=percentile_ranges_F2[f, 0]
            high=percentile_ranges_F2[f, 1]
            if low==high:
                high=low+1
            init_means[:,f] = prng.uniform(low=low, 
                                            high=high, 
                                            size=args.n_states)
            init_covs = np.stack([np.zeros(F) for i in range(n_states)])
    
#     # get cluster covariance in each cluster
#     init_covs = np.stack([np.zeros(F) for i in range(n_states)])
    
    # train model
    optimizer = get_optimizer('adam', lr = args.lr)

#     if n_states<=10:
    initial_state_logits = tf.zeros([n_states]) # uniform distribution
    daily_change_prob = 0.5#0.05
    transition_probs = tf.fill([n_states, n_states],
                               daily_change_prob / (n_states - 1))
    transition_logits = np.log(tf.linalg.set_diag(transition_probs,
                                          tf.fill([n_states], 
                                                  1 - daily_change_prob)))
#     else:        
#         start_n_states = 5
#         # Give probability exp(-100) ~= 0 to states outside of the current model.
#         active_states_mask = tf.concat([tf.ones([start_n_states]),
#                                         tf.zeros([n_states - start_n_states])],
#                                         axis=0)
        
#         daily_change_prob = 0.5
#         initial_state_logits = -100. * (1 - active_states_mask)
#         transition_probs = tf.fill([start_n_states, start_n_states],
#                              0. if start_n_states == 1
#                              else daily_change_prob / (start_n_states - 1)) 
        
#         padded_transition_probs = tf.eye(n_states) + tf.pad(
#         tf.linalg.set_diag(transition_probs,
#                          tf.fill([start_n_states], - daily_change_prob)),
#             paddings=[(0, n_states - start_n_states),
#                 (0, n_states - start_n_states)])
        
#         transition_logits = np.log(padded_transition_probs+1e-5)
    
    model = HMM(
            states=n_states,                                     
            lam=args.lamb,                                      
            prior_weight=0.01,
            predictor_weight=args.predictor_l2_penalty,
            observation_dist='NormalWithMissing',
            observation_initializer=dict(loc=init_means, 
                scale=init_covs),
            initial_state_alpha=1.,
            initial_state_initializer=initial_state_logits,
            predictor_dist='OrderedLogistic',
            transition_alpha=1.,
            transition_initializer=transition_logits, 
        optimizer=optimizer)
    
    
#     data = custom_dataset(data_dict=data_dict)


    data = custom_regression_dataset(data_dict=data_dict)
    if args.batch_size==-1:
        data.batch_size=X_train.shape[0]
    else:
        data.batch_size = args.batch_size    
    
    model.build(data) 
        
#     # initialize the predictor weights
#     x_train,y_train = data.train().numpy()
#     z_train = model.hmm_model.predict(x_train)
#     z_train_mean = np.mean(z_train, axis=1)

#     ordinal_model = tf.keras.Sequential([tf.keras.layers.Dense(1),
#                                  tfp.layers.DistributionLambda(lambda t: tfp.distributions.OrderedLogistic(cutpoints=[.25, 
#                                                                                                                       .5, .75],
#                                                                                                            loc=t))])

#     ordinal_model.compile(optimizer=Adam(lr=0.001), loss=OrdinalLoss())
#     ordinal_model.fit(z_train_mean, y_train, epochs=100)
    
    
#     model._predictor.set_weights(ordinal_model.get_weights())
    
    
    
    steps_per_epoch = np.ceil(N/args.batch_size)

    
    # temporarily using pre-trained weights
    model.fit(data, 
              steps_per_epoch=steps_per_epoch, 
              epochs=1000,#200
              reduce_lr=False, 
              batch_size=args.batch_size, 
              lr=args.lr,
              initial_weights=model.model.get_weights())
    
    # train just the predictor after training HMM if lambda=0
    if args.lamb==0:
        print('Training the predictor only since lambda=0')
        x_train, y_train = data.train().numpy()
        z_train = model.hmm_model.predict(x_train)
        z_train_mean = np.mean(z_train, axis=1)
        
#         R = max(y_train)+1
#         cutpoints = tf.Variable(initial_value=tf.range(R-1, dtype=np.float32) / (R-2),
#                                 trainable=True)
        
        ordinal_model = tf.keras.Sequential([tf.keras.layers.Dense(1),
                                     tfp.layers.DistributionLambda(lambda t: tfp.distributions.OrderedLogistic(cutpoints=[.25, 
                                                                                                                          .5, .75],
                                                                                                               loc=t))])
        
        ordinal_model.compile(optimizer=Adam(lr=0.001), loss=OrdinalLoss())
        keep_inds = ~np.isnan(y_train)
        ordinal_model.fit(z_train_mean[keep_inds], y_train[keep_inds], epochs=100)
        
        model._predictor.set_weights(ordinal_model.get_weights())
        
    
    # evaluate on train set
    x_train, y_train = data.train().numpy()
    z_train = model.hmm_model.predict(x_train)
    y_train_pred_proba_NC = model._predictor(z_train).distribution.categorical_probs().numpy().squeeze()
    
    
    y_train_N3, y_train_pred_proba_N3, train_auprc_3, train_roc_auc_3, y_train_N7, y_train_pred_proba_N7, train_auprc_7, train_roc_auc_7, y_train_N11, y_train_pred_proba_N11, train_auprc_11, train_roc_auc_11 = compute_binary_classification_perf(y_train, y_train_pred_proba_NC)
    print('ROC AUC for los> 3 days on train : %.4f'%train_roc_auc_3)
    print('AUPRC for los> 3 days on train : %.4f'%train_auprc_3)
    
    print('ROC AUC for los> 7 days on train : %.4f'%train_roc_auc_7)
    print('AUPRC for los> 7 days on train : %.4f'%train_auprc_7)
    
    print('ROC AUC for los> 11 days on train : %.4f'%train_roc_auc_11)
    print('AUPRC for los> 11 days on train : %.4f'%train_auprc_11)
    
    
    
    
    # evaluate on valid set
    x_valid, y_valid = data.valid().numpy()
    z_valid = model.hmm_model.predict(x_valid)
    y_valid_pred_proba_NC = model._predictor(z_valid).distribution.categorical_probs().numpy().squeeze()
    
    
    y_valid_N3, y_valid_pred_proba_N3, valid_auprc_3, valid_roc_auc_3, y_valid_N7, y_valid_pred_proba_N7, valid_auprc_7, valid_roc_auc_7, y_valid_N11, y_valid_pred_proba_N11, valid_auprc_11, valid_roc_auc_11 = compute_binary_classification_perf(y_valid, y_valid_pred_proba_NC)
    print('ROC AUC for los> 3 days on valid : %.4f'%valid_roc_auc_3)
    print('AUPRC for los> 3 days on valid : %.4f'%valid_auprc_3)
    
    print('ROC AUC for los> 7 days on valid : %.4f'%valid_roc_auc_7)
    print('AUPRC for los> 7 days on valid : %.4f'%valid_auprc_7)
    
    print('ROC AUC for los> 11 days on valid : %.4f'%valid_roc_auc_11)
    print('AUPRC for los> 11 days on valid : %.4f'%valid_auprc_11)
    
    
    # evaluate on test set
    x_test, y_test = data.test().numpy()
    z_test = model.hmm_model.predict(x_test)
    y_test_pred_proba_NC = model._predictor(z_test).distribution.categorical_probs().numpy().squeeze()

    y_test_N3, y_test_pred_proba_N3, test_auprc_3, test_roc_auc_3, y_test_N7, y_test_pred_proba_N7, test_auprc_7, test_roc_auc_7, y_test_N11, y_test_pred_proba_N11, test_auprc_11, test_roc_auc_11 = compute_binary_classification_perf(y_test, y_test_pred_proba_NC)
    print('ROC AUC for los> 3 days on test : %.4f'%test_roc_auc_3)
    print('AUPRC for los> 3 days on test : %.4f'%test_auprc_3)
    
    print('ROC AUC for los> 7 days on test : %.4f'%test_roc_auc_7)
    print('AUPRC for los> 7 days on test : %.4f'%test_auprc_7)
    
    print('ROC AUC for los> 11 days on test : %.4f'%test_roc_auc_11)
    print('AUPRC for los> 11 days on test : %.4f'%test_auprc_11)    
    
    # get the parameters of the fit distribution
    mu_all = model.hmm_model(x_train[:10]).observation_distribution.distribution.mean().numpy()
    try:
        cov_all = model.hmm_model(x_train[:10]).observation_distribution.distribution.covariance().numpy()
    except:
        cov_all = model.hmm_model(x_train[:10]).observation_distribution.distribution.scale.numpy()
        
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
    model_perf_df = pd.DataFrame([{'train_AUPRC_3' : train_auprc_3, 
                                   'valid_AUPRC_3' : valid_auprc_3, 
                                   'test_AUPRC_3' : test_auprc_3,
                                   'train_AUC_3' : train_roc_auc_3,
                                   'valid AUC_3' : valid_roc_auc_3,
                                   'test AUC_3' : test_roc_auc_3,
                                   'train_AUPRC_7' : train_auprc_7, 
                                   'valid_AUPRC_7' : valid_auprc_7, 
                                   'test_AUPRC_7' : test_auprc_7,
                                   'train_AUC_7' : train_roc_auc_7,
                                   'valid AUC_7' : valid_roc_auc_7,
                                   'test AUC_7' : test_roc_auc_7,
                                   'train_AUPRC_11' : train_auprc_11, 
                                   'valid_AUPRC_11' : valid_auprc_11, 
                                   'test_AUPRC_11' : test_auprc_11,
                                   'train_AUC_11' : train_roc_auc_11,
                                   'valid AUC_11' : valid_roc_auc_11,
                                   'test AUC_11' : test_roc_auc_11}])
    
    
    model_perf_df.to_csv(perf_save_file, index=False)
    print('performance saved to %s'%perf_save_file)
    
    
# if __name__ == '__main__':
#     main()
    
    '''
    Debugging code
    
    from sklearn.linear_model import LogisticRegression
    # get 10 positive and 10 negative sequences
    pos_inds = np.where(y[:,1]==1)[0]
    neg_inds = np.where(y[:,0]==1)[0]
    
    x_pos = x[pos_inds]
    y_pos = y[pos_inds]
    x_neg = x[neg_inds]
    y_neg = y[neg_inds]
    
    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)
    
    # print the average belief states across time
    beliefs_pos = z_pos.mean(axis=1)
    beliefs_neg = z_neg.mean(axis=1)
    
    # perform logistic regression with belief states as features for these positive and negative samples
    
    logistic = LogisticRegression(solver='lbfgs', random_state = 42)
    
    
    penalty = ['l2']
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4]
    hyperparameters = dict(C=C, penalty=penalty)
    classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = 'roc_auc')
    
    X_tr = np.vstack([beliefs_pos, beliefs_neg])
    y_tr = np.vstack([y_pos, y_neg])
    cv = classifier.fit(X_tr, y_tr[:,1])
    
    # get the logistic regression coefficients. These are the optimal eta coefficients.
    lr_weights = cv.best_estimator_.coef_
    
    # plot the log probas from of a few negative and positive samples using the logistic regression estimates 
    n=50
    x_pos = x[pos_inds[:n]]
    y_pos = y[pos_inds[:n]]
    x_neg = x[neg_inds[:n]]
    y_neg = y[neg_inds[:n]]

    # get their belief states of the positive and negative sequences
    z_pos = model.hmm_model.predict(x_pos)
    z_neg = model.hmm_model.predict(x_neg)

    # print the average belief states across time
    beliefs_pos = z_pos.mean(axis=1)
    beliefs_neg = z_neg.mean(axis=1)
    
    X_eval = np.vstack([beliefs_pos, beliefs_neg])
    predict_probas = cv.predict_log_proba(X_eval)
    predict_probas_pos = predict_probas[:n, 1]
    predict_probas_neg = predict_probas[n:, 1]
    
    f, axs = plt.subplots(1,1, figsize=(5, 5))
    axs.plot(np.zeros_like(predict_probas_neg), predict_probas_neg, '.')
    axs.plot(np.ones_like(predict_probas_pos), predict_probas_pos, '.')
    axs.set_ylabel('log p(y=1)')
    axs.set_xticks([0, 1])
    axs.set_xticklabels(['50 negative samples', '50 positive samples'])
    f.savefig('predicted_proba_viz_post_lr.png')
    
    # plot some negative sequences that have a high log p(y=1)
    outcast_inds = np.where(predict_probas_neg>-3)
    x_neg_outcasts = x_neg(outcast_inds)
    f, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x_neg_outcasts[:, :, :, 0], x_neg_outcasts[:, :, :, 1], s=2, marker='x')
    ax.set_xlim([-5, 50])
    ax.set_ylim([-5, 5])
    fontsize=10
    ax.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
    ax.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)
    ax.set_title('negative sequences with log p(y=1)>-3')
    f.savefig('x_outcasts.png')
    
    ###### Converting logistic regression weights to etas
    # set the lr weights as initial etas
    opt_eta_weights = np.vstack([np.zeros_like(lr_weights), lr_weights]).T
    init_etas[0] = opt_eta_weights
    model._predictor.set_weights(init_etas)
    
    # print the predicted probabilities of class 0 and class 1
    y_pred_pos = model._predictor(z_pos).distribution.logits.numpy()
    y_pred_neg = model._predictor(z_neg).distribution.logits.numpy()
    
    # the 2 lines above is the same as 
    y_pred_pos = np.matmul(beliefs_pos, opt_eta_weights)+opt_eta_intercept
    y_pred_neg = np.matmul(beliefs_neg, opt_eta_weights)+opt_eta_intercept
    
    
    # plot the log p(y=1) for the negative and positive samples
    y1_pos = expit(y_pred_pos[:,1])
    y1_neg = expit(y_pred_neg[:,1])
    f, axs = plt.subplots(1,1, figsize=(5, 5))
    axs.plot(np.zeros_like(y1_neg), y1_neg, '.')
    axs.plot(np.ones_like(y1_pos), y1_pos, '.')
    axs.set_ylabel('p(y=1)')
    axs.set_xticks([0, 1])
    axs.set_xticklabels(['50 negative samples', '50 positive samples'])
    f.savefig('predicted_proba_viz.png')
    ''' 
    '''
    Train only the predictor
    -------------------------
    mlp = MLPClassifier(solver='lbfgs', random_state = 42)
    alpha = [1e-3, 1e-2, 1e-1, 1e0]
    hidden_layer_sizes = [(8, 8, ), (16, 16, ), (32, 32, )]
    hyperparameters = dict(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes)
    classifier = GridSearchCV(mlp, hyperparameters, cv=5, verbose=10, scoring = 'average_precision')
    

    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    max_features = [0.33, 0.66, .99]
    max_samples = [0.33, 0.66, .99]
    hyperparameters = dict(max_features=max_features, max_samples=max_samples)
    classifier = GridSearchCV(rf, hyperparameters, cv=5, verbose=10, scoring = 'average_precision')

    cv = classifier.fit(z_train_mean, y_train[:,1])
    '''

