import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                             balanced_accuracy_score, confusion_matrix, 
                             roc_auc_score, make_scorer, average_precision_score)
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
from pcvae.datasets.toy import toy_line, custom_dataset, custom_regression_dataset
from pcvae.models.hmm import HMM
from pcvae.datasets.base import dataset, real_dataset, classification_dataset, make_dataset
from sklearn.model_selection import train_test_split
from pcvae.util.optimizers import get_optimizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy.ma as ma
from sklearn.cluster import KMeans
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow as tf

class OrdinalLoss(tf.keras.losses.Loss):
    """Ordinal Loss class"""
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ordinal_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)


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
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pchmm fitting')
    parser.add_argument('--outcome_col_name', type=str, required=True)
    parser.add_argument('--train_csv_files', type=str, required=True)
    parser.add_argument('--valid_csv_files', type=str, default=None)
    parser.add_argument('--test_csv_files', type=str, required=True)
#     parser.add_argument('--data_dict_files', type=str, required=True)
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
    parser.add_argument('--n_states', type=int, default=10,
                        help='number of states')
    parser.add_argument('--missing_handling', type=str, default="no_imp",
                        help='how to handle missing observations')
    parser.add_argument('--perc_missing', type=str, default=20,
                        help='percentage missing observations')
    
    args = parser.parse_args()

    rs = RandomState(args.seed)

    x_train_csv_filename, y_train_csv_filename = args.train_csv_files.split(',')
    x_test_csv_filename, y_test_csv_filename = args.test_csv_files.split(',')
    
    x_train_np_filename, y_train_np_filename = args.train_csv_files.split(',')
    x_test_np_filename, y_test_np_filename = args.test_csv_files.split(',')
    x_valid_np_filename, y_valid_np_filename = args.valid_csv_files.split(',')
    
    
    X_train = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename).astype(np.float32)
    y_test = np.load(y_test_np_filename).astype(np.float32)
    X_val = np.load(x_valid_np_filename)
    y_val = np.load(y_valid_np_filename).astype(np.float32)
    
    
    '''
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
    '''
    
    
    
    # get the init means and covariances of the observation distribution by clustering
    print('Initializing per class cluster means with kmeans...')
    prng = np.random.RandomState(args.seed)
    
    n_classes = int(y_train.max())+1
    n_states = args.n_states
    
    _,T,F = X_train.shape
    per_class_init_means = []
    per_class_kmeans_models = []
    
    for jj in range(n_classes):
        X_train_jj = X_train[y_train==jj]
        N = len(X_train_jj)
        init_inds = prng.permutation(N)# select random samples from training set
        X_flat = X_train_jj.reshape(N*T, X_train_jj.shape[-1])# flatten across time
        X_flat = np.where(np.isnan(X_flat), ma.array(X_flat, mask=np.isnan(X_flat)).mean(axis=0), X_flat)
        
        
        # get cluster means
        kmeans = KMeans(n_clusters=n_states, n_init=10).fit(X_flat)
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

    
    init_means = init_means[np.argsort(init_means[:, 0]), :]
        
    # get cluster covariance in each cluster
    init_covs = np.stack([np.array([0., 0.]) for i in range(n_states)])
    
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    
    # standardize data for PC-HMM
    key_list = ['train', 'valid', 'test']
    data_dict = dict.fromkeys(key_list)
    data_dict['train'] = (X_train, y_train.astype(np.float32))
    data_dict['valid'] = (X_val, y_val.astype(np.float32))
    data_dict['test'] = (X_test, y_test.astype(np.float32))
    
    
    # train model
    optimizer = get_optimizer('adam', lr = args.lr)
    
    init_state_probas = (1/n_states)*np.ones(n_states)
    
    model = HMM(
            states=n_states,                                     
            lam=args.lamb,                                      
            prior_weight=0.01,
            observation_dist='NormalWithMissing',
            observation_initializer=dict(loc=init_means, 
                scale=init_covs),
            initial_state_alpha=1.,
            predictor_dist='OrderedLogistic',
            initial_state_initializer=np.log(init_state_probas),
            transition_alpha=1.,
            transition_initializer=None, optimizer=optimizer)
    
    
    data = custom_regression_dataset(data_dict=data_dict)
    
    if args.batch_size==-1:
        data.batch_size = X_train.shape[0]
    else:
        data.batch_size = args.batch_size
    model.build(data) 
    
    
    x_train,y_train = data.train().numpy()
    z_train = model.hmm_model.predict(x_train)
    z_train_mean = np.mean(z_train, axis=1)
    

    ordinal_model = tf.keras.Sequential([tf.keras.layers.Dense(1),
                                 tfp.layers.DistributionLambda(lambda t: tfp.distributions.OrderedLogistic(cutpoints=[.25, 
                                                                                                                      0.5, .75],
                                                                                                           loc=t))])

    ordinal_model.compile(optimizer=Adam(lr=0.05), loss=OrdinalLoss())
    ordinal_model.fit(z_train_mean, y_train, epochs=100)
    
    
    model._predictor.set_weights(ordinal_model.get_weights())
    
    model.fit(data, steps_per_epoch=1, epochs=500, 
              batch_size=args.batch_size, 
              reduce_lr=False,
              lr=args.lr,
              initial_weights=model.model.get_weights()
             )
    
    
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
    
    
    # evaluate on train set
    x_train, y_train = data.train().numpy()
    z_train = model.hmm_model.predict(x_train)
    y_train_pred_proba_NC = model._predictor(z_train).distribution.categorical_probs().numpy().squeeze()
    

    # evaluate on valid set
    x_valid, y_valid = data.valid().numpy()
    z_valid = model.hmm_model.predict(x_valid)
    y_valid_pred_proba_NC = model._predictor(z_valid).distribution.categorical_probs().numpy().squeeze()
    
    # evaluate on test set
    x_test, y_test = data.test().numpy()
    z_test = model.hmm_model.predict(x_test)
    y_test_pred_proba_NC = model._predictor(z_test).distribution.categorical_probs().numpy().squeeze()    
    
    
    perf_dict = {}
    perf_dict['train_AUC_mean']=0
    perf_dict['valid_AUC_mean']=0
    perf_dict['test_AUC_mean']=0
    
    perf_dict['train_AUPRC_mean']=0
    perf_dict['valid_AUPRC_mean']=0
    perf_dict['test_AUPRC_mean']=0
    
    for jj in range(n_classes):
        class_inds_tr = y_train==jj
        perf_dict['train_AUC_%s'%jj]=roc_auc_score((class_inds_tr)*1, y_train_pred_proba_NC[:, jj])
        perf_dict['train_AUPRC_%s'%jj]=average_precision_score((class_inds_tr)*1, y_train_pred_proba_NC[:, jj])
        
        class_inds_va = y_valid==jj
        perf_dict['valid_AUC_%s'%jj]=roc_auc_score((class_inds_va)*1, y_valid_pred_proba_NC[:, jj])
        perf_dict['valid_AUPRC_%s'%jj]=average_precision_score((class_inds_va)*1, y_valid_pred_proba_NC[:, jj])        
    
        class_inds_te = y_test==jj
        perf_dict['test_AUC_%s'%jj]=roc_auc_score((class_inds_te)*1, y_test_pred_proba_NC[:, jj])
        perf_dict['test_AUPRC_%s'%jj]=average_precision_score((class_inds_te)*1, y_test_pred_proba_NC[:, jj])     
        
        # compute average auc across classes
        perf_dict['train_AUC_mean'] += perf_dict['train_AUC_%s'%jj]/n_classes
        perf_dict['valid_AUC_mean'] += perf_dict['valid_AUC_%s'%jj]/n_classes
        perf_dict['test_AUC_mean'] += perf_dict['test_AUC_%s'%jj]/n_classes
        
        perf_dict['train_AUPRC_mean'] += perf_dict['train_AUPRC_%s'%jj]/n_classes
        perf_dict['valid_AUPRC_mean'] += perf_dict['valid_AUPRC_%s'%jj]/n_classes
        perf_dict['test_AUPRC_mean'] += perf_dict['test_AUPRC_%s'%jj]/n_classes
    
    
    print(perf_dict)
        
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
    
    model_perf_df = pd.DataFrame([perf_dict])
    
    model_perf_df.to_csv(perf_save_file, index=False)
    print('performance saved to %s'%perf_save_file)
        