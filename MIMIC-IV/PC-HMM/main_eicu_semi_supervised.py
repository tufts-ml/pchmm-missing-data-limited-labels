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
from pcvae.datasets.toy import toy_line, custom_dataset
from pcvae.models.hmm import HMM
from pcvae.datasets.base import dataset, real_dataset, classification_dataset, make_dataset
from sklearn.model_selection import train_test_split
from pcvae.util.optimizers import get_optimizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy.ma as ma
from sklearn.cluster import KMeans

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
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_val = np.load(x_valid_np_filename)
    y_val = np.load(y_valid_np_filename)
    
        
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

    # draw the initial probabilities from dirichlet distribution
#     prng = np.random.RandomState(args.seed)
#     init_state_probas = prng.dirichlet(5*np.ones(n_states))
#     initial_state_logits = np.log(init_state_probas)
    
    initial_state_logits = tf.zeros([n_states]) # uniform distribution
    daily_change_prob = 0.5#0.05
    transition_probs = tf.fill([n_states, n_states],
                               daily_change_prob / (n_states - 1))
    transition_logits = np.log(tf.linalg.set_diag(transition_probs,
                                          tf.fill([n_states], 
                                                  1 - daily_change_prob)))
    
    
    
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
            transition_alpha=1.,
            transition_initializer=transition_logits, 
        optimizer=optimizer)
    
    
    data = custom_dataset(data_dict=data_dict)
    if args.batch_size==-1:
        data.batch_size=X_train.shape[0]
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
    classifier = GridSearchCV(logistic, hyperparameters, cv=5, verbose=10, scoring = 'average_precision')
    
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
    
    steps_per_epoch = np.ceil(N/args.batch_size)
        
    # temporarily using pre-trained weights
    '''
    best_model_weights = '/cluster/tufts/hugheslab/prath01/results/mimic3/semi_supervised_pchmm/v05112022/semi-supervised-pchmm-lr=0.05-seed=2577-init_strategy=uniform-batch_size=512-perc_labelled=100-predictor_l2_penalty=0-n_states=5-lamb=1000-weights.h5'
    
    model_init = HMM(
            states=5,                                     
            lam=args.lamb,                                      
            observation_dist='NormalWithMissing')
    
    model_init.build(data)
    model_init.model.load_weights(best_model_weights)
    init_model_weights = model_init.model.get_weights()
    transfer_model_weights = model.model.get_weights()
    for ii, ww in enumerate(transfer_model_weights):
        if ii<=1:
            continue
        if ii==1:
            ww[:5, :5]=init_model_weights[ii]
        else:
            ww[:5] = init_model_weights[ii]
    
    model.model.set_weights(transfer_model_weights)
    '''
        
    '''
    model.fit(data, 
          steps_per_epoch=steps_per_epoch, 
          epochs=1000
          reduce_lr=False, 
          batch_size=args.batch_size, 
          lr=args.lr,
          initial_weights=model.model.get_weights())
    '''
    model.fit(data, 
              steps_per_epoch=steps_per_epoch, 
              epochs=200,#150
              reduce_lr=False, 
              batch_size=args.batch_size, 
              lr=args.lr,
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
    
    # evaluate on test set
#     x_train, y_train = data.train().numpy()
    z_train = model.hmm_model.predict(x_train)
    y_train_pred_proba = model._predictor.predict(z_train)
    labelled_inds_tr = ~np.isnan(y_train[:,0])
    train_roc_auc = roc_auc_score(y_train[labelled_inds_tr, 1], y_train_pred_proba[labelled_inds_tr, 1])
    train_auprc = average_precision_score(y_train[labelled_inds_tr, 1], y_train_pred_proba[labelled_inds_tr, 1])
    print('ROC AUC on train : %.4f'%train_roc_auc)
    print('AUPRC on train : %.4f'%train_auprc)
    
    
    # evaluate on valid set
    x_valid, y_valid = data.valid().numpy()
    z_valid = model.hmm_model.predict(x_valid)
    y_valid_pred_proba = model._predictor.predict(z_valid)
    labelled_inds_va = ~np.isnan(y_valid[:,0])
    valid_roc_auc = roc_auc_score(y_valid[labelled_inds_va], y_valid_pred_proba[labelled_inds_va])
    valid_auprc = average_precision_score(y_valid[labelled_inds_va, 1], y_valid_pred_proba[labelled_inds_va, 1])
    print('ROC AUC on valid : %.4f'%valid_roc_auc)
    print('AUPRC on valid : %.4f'%valid_auprc)
    
    
    # evaluate on test set
    x_test, y_test = data.test().numpy()
    z_test = model.hmm_model.predict(x_test)
    y_test_pred_proba = model._predictor.predict(z_test)
    labelled_inds_te = ~np.isnan(y_test[:,0])
    test_roc_auc = roc_auc_score(y_test[labelled_inds_te], y_test_pred_proba[labelled_inds_te])
    test_auprc = average_precision_score(y_test[labelled_inds_te, 1], y_test_pred_proba[labelled_inds_te, 1])
    print('ROC AUC on test : %.4f'%test_roc_auc)
    print('AUPRC on test : %.4f'%test_auprc)
    
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
    model_perf_df = pd.DataFrame([{'train_AUPRC' : train_auprc, 
                                   'valid_AUPRC' : valid_auprc, 
                                   'test_AUPRC' : test_auprc,
                                   'train_AUC' : train_roc_auc,
                                   'valid_AUC' : valid_roc_auc,
                                   'test_AUC' : test_roc_auc}])
    
    
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

