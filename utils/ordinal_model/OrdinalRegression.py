#!/usr/bin/env python3
"""
TODO: Module Docstring
"""
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.scipy.stats import norm
from scipy.optimize import minimize

from ordinal_model.ordinal_log_likelihood import log_likelihood, proba


class OrdinalRegression:
    def __init__(self, padding: float = 6, noise_variance: float = 1, random_state=None) -> None:
        # Parameters
        self.w = None
        self.X = None
        self.cut_points = None
        self.padding = padding
        self.noise_variance = noise_variance

        # Random State
        self.rs = np.random.RandomState(random_state)
        return None

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO:
        # * Decide if value_and_grad or grad should be used
        # * Does minimize work since the loss_function is convex?
        # * Choose method and jac=bool
        # * How should init_w be determined?

        # Dimensions of X
        self.N = X.shape[0]
        self.M = X.shape[1]

        # Transform
        # * Add weight field to X
        # * Assuming ordinal class starts at 0, add 1 to all?
        X_transformed = np.hstack((np.ones((self.N, 1)), X))
        # print(X_transformed)
        # y_transformed = y + 1

        # Cutpoints
        ordinal_outcomes = set(range(y.max()+1))
        self.R = len(ordinal_outcomes)
        cut_points_start = -self.padding*(self.R-2)/2
        cut_points_end = -cut_points_start + self.padding
        base_cut_points = np.arange(
            cut_points_start, cut_points_end, self.padding)
        self.cut_points = np.hstack((-np.inf, base_cut_points, np.inf))

        # Initialization
        init_w = self.rs.rand(self.M+1)

        # MLE
        def loss_function(w):
            return -self.log_likelihood(w, X_transformed, y)
        # def loss_function(w):
        #     return -log_likelihood(w, X_transformed, y, self.cut_points, self.noise_variance)

        # print(loss_function(init_w))

        # dldw = grad(loss_function)
        # print(dldw)
        # print(dldw(init_w))

        self.w = minimize(
            value_and_grad(loss_function),
            init_w,
            jac=True,
            method='CG',
        ).x
        return None

    def predict(self, X) -> np.ndarray:
        X_transformed = np.hstack((np.ones((X.shape[0], 1)), X))
        best_proba_NR = self.proba(self.w, X_transformed)
        y_predict = np.argmax(best_proba_NR, axis=1)
        return y_predict

    def log_likelihood(self, w, X, y) -> float:
        # TODO: Figure out how to set up the log_likelihood so that it can be
        # passed into minimize(value_and_grad(-log_likelihood))
        # I'm a little confused how to calculate the log_likelihood wihtout the
        # weight vector, which I think should be passed into it
        # * Do I need to return a number or a vector?
        N = X.shape[0]

        # Log Likelihood
        proba_NR = self.proba(w, X)
        # print('PROBA')
        # print(proba_NR)
        # print(proba_NR.shape)
        log_likelihood_N = np.log(proba_NR[np.arange(N), y])
        # print('LIKELIHOOD')
        # print(log_likelihood_N)
        # print(log_likelihood_N.shape)
        return np.sum(log_likelihood_N)

    def proba(self, w, X) -> np.ndarray:
        # z_matrix_NR2 = np.empty((self.N, self.R, 2))
        N = X.shape[0]
        z_matrix_NR_1 = np.zeros((N, 0))
        z_matrix_NR_2 = np.zeros((N, 0))
        for bi in range(self.R):
            # Note: Indexing is not supported
            # z_matrix_NR2[:, bi, 0] = (
            #     self.cut_points[bi+1] - (X @ w)
            # ) / np.sqrt(self.noise_variance)
            # z_matrix_NR2[:, bi, 1] = (
            #     self.cut_points[bi] - (X @ w)
            # ) / np.sqrt(self.noise_variance)
            z_bi_1 = ((self.cut_points[bi+1] - (X @ w)) /
                      np.sqrt(self.noise_variance))[:, np.newaxis]
            z_bi_2 = ((self.cut_points[bi] - (X @ w)) /
                      np.sqrt(self.noise_variance))[:, np.newaxis]
            # if z_matrix_NR_1 is None and z_matrix_NR_2 is None:
            #     z_matrix_NR_1 = z_bi_1
            #     z_matrix_NR_2 = z_bi_2
            # else:
            z_matrix_NR_1 = np.hstack((z_matrix_NR_1, z_bi_1))
            z_matrix_NR_2 = np.hstack((z_matrix_NR_2, z_bi_2))
        z_matrix_NR2 = np.concatenate(
            (z_matrix_NR_1[:, :, np.newaxis], z_matrix_NR_2[:, :, np.newaxis]), axis=2)
        gaussian_cdf_NR2 = norm.cdf(z_matrix_NR2)
        proba_NR = gaussian_cdf_NR2[:, :, 0] - gaussian_cdf_NR2[:, :, 1]
        return proba_NR

    def predict_proba(self, X) -> np.ndarray:
        X_transformed = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.proba(self.w, X_transformed)
