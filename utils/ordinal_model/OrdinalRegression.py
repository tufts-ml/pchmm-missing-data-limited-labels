#!/usr/bin/env python3
"""
Maximum-likelihood estimator for ordinal regression.

Usage
-----
TODO
"""
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.scipy.stats import norm
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler

from ordinal_model.ordinal_log_likelihood import log_likelihood, proba


class OrdinalRegression:
    """Class to fit and predict ordinal outcomes.
    """

    def __init__(self, noise_variance: float = 1, random_state: int = None) -> None:
        """Constructer for OrdinalRegression class.

        Parameters
        ----------
        noise_variance : float, optional
            _description_, by default 1
        random_state : int, optional
            _description_, by default None

        Returns
        -------
        None
        """
        # Parameters
        self.w = None
        self.noise_variance = noise_variance

        # Random State
        self.rs = np.random.RandomState(random_state)
        return None

    def set_params(self, **kwargs) -> None:
        """Class method to manually set parameters if desired.

        Useful for testing

        Returns
        -------
        None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """_summary_

        Parameters
        ----------
        X : np.ndarray
            _description_
        y : np.ndarray
            _description_

        Returns
        -------
        None

        TODO
        ----
        * What should the parameters for scipy.minimize be?
        * Should be eventually rewritten in tensorflow/pytorch
        """
        # Relevant parameters
        N, M = X.shape
        R = y.max()+1

        # Transform
        # * Add weight field to X
        # * Assuming ordinal class starts at 0
        X_transformed = self._transform(X)

        # Cutpoints
        # ordinal_outcomes = set(range(y.max()+1))
        # self.R = len(ordinal_outcomes)
        # cut_points_start = -self.padding*(self.R-2)/2
        # cut_points_end = -cut_points_start + self.padding
        # base_cut_points = np.arange(
        #     cut_points_start, cut_points_end, self.padding)
        # self.cut_points = np.hstack((-np.inf, base_cut_points, np.inf))

        # Initialized parameters
        # Combine into a single np.ndarray since scipy only accepts arrays
        init_w = self.rs.normal(size=M+1)
        init_cut_points = np.sort(self.rs.normal(size=R-1))
        init_params = np.hstack((init_w, init_cut_points))

        # MLE Estimate
        # Set up loss function
        def loss_function(params):
            w = params[:M+1]
            b = params[M+1:]
            return -self.log_likelihood(w, b, X_transformed, y)

        # Use scipy.minimize to find global minimum and return parameters
        params = minimize(
            value_and_grad(loss_function),
            init_params,
            jac=True,
            method='CG',
        ).x
        self.w = params[:M+1]
        self.b = params[M+1:]
        return None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.

        Uses standard scaler to scale data and add ones column for the bias
        weight.

        Parameters
        ----------
        X : np.ndarray
            _description_

        Returns
        -------
        X_transformed : np.ndarray
            _description_
        """
        # Add additional column for bias weight
        N = X.shape[0]
        X = np.hstack((np.ones((N, 1)), X))

        # Standard scale data
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X_transformed = self.scaler.transform(X)
        return X_transformed

    def _inverse_transform(self, X_transformed) -> np.ndarray:
        """Reconstruct the original data.

        Parameters
        ----------
        X_transformed : _type_
            _description_

        Returns
        -------
        X : np.ndarray
            _description_
        """
        return self.scaler.inverse_transform(X_transformed)[:, 1:]

    def predict(self, X) -> np.ndarray:
        """Predict ordinal outcomes.

        Parameters
        ----------
        X : _type_
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        X_transformed = self._transform(X)
        best_proba_NR = self.proba(self.w, self.b, X_transformed)
        y_predict = np.argmax(best_proba_NR, axis=1)
        return y_predict

    def log_likelihood(self, w, b, X, y) -> float:
        """Compute log-likelihood.

        Parameters
        ----------
        w : _type_
            _description_
        b : _type_
            _description_
        X : _type_
            _description_
        y : _type_
            _description_

        Returns
        -------
        log_likelihood : float
            _description_
        """
        # Useful parameters
        N = X.shape[0]

        # Log Likelihood
        proba_NR = self.proba(w, b, X)
        log_likelihood_N = np.log(proba_NR[np.arange(N), y])
        return np.sum(log_likelihood_N)

    def proba(self, w, b, X) -> np.ndarray:
        """Compute probabilities of each ordinal outcome given a set of weights
        and cut-points.

        Parameters
        ----------
        w : _type_
            _description_
        b : _type_
            _description_
        X : _type_
            _description_

        Returns
        -------
        proba : np.ndarray
            _description_
        """
        # Initialize values
        b = np.hstack((-np.inf, b, np.inf))
        N = X.shape[0]
        R = b.size - 1
        z_matrix_NR_1 = np.zeros((N, 0))
        z_matrix_NR_2 = np.zeros((N, 0))
        # Iterate through possible ordinal outcomes
        for j in range(R):
            # Note: Indexing is not supported
            # z_matrix_NR2[:, bi, 0] = (
            #     self.cut_points[bi+1] - (X @ w)
            # ) / np.sqrt(self.noise_variance)
            # z_matrix_NR2[:, bi, 1] = (
            #     self.cut_points[bi] - (X @ w)
            # ) / np.sqrt(self.noise_variance)
            z_bj_1 = ((b[j+1] - (X @ w)) /
                      np.sqrt(self.noise_variance))[:, np.newaxis]
            z_bj_2 = ((b[j] - (X @ w)) /
                      np.sqrt(self.noise_variance))[:, np.newaxis]
            # if z_matrix_NR_1 is None and z_matrix_NR_2 is None:
            #     z_matrix_NR_1 = z_bi_1
            #     z_matrix_NR_2 = z_bi_2
            # else:
            z_matrix_NR_1 = np.hstack((z_matrix_NR_1, z_bj_1))
            z_matrix_NR_2 = np.hstack((z_matrix_NR_2, z_bj_2))
        z_matrix_NR2 = np.concatenate(
            (z_matrix_NR_1[:, :, np.newaxis], z_matrix_NR_2[:, :, np.newaxis]), axis=2)
        gaussian_cdf_NR2 = norm.cdf(z_matrix_NR2)
        proba_NR = gaussian_cdf_NR2[:, :, 0] - gaussian_cdf_NR2[:, :, 1]
        return proba_NR

    def predict_proba(self, X) -> np.ndarray:
        """Obtain the probabilities of each ordinal outcome given the best
        weights and cut-points.

        Parameters
        ----------
        X : _type_
            _description_

        Returns
        -------
        best_proba : np.ndarray
            _description_
        """
        X_transformed = self._transform(X)
        return self.proba(self.w, self.b, X_transformed)
