#!/usr/bin/env python3
"""
Maximum-likelihood estimator for ordinal regression.

TODO
----
* [Done] Write additional method for using softplus link function: f(x) = ln(1+e^x)
* [Done] Include parameter on fit to show loss vs epochs
* [N/A] Include parameter on fit to show traceplots
* [Done] Write plotting function for loss vs epochs
* [N/A ]Write plotting function for traceplots
* [Done] Remove scaling
    * Initialize weights accordingly
* [Done] Confirm with Preetish/Hughes what exactly needs to be plotted for the traceplots
    * [Done] Preetish suggested decision boundaries
* [Done] Vectorize the probability computation so that we do not have to loop
* Develop a log_proba method so that we do not have to worry about over/underflow
    * [Done] Need to discuss how to properly set up with Prof. Hughes
    * Use logsumexp or equivalent to compute log_proba
* Test on multi-dimensional (3+) data
* Test with kernel to improve separation boundaries

FIXME
-----
* Optimize cannot train on variance
    * [Done] Generate plot of loss at various variances by grid searching on variance
      once the rest of the parameters have been optimally found
      (hope that this will illuminate why the params are going to nan)
    * [Done] Alternate optimizing with constant variance and finding minimum loss variance
      using trained parameters. Generate gif plot to show learning.
    * SOLUTION: freeze sigma/variance to 1 since model is over-parameterized
"""
import pathlib

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
# import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import autograd.numpy as ag_np
from autograd import grad, value_and_grad
import autograd.scipy.stats as ag_stats
import autograd.scipy.special as ag_special
# from autograd.scipy.stats import norm
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler

import imageio

################################################################################
# Class
################################################################################


class OrdinalRegression:
    """Class to fit and predict ordinal outcomes.
    """

    def __init__(self,
                 noise_variance: float = 1,
                 C: float = 0,
                 save_loss: bool = False,
                 log_training: bool = False,
                 random_state: int = None,
                 ) -> None:
        """Constructor for OrdinalRegression class.

        Parameters
        ----------
        noise_variance : float, optional
            Variance of gaussian noise that assume to contaminate latent 
            function, by default 1
        C : float, optional
            Regularization strength of cutpoints
        save_loss : bool, optional
            Whether to save the save the loss progression as training occurs
        random_state : int, optional
            Seed to set random state if desired, by default None

        Returns
        -------
        None

        TODO
        ----
        """
        # Parameters
        self.w = None
        self.noise_variance = noise_variance
        self.C = C

        # Random State
        self.rs = ag_np.random.RandomState(random_state)

        # File path(s)
        # self.directory = pathlib.Path(__file__).parent
        self.directory = pathlib.Path.cwd()

        # Logging
        self.save_loss = save_loss
        if log_training:
            self.directory.joinpath('logging').mkdir(exist_ok=True)
            self.log_file = self.directory.joinpath('logging', 'log.txt')
            self.log_file.touch()
        else:
            self.log_file = None
        return None

    def set_params(self, **kwargs) -> None:
        """Class method to manually set parameters if desired.

        Useful for testing.

        Returns
        -------
        None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return None

    def fit(self, X: ag_np.ndarray, y: ag_np.ndarray, fit_noise_variance: float = None) -> None:
        """Fit the model using the training data and ordinal labels.

        Parameters
        ----------
        X : ag_np.ndarray
            Raw feature data
        y : ag_np.ndarray
            Ground truth ordinal labels
        fit_noise_variance : float
            Train the model using a specific noise variance if desired

        Returns
        -------
        None

        TODO
        ----
        * What should the parameters for scipy.minimize be?--DONE
            * Prof. Hughes suggested to switch back to value_and_grad--Done
        * Rewrite parameters to be dummy values that get passed through
          softplus in the learning function so that we ensure positive real--DONE
        * Add epsilons additively to the cutpoints (where the first is just a 
          positive real) to ensure that they remain in the same order--DONE
        * Change minimize() method to L-BFGS--DONE

        FIXME
        -----
        * Troubleshoot why constraining sigma leads to NaNs--DONE
        * Rewrite in TensorFlow
        """
        # Relevant parameters
        N, M = X.shape
        R = y.max()+1

        # Transform
        # * Adds ones feature column to X for bias weight
        X_transformed = self._transform(X)
        self.X = X
        self.y = y

        # Cutpoints
        # ordinal_outcomes = set(range(y.max()+1))
        # self.R = len(ordinal_outcomes)
        # cut_points_start = -self.padding*(self.R-2)/2
        # cut_points_end = -cut_points_start + self.padding
        # base_cut_points = ag_np.arange(
        #     cut_points_start, cut_points_end, self.padding)
        # self.cut_points = ag_np.hstack((-ag_np.inf, base_cut_points, ag_np.inf))

        # Initialized parameters

        # Variance - Noise variance initialized at variance=1
        # * r is unconstrained parameter to represent constrained variance at > 0
        # * Choose whether to learn variance as a parameter or not
        if fit_noise_variance is not None:
            init_r = softplus_inv(fit_noise_variance)
        else:
            if self.noise_variance is None:
                init_r = softplus_inv(1)
            else:
                init_r = softplus_inv(self.noise_variance)

        # Weights
        # TODO: can't remember how to initialize weights differently (if needed)
        init_w = self.rs.normal(size=M+1)

        # Cutpoints
        init_cut_points = ag_np.linspace(-3, 3, num=R-1)
        init_b1 = init_cut_points[0]
        init_epsilons = softplus_inv(ag_np.diff(init_cut_points))

        # Combine into a single ag_np.ndarray since scipy only accepts arrays
        init_params = ag_np.hstack((init_r, init_b1, init_epsilons, init_w))

        # Log
        # TODO: consider appending logs to list and printing all at once
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                if fit_noise_variance is not None:
                    message = f'Trained Variance = {fit_noise_variance}'
                    pad = '='
                    print(f'{message:{pad}^8}', file=f)
                print('INIT weights:', file=f)
                print(init_w, file=f)
                print('INIT cutpoints:', file=f)
                print(init_cut_points, file=f)
                print('INIT deltas:', file=f)
                print(ag_np.diff(init_cut_points), file=f)
                print('INIT epsilons:', file=f)
                print(init_epsilons, file=f)
                print('INIT PARAMS:', file=f)
                print(init_params, file=f)

        # MLE Estimate
        # Set up loss function
        def loss_function(params):
            # Log
            if self.log_file is not None:
                with open(self.log_file, 'a') as f:
                    print('PARAMS:', file=f)
                    print(params, file=f)
            # Variance
            # FIXME: for some reason, trying to learn sigma is leading to NaNs
            if fit_noise_variance is not None:
                variance = fit_noise_variance
            else:
                if self.noise_variance is None:
                    variance = softplus(params[0])
                else:
                    variance = self.noise_variance  # baseline where sigma doesn't change

            # Cutpoints
            deltas = softplus(params[2:R])  # 2+R-2 = R
            # print('deltas:')
            # print(deltas)
            # Use cumsum() to construct cutpoints from b1 and deltas
            b = ag_np.cumsum(ag_np.hstack((params[1], deltas)))
            # print('cutpoints:')
            # print(b)

            # Weights
            w = params[R:]

            # Return negative log-likelihood with complexity penalty (optional)
            return -self.log_likelihood(variance, w, b, X_transformed, y) + self.C * ag_np.sum(b**2)

        # Use scipy.minimize to find global minimum and return parameters
        # params = minimize(
        #     value_and_grad(loss_function),
        #     init_params,
        #     jac=True,
        #     method='L-BFGS-B',
        # ).x
        # a_gradient = grad(loss_function)
        # print('GRADIENT')
        # print(a_gradient(init_params))

        # Callback function to produce Neg log likelihood log file
        def callbackF(xk):
            # print(xk)
            with open(self.directory.joinpath('neg_log_likelihood.csv'), 'a') as f:
                print(f'{self.Nfeval},{loss_function(xk)/N}', file=f)
            self.Nfeval += 1

        # Use scipy.minimize to find global minimum and return parameters
        # Save negative log loss plot as csv and plot if desired
        if self.save_loss == True and fit_noise_variance is None:
            self.Nfeval = 1
            # Create log file
            with open(self.directory.joinpath('neg_log_likelihood.csv'), 'w') as f:
                print('Iter,Neg_Log_Likelihood_per_sample', file=f)
            params = minimize(
                fun=value_and_grad(loss_function),
                x0=init_params,
                jac=True,
                method='L-BFGS-B',
                callback=callbackF,
            ).x

            # Plot the neg log likelihood over time
            self._plot_log_likelihood()
        else:
            params = minimize(
                fun=value_and_grad(loss_function),
                x0=init_params,
                jac=True,
                method='L-BFGS-B',
            ).x
        self.noise_variance = softplus(params[0])
        deltas = softplus(params[2:R])  # 2+R-2 = R
        self.b = ag_np.cumsum(ag_np.hstack((params[1], deltas)))
        self.w = params[R:]

        # Log best values
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                print('BEST noise:')
                print(self.noise_variance)
                print('BEST deltas:')
                print(deltas)
                print('BEST cutpoints:')
                print(self.b)
                print('BEST weights:')
                print(self.w)

        return None

    def _transform(self, X: ag_np.ndarray) -> ag_np.ndarray:
        """Transform the data.

        Uses standard scaler to scale data and add ones column for the bias
        weight.

        Parameters
        ----------
        X : ag_np.ndarray
            Raw feature data

        Returns
        -------
        X_transformed : ag_np.ndarray
            Transformed feature data
        """
        # Add additional column of 1s for bias weight
        N = X.shape[0]
        X_transformed = ag_np.hstack((ag_np.ones((N, 1)), X))
        return X_transformed

    def _inverse_transform(self, X_transformed: ag_np.ndarray) -> ag_np.ndarray:
        """Reconstruct the original data.

        Parameters
        ----------
        X_transformed : ag_np.ndarray
            Transformed feature data

        Returns
        -------
        X : ag_np.ndarray
            Raw feature data
        """
        return X_transformed[:, 1:]

    def predict(self, X: ag_np.ndarray) -> ag_np.ndarray:
        """Predict ordinal outcomes.

        Parameters
        ----------
        X : ag_np.ndarray
            Raw feature data.

        Returns
        -------
        y_predict : ag_np.ndarray
            Predicted ordinal labels.
        """
        X_transformed = self._transform(X)
        best_log_proba_NR = self.log_proba(
            self.noise_variance, self.w, self.b, X_transformed)
        y_predict = ag_np.argmax(best_log_proba_NR, axis=1)
        return y_predict

    def log_likelihood(self,
                       variance: float,
                       w: ag_np.ndarray,
                       b: ag_np.ndarray,
                       X: ag_np.ndarray,
                       y: ag_np.ndarray
                       ) -> float:
        """Compute log-likelihood.

        Parameters
        ----------
        variance : float
                Noise variance
        w : ag_np.ndarray
            Latent function feature weights
        b : ag_np.ndarray
            Cutpoints
        X : ag_np.ndarray
            Raw feature data
        y : ag_np.ndarray
            Corresponding ordinal labels

        Returns
        -------
        log_likelihood : float
            Log likelihood given sample data.
        """
        # Useful parameters
        N = X.shape[0]

        # Log Likelihood
        # proba_NR = self.log_proba(variance, w, b, X)
        # log_likelihood_N = ag_np.log(proba_NR[ag_np.arange(N), y] + 1e-7)
        # log_proba_NR = self.log_proba(variance, w, b, X)
        log_proba_NR = ag_np.log(self.proba(variance, w, b, X))
        log_likelihood_N = log_proba_NR[ag_np.arange(N), y]
        # print('NEG LOG LIKELIHOOD:')
        # if self.save_loss == True:
        #     with open(self.directory.joinpath('neg_log_likelihood.csv'), 'a') as f:
        #         print(-ag_np.sum(log_likelihood_N)/N, file=f)
        return ag_np.sum(log_likelihood_N)

    def log_proba(self,
                  variance: float,
                  w: ag_np.ndarray,
                  b: ag_np.ndarray,
                  X: ag_np.ndarray,
                  ) -> ag_np.ndarray:
        """Compute log-probabilities of each ordinal outcome given a set of weights
        and cut-points.

        Vectorizes probability computation for improved. efficiency

        Parameters
        ----------
        variance : float
                Noise variance
        w : ag_np.ndarray
            Latent function feature weights
        b : ag_np.ndarray
            Cutpoints
        X : ag_np.ndarray
            Raw feature data

        Returns
        -------
        proba : ag_np.ndarray, shape: (NxD)
            Probabilities of being classified as each ordinal label

        TODO
        ----
        * Ask for help on how to compute stable log_proba--DONE
        * Broadcast the subtraction for the z values instead of tiling the latent function outputs
        ```python
        z_NRplus1 = (b - f_x_N[..., np.newaxis]) / ag_np.sqrt(variance)
        log_cdf_NRplus1 = ag_stats.norm.logcdf(z_NRplus1)
        log_proba_NR = ag_special.logsumexp(
            a=ag_np.stack((log_cdf_NRplus1[..., 1:], log_cdf_NRplus1[..., :-1]), axis=-1),
            axis=2,
            b=ag_np.array([1, -1]),
        )
        ```

        FIXME
        -----
        * Figure out why autograd/scipy is failing with the logsumexp()
            * Try changing b to be same dimension as a
            * Consider iterating and rebuilding array
            * Try stacking outside logsumexp()
            * Try running without b
        """
        # Include pos/neg Inf to cutpoints
        b = ag_np.hstack((-ag_np.inf, b, ag_np.inf))

        # Dimensions
        N = X.shape[0]
        R = b.size - 1

        # Compute latent function output and tile to create NR shape
        f_x_N = X @ w
        f_x_NR = ag_np.tile(f_x_N[:, ag_np.newaxis], (1, R))

        # Isolate cutpoints for z_1 and z_2 calcs
        b_1 = b[1:]
        b_2 = b[:-1]

        # Compute z_1 and z_2 for each data feature
        z_matrix_NR_1 = (b_1 - f_x_NR) / ag_np.sqrt(variance)
        z_matrix_NR_2 = (b_2 - f_x_NR) / ag_np.sqrt(variance)

        # Compute logcdf
        # cdf_NR_1 = ag_stats.norm.cdf(z_matrix_NR_1)
        # cdf_NR_2 = ag_stats.norm.cdf(z_matrix_NR_2)
        log_cdf_NR_1 = ag_stats.norm.logcdf(z_matrix_NR_1)
        log_cdf_NR_2 = ag_stats.norm.logcdf(z_matrix_NR_2)

        log_proba_NR = ag_special.logsumexp(
            a=ag_np.stack((log_cdf_NR_1, log_cdf_NR_2), axis=-1),
            axis=2,
            b=ag_np.array([1, -1]),
        )
        # log_proba_NR = ag_np.log(
        #     ag_np.exp(log_cdf_NR_1) - ag_np.exp(log_cdf_NR_1))

        # log_proba_NR = cdf_NR_1 - cdf_NR_2

        return log_proba_NR

    def proba(self,
              variance: float,
              w: ag_np.ndarray,
              b: ag_np.ndarray,
              X: ag_np.ndarray
              ) -> ag_np.ndarray:
        """Compute probabilities of each ordinal outcome given a set of weights
        and cut-points.

        Parameters
        ----------
        variance : float
                Noise variance
        w : ag_np.ndarray
            Latent function feature weights
        b : ag_np.ndarray
            Cutpoints
        X : ag_np.ndarray
            Raw feature data

        Returns
        -------
        proba : ag_np.ndarray, shape: (NxD)
            Probabilities of being classified as each ordinal label

        TODO
        ----
        * Call log_proba() to compute probability values
        """
        # Initialize values
        # Include pos/neg Inf to cutpoints
        b = ag_np.hstack((-ag_np.inf, b, ag_np.inf))

        # Dimensions
        N = X.shape[0]
        R = b.size - 1

        # Compute latent function output and tile to create NR shape
        # z_matrix_NR_1 = ag_np.zeros((N, 0))
        # z_matrix_NR_2 = ag_np.zeros((N, 0))
        f_x_N = X @ w
        f_x_NR = ag_np.tile(f_x_N[:, ag_np.newaxis], (1, R))

        # variance = ag_np.array(variance)
        # print('VARIANCE:')
        # print(variance)
        # sigma = ag_np.sqrt(ag_np.repeat(variance, N))

        b_1 = b[1:]
        b_2 = b[:-1]

        z_matrix_NR_1 = (b_1 - f_x_NR) / ag_np.sqrt(variance)
        z_matrix_NR_2 = (b_2 - f_x_NR) / ag_np.sqrt(variance)

        cdf_NR_1 = ag_stats.norm.cdf(z_matrix_NR_1)
        cdf_NR_2 = ag_stats.norm.cdf(z_matrix_NR_2)
        # log_cdf_NR_1 = ag_stats.norm.logcdf(z_matrix_NR_1)
        # log_cdf_NR_2 = ag_stats.norm.logcdf(z_matrix_NR_2)

        # log_proba_NR = log_cdf_NR_1 + \
        #     ag_np.log1p(-ag_np.exp(log_cdf_NR_2-log_cdf_NR_1))
        # log_proba_NR = ag_np.log(
        #     ag_np.exp(log_cdf_NR_1) - ag_np.exp(log_cdf_NR_1))

        proba_NR = cdf_NR_1 - cdf_NR_2

        return proba_NR + 1e-7

    def predict_proba(self, X: ag_np.ndarray) -> ag_np.ndarray:
        """Obtain the probabilities of each ordinal outcome given the best
        weights and cut-points.

        Parameters
        ----------
        X : ag_np.ndarray
            Raw feature data

        Returns
        -------
        best_proba : ag_np.ndarray
            _description_
        """
        X_transformed = self._transform(X)
        return self.proba(self.noise_variance, self.w, self.b, X_transformed)

    def predict_log_proba(self, X: ag_np.ndarray) -> ag_np.ndarray:
        """Obtain the log-probabilities of each ordinal outcome given the best
        weights and cut-points.

        Parameters
        ----------
        X : ag_np.ndarray
            Raw feature data

        Returns
        -------
        best_proba : ag_np.ndarray
            _description_
        """
        X_transformed = self._transform(X)
        return self.log_proba(self.noise_variance, self.w, self.b, X_transformed)

    def _plot_log_likelihood(self) -> None:
        """Helper method to plot the negative log-likelihood per sample over
        time.

        Might include complexity penalty in evaluation.

        Plots
        -----
        Log-Likelihood vs Time plot
            Exports plot to current working directory
        """
        # Read the csv containing the losses
        df_log_likelihood_per_sample = pd.read_csv(
            self.directory.joinpath('neg_log_likelihood.csv'))

        # Plot
        fig, ax = plt.subplots()
        ax.plot(df_log_likelihood_per_sample['Iter'],
                df_log_likelihood_per_sample['Neg_Log_Likelihood_per_sample'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Negative log-likelihood per sample')
        ax.set_title(
            'Ordinal Regression Log-Likelihood Change with Gradient Descent (L-BFGS-B)')
        ax.grid(True)
        plt.savefig(self.directory.joinpath('neg_log_likelihood.png'),
                    bbox_inches='tight', pad_inches=0)

    def grid_search_variance(self, out_dir: pathlib.Path = None, iteration: int = None) -> tuple[float, float]:
        """Given the current best parameters obtained from optimization,
        determines the best variance that minimizes loss.

        Performs a grid search over possible variances given trained parameters.

        Returns
        -------
        min_variance : float
            Variances that minimizes loss

        min_loss : float
            Minimum loss obtained from grid search
        """
        # Set output directory
        if out_dir is None:
            out_dir = self.directory

        # Training data
        X = self.X
        y = self.y
        X_transformed = self._transform(X)

        # Relevant parameters
        N, M = X.shape
        R = y.max()+1

        # Define a constrain and inverse constrain params functions
        best_b = self.b
        best_w = self.w

        b1 = best_b[0]
        deltas = ag_np.diff(best_b)
        # print(deltas)

        # Define loss function
        def loss_function(params):
            # Variance
            variance = softplus(params[0])

            # Cutpoints
            deltas = softplus(params[2:R])  # 2+R-2 = R
            # Use cumsum() to construct cutpoints from b1 and deltas
            b = ag_np.cumsum(ag_np.hstack((params[1], deltas)))

            # Weights
            w = params[R:]

            # Return negative log-likelihood with complexity penalty (optional)
            return -self.log_likelihood(variance, w, b, X_transformed, y) + self.C * ag_np.sum(b**2)

        # Choose grid to search over (various values for variance)
        variances = ag_np.logspace(-10, 2, 1000)
        # variances = ag_np.hstack((ag_np.linspace(0, 1, 1000), variances))
        # variances[0] = 0.01
        rs, epsilons = constrain_inv(variances, deltas)
        # print(rs)
        # print(epsilons)

        # Compute loss over each variance
        losses = []
        for i in range(variances.size):
            r = rs[i]
            params = ag_np.hstack((r, b1, epsilons, best_w))
            losses.append(loss_function(params))

        min_loss = ag_np.min(losses)
        min_variance = variances[ag_np.argmin(losses)]

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(variances, losses)
        ax.plot(min_variance, min_loss, 'rx')
        ax.annotate(
            f'Minimum loss = {min_loss:0.3f} @ variance = {min_variance:0.3f}',
            (min_variance, min_loss),
            xytext=(5, -10),
            textcoords='offset pixels',
        )
        ax.set_xlabel('Noise variance')
        ax.set_ylabel('Computed loss')
        ax.set_title(
            f'Loss at various noise variances\nTrained variance = {self.noise_variance:.3f}')
        ax.grid(True)

        filename = f'trained_variance.png'
        if iteration is not None:
            filename = f'{iteration:03d}_' + filename
        else:
            filename = f'{self.noise_variance:.3f}_' + filename
        plt.savefig(out_dir.joinpath('loss_variance', filename),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # plt.show()

        # Decision Boundary
        plot_model(self, export_path=out_dir.joinpath(
            'decision_boundary', filename))
        return min_variance, min_loss

    def find_global_minimum_variance(self, iter=50) -> float:
        """Attempts to find variance that minimizes loss globally by alternating
        between grid searching variance given trained parameters and retraining
        parameters with new variance.

        Assumes that at least one optimization has already occurred for the
        model (i.e. self.fit() has been called already).

        Parameters
        ----------
        iter : int, optional
            Number of iterations desired, by default 50

        Returns
        -------
        min_variance : float
            Best variances obtained from algorithm search
        """
        # Set up output directory
        out_dir = self.directory.joinpath('frames')
        out_dir.mkdir(exist_ok=False)
        out_dir.joinpath('loss_variance').mkdir(exist_ok=False)
        out_dir.joinpath('decision_boundary').mkdir(exist_ok=False)

        # Alternate grid searching and training
        for i in range(iter):
            min_variance, _ = self.grid_search_variance(
                out_dir=out_dir, iteration=i)
            self.fit(self.X, self.y, fit_noise_variance=min_variance)
            print(f'CUTPOINTS after iteration {i}:')
            print(self.b)

        # Generate animation to show loss curve and minimum changes
        frames = []
        for path in sorted(out_dir.joinpath('loss_variance').glob('*.png'), reverse=False):
            if path.is_file() and path.suffix == '.png':
                image = imageio.imread(path)
                frames.append(image)
        imageio.mimsave(out_dir.joinpath('loss_variance.gif'),  # output gif
                        frames,          # array of input frames
                        fps=5,           # optional: frames per second
                        )

        # Decision Boundary Animation
        frames = []
        for path in sorted(out_dir.joinpath('decision_boundary').glob('*.png'), reverse=False):
            if path.is_file() and path.suffix == '.png':
                image = imageio.imread(path)
                frames.append(image)
        imageio.mimsave(out_dir.joinpath('decision_boundary.gif'),  # output gif
                        frames,          # array of input frames
                        fps=5,           # optional: frames per second
                        )
        return min_variance


def softplus(x: ag_np.ndarray) -> ag_np.ndarray:
    """Implements the softplus activation function to constrain values.

    Broadcasts function to all elements in np.ndarray.

    Parameters
    ----------
    x : ag_np.ndarray
        Array-like to be transformed with softplus

    Returns
    -------
    ag_np.ndarray
        Softplus returned values in same shape as input array
    """
    # print('SOFTPLUS')
    # print(x)
    # print(ag_np.log(1 + ag_np.exp(x)))
    # print('X VALUES')
    # print(x)
    # print(x.shape)
    # mask1_N = x > 5
    # mask0_N = ag_np.logical_not(mask1_N)
    # out_N = ag_np.zeros(x.shape, dtype=ag_np.float64)
    # # print('MASK1')
    # # print(mask1_N)
    # # print(mask1_N.shape)
    # # print('MASK0')
    # # print(mask0_N)
    # # print(mask0_N.shape)
    # # print('OUT')
    # # print(out_N)
    # # print(out_N.shape)
    # out_N[mask0_N] = ag_np.log1p(ag_np.exp(x[mask0_N]))
    # out_N[mask1_N] = ag_np.log1p(ag_np.exp(-x[mask1_N])) + x[mask1_N]
    # ag_np.log1p(ag_np.exp(-ag_np.abs(x))) + ag_np.maximum(x, 0)
    # out_N = mask0_N * ag_np.log1p(ag_np.exp(x)) + mask1_N * (ag_np.log1p(ag_np.exp(-x))+x)
    return ag_np.log1p(ag_np.exp(-ag_np.abs(x))) + ag_np.maximum(x, 0)


################################################################################
# Module Functions
################################################################################

def softplus_inv(x: ag_np.ndarray) -> ag_np.ndarray:
    """Inverse softplus function.

    Recovers input values from softplus activation function.

    Parameters
    ----------
    x : ag_np.ndarray
        Array-like with values transformed by softplus

    Returns
    -------
    ag_np.ndarray
        Inverse softplus returned values in same shape as input array
    """
    # print('SOFTPLUS INVERSE')
    # print(x)
    # print(ag_np.log(ag_np.exp(x) - 1))
    return ag_np.log1p(-ag_np.exp(-x)) + x


def constrain(*omega_params) -> tuple[ag_np.array]:
    """Helper function to constrain multiple variables with softplus.

    Returns
    -------
    tuple[ag_np.array]
        Transformed variables with softplus
    """
    return tuple(map(softplus_inv, omega_params))


def constrain_inv(*params) -> tuple[ag_np.array]:
    """Helper function to recover multiple variables with inverse softplus.

    Returns
    -------
    tuple[ag_np.array]
        Inverse transformed variables with softplus
    """
    return tuple(map(softplus_inv, params))


def plot_model(model, export_path=None):
    """Function to plot the decision boundaries given a model.

    Parameters
    ----------
    model : _type_
        sklearn like classifier or regressor that outputs decision scores f(x) for any input x

    TODO
    ----
    * Generalize so that the function works with any number of ordinal labels--DONE (up until 5)
        * create a list of colors that are guaranteed to have the right color
          maps--DONE
        * Probably limited to the number of colormaps available. Assert that
          the function is not supported for more than N color maps
    * Assert that this is not supported for more than 2 dimensions
    * Figure out how to create own colormap so that the label limitation can be removed
    * Plot decision boundaries using contours
    """
    # Obtain Training data
    y = model.y
    x0 = model.X[:, 0]
    x1 = model.X[:, 1]

    # Set the limits on x/y-axis close to the data
    pad = 1
    x0_lims = [x0.min()-pad, x0.max()+pad]
    x1_lims = [x1.min()-pad, x1.max()+pad]

    # Set up bounds and resolution for probability shading
    grid_resolution = 1000
    eps = 1.0
    x0_min, x0_max = x0_lims[0] - eps, x0_lims[1] + eps
    x1_min, x1_max = x1_lims[0] - eps, x1_lims[1] + eps
    left, right = x0_min, x0_max
    bottom, top = x1_min, x1_max

    # Create a grid of 2-d points to plot the decision scores
    xx0, xx1 = ag_np.meshgrid(
        ag_np.linspace(x0_min, x0_max, grid_resolution),
        ag_np.linspace(x1_min, x1_max, grid_resolution),
    )

    # Flatten the grid
    X_grid = ag_np.c_[xx0.ravel(), xx1.ravel()]

    # Predict the scores on the grid of points
    p_grid = model.predict_proba(X_grid)

    # Set up the shade and marker colors
    shade_colors = [plt.cm.Blues, plt.cm.Oranges,
                    plt.cm.Greens, plt.cm.Reds, plt.cm.Purples]
    marker_colors = ['darkblue', 'darkorange',
                     'darkgreen', 'darkred', 'darkmagenta']

    # blue_colors = plt.cm.Blues(ag_np.linspace(0, 1, 201))
    # blue_colors[:, 3] = ag_np.linspace(0, 1, 201)
    # blue_cmap = matplotlib.colors.ListedColormap(blue_colors)
    # orange_colors = plt.cm.Oranges(ag_np.linspace(0, 1, 201))
    # orange_colors[:, 3] = ag_np.linspace(0, 1, 201)
    # orange_cmap = matplotlib.colors.ListedColormap(orange_colors)
    # green_colors = plt.cm.Greens(ag_np.linspace(0, 1, 201))
    # green_colors[:, 3] = ag_np.linspace(0, 1, 201)
    # green_cmap = matplotlib.colors.ListedColormap(green_colors)
    # red_colors = plt.cm.Reds(ag_np.linspace(0, 1, 201))
    # red_colors[:, 3] = ag_np.linspace(0, 1, 201)
    # red_cmap = matplotlib.colors.ListedColormap(red_colors)

    # Set the Greys colormap (for colorbar)
    grey_colors = plt.cm.Greys(ag_np.linspace(0, 1, 201))
    grey_colors[:, 3] = 0.4
    grey_cmap = matplotlib.colors.ListedColormap(grey_colors)

    # Set up the figure, axis, and colorbar location
    f, axs = plt.subplots(1, 1, figsize=(5, 5))
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    # cax0 = divider.append_axes('right', size='5%', pad=0.2)
    # cax1 = divider.append_axes('right', size='5%', pad=0.3)
    # cax2 = divider.append_axes('right', size='5%', pad=0.4)
    # cax3 = divider.append_axes('right', size='5%', pad=0.5)
    axs.set_xlim(x0_lims)
    axs.set_ylim(x1_lims)
    axs.grid(False)

    # Decision boundaries using contours

    # contour_values_to_plot = [0.0]
    # L = ag_np.maximum(len(contour_values_to_plot), 11)
    # level_colors = plt.cm.Greys(ag_np.linspace(0, 1, L))
    # m = L // 2
    # nrem = len(contour_values_to_plot)
    # mlow = m - nrem // 2
    # mhigh = m + nrem // 2 + 1
    # if mhigh - mlow < len(contour_values_to_plot):
    #     mhigh += 1
    # levels_gray_cmap = matplotlib.colors.ListedColormap(
    #     level_colors[mlow:mhigh])
    # axs.contour(
    #     xx0, xx1, p_grid[:, 0].reshape(xx0.shape),
    #     # levels=contour_values_to_plot,
    #     cmap=blue_cmap,
    #     vmin=-2, vmax=+2)
    # axs.contour(
    #     xx0, xx1, p_grid[:, 1].reshape(xx0.shape),
    #     # levels=contour_values_to_plot,
    #     cmap=orange_cmap,
    #     vmin=-2, vmax=+2)
    # axs.contour(
    #     xx0, xx1, p_grid[:, 2].reshape(xx0.shape),
    #     # levels=contour_values_to_plot,
    #     cmap=green_cmap,
    #     vmin=-2, vmax=+2)
    # axs.contour(
    #     xx0, xx1, p_grid[:, 3].reshape(xx0.shape),
    #     # levels=contour_values_to_plot,
    #     cmap=red_cmap,
    #     vmin=-2, vmax=+2)

    # Iterate through labels and plot data and probability maps
    for label in range(ag_np.max(y)+1):
        # Generate RGB-A values
        rgba_values = shade_colors[label](ag_np.linspace(0, 1, 201))
        # Make opacity values increasingly transparent as colors become lighter
        rgba_values[:, 3] = ag_np.linspace(0, 1, 201)
        # Create colormap object
        shade_cmap = matplotlib.colors.ListedColormap(rgba_values)

        # Plot training data markers and label colors
        axs.scatter(x0[y == label], x1[y == label],
                    marker='x',
                    linewidths=1,
                    color=marker_colors[label],
                    alpha=0.9,
                    label=f'y={label}',
                    )
        # Plot image of probability values for respective label
        axs.imshow(
            p_grid[:, [label]].reshape(xx0.shape),
            alpha=0.4, cmap=shade_cmap,
            interpolation='nearest',
            origin='lower',  # this is crucial
            extent=(left, right, bottom, top),
            vmin=0.0, vmax=1.0)
        # im1 = axs.imshow(
        #     p_grid[:, [1]].reshape(xx0.shape),
        #     alpha=0.4, cmap=orange_cmap,
        #     interpolation='nearest',
        #     origin='lower',  # this is crucial
        #     extent=(left, right, bottom, top),
        #     vmin=0.0, vmax=1.0)
        # im2 = axs.imshow(
        #     p_grid[:, [2]].reshape(xx0.shape),
        #     alpha=0.4, cmap=green_cmap,
        #     interpolation='nearest',
        #     origin='lower',  # this is crucial
        #     extent=(left, right, bottom, top),
        #     vmin=0.0, vmax=1.0)
        # im3 = axs.imshow(
        #     p_grid[:, [3]].reshape(xx0.shape),
        #     alpha=0.4, cmap=red_cmap,
        #     interpolation='nearest',
        #     origin='lower',  # this is crucial
        #     extent=(left, right, bottom, top),
        #     vmin=0.0, vmax=1.0)

    # Create the color bar -- Generalized to be grey
    cbar = plt.colorbar(plt.cm.ScalarMappable(
        cmap=grey_cmap), cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar0 = plt.colorbar(im0, cax=cax0, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar1 = plt.colorbar(im1, cax=cax1, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar2 = plt.colorbar(im2, cax=cax2, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar3 = plt.colorbar(im3, cax=cax3, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar0.draw_all()
    # cbar1.draw_all()
    # cbar2.draw_all()
    # cbar3.draw_all()

    # Include legend
    # handles, labels = axs.get_legend_handles_labels()
    # axs.legend(handles[::-1], labels[::-1],
    #            bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # axs.legend()
    if export_path is not None:
        plt.savefig(export_path,
                    bbox_inches='tight', pad_inches=0)
        plt.close(f)
    else:
        plt.show()
