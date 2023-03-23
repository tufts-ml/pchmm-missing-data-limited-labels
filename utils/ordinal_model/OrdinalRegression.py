#!/usr/bin/env python3
"""
Maximum-likelihood estimator for ordinal regression.

TODO
----
* [Done] Write additional method for using softplus link function: f(x) = ln(1+e^x)
* [Done] Include parameter on fit to show loss vs epochs
* Include parameter on fit to show traceplots
* [Done] Write plotting function for loss vs epochs
* Write plotting function for traceplots
* [Done] Remove scaling
    * Initialize weights accordingly
* Confirm with Preetish/Hughes what exactly needs to be plotted for the traceplots
    * [Done] Preetish suggested decision boundaries
* Vectorize the probability computation so that we do not have to loop--DONE
* Develop a log_proba method so that we do not have to worry about over/underflow
    * Need to discuss how to properly set up with Prof. Hughes
* Test on multi-dimensional (3+) data
* Test with kernel to improve separation boundaries

FIXME
-----
* Optimize cannot train on variance
    * Generate plot of loss at various variances by grid searching on variance
      once the rest of the parameters have been optimally found
      (hope that this will illuminate why the params are going to nan)--DONE
    * Alternate optimizing with constant variance and finding minimum loss variance
      using trained parameters. Generate gif plot to show learning.--DONE
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
# from autograd.scipy.stats import norm
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler

import imageio


class OrdinalRegression:
    """Class to fit and predict ordinal outcomes.
    """

    def __init__(self,
                 noise_variance: float = None,
                 C: float = 0,
                 save_loss: bool = False,
                 log_training: bool = False,
                 random_state: int = None,
                 ) -> None:
        """Constructor for OrdinalRegression class.

        Parameters
        ----------
        noise_variance : float, optional
            _description_, by default 1
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
        * Remove noise variance since this will be learnt by the model--Done
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
            Train the model using a specific noise variance

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
        * Troubleshoot why constraining sigma leads to NaNs
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
        X_transformed : _type_
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
        best_proba_NR = self.proba(
            self.noise_variance, self.w, self.b, X_transformed)
        y_predict = ag_np.argmax(best_proba_NR, axis=1)
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
        log_proba_NR = self.log_proba(variance, w, b, X)
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
                  X: ag_np.ndarray
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
        * Ask for help on how to compute stable log_proba
        """
        # Initialize values
        # print('SIGMA')
        # print(sigma)
        # sigma += 1e-3
        b = ag_np.hstack((-ag_np.inf, b, ag_np.inf))
        # print(b)
        N = X.shape[0]
        R = b.size - 1
        z_matrix_NR_1 = ag_np.zeros((N, 0))
        z_matrix_NR_2 = ag_np.zeros((N, 0))
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

        return ag_np.log(proba_NR + 1e-7)

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
        # print('SIGMA')
        # print(sigma)
        # sigma += 1e-3
        b = ag_np.hstack((-ag_np.inf, b, ag_np.inf))
        # print(b)
        N = X.shape[0]
        R = b.size - 1
        z_matrix_NR_1 = ag_np.zeros((N, 0))
        z_matrix_NR_2 = ag_np.zeros((N, 0))
        f_x = X @ w

        # variance = ag_np.array(variance)
        # print('VARIANCE:')
        # print(variance)
        # sigma = ag_np.sqrt(ag_np.repeat(variance, N))

        # Iterate through possible ordinal outcomes
        for j in range(R):
            # Note: Indexing is not supported
            # z_matrix_NR2[:, bi, 0] = (
            #     self.cut_points[bi+1] - (X @ w)
            # ) / ag_np.sqrt(self.noise_variance)
            # z_matrix_NR2[:, bi, 1] = (
            #     self.cut_points[bi] - (X @ w)
            # ) / ag_np.sqrt(self.noise_variance)
            z_bj_1 = ((b[j+1] - (f_x)) /
                      ag_np.sqrt(variance))[:, ag_np.newaxis]
            z_bj_2 = ((b[j] - (f_x)) / ag_np.sqrt(variance))[:, ag_np.newaxis]
            # if z_matrix_NR_1 is None and z_matrix_NR_2 is None:
            #     z_matrix_NR_1 = z_bi_1
            #     z_matrix_NR_2 = z_bi_2
            # else:
            # print('SHAPES')
            # print(z_matrix_NR_1.shape)
            # print(z_bj_1.shape)
            # print(z_matrix_NR_2.shape)
            # print(z_bj_2.shape)
            z_matrix_NR_1 = ag_np.hstack((z_matrix_NR_1, z_bj_1))
            z_matrix_NR_2 = ag_np.hstack((z_matrix_NR_2, z_bj_2))
        z_matrix_NR2 = ag_np.concatenate(
            (z_matrix_NR_1[:, :, ag_np.newaxis], z_matrix_NR_2[:, :, ag_np.newaxis]), axis=2)
        gaussian_cdf_NR2 = ag_stats.norm.cdf(z_matrix_NR2)
        proba_NR = gaussian_cdf_NR2[:, :, 0] - gaussian_cdf_NR2[:, :, 1]
        return proba_NR

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

    def _plot_log_likelihood(self) -> None:
        """Helper method to plot the negative log-likelihood per sample over
        time.

        Might include complexity penalty in evaluation.
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
        """_summary_

        Returns
        -------
        tuple[float, float]
            _description_
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

        filename = f'{self.noise_variance:.3f}_trained_variance.png'
        if iteration is not None:
            filename = f'{iteration:03d}_' + filename
        plt.savefig(out_dir.joinpath(filename),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # plt.show()
        return min_variance, min_loss

    def find_global_minimum_variance(self, iter=50) -> float:
        # Set up output directory
        out_dir = self.directory.joinpath('loss_variance_frames')
        out_dir.mkdir(exist_ok=False)

        for i in range(iter):
            min_variance, _ = self.grid_search_variance(
                out_dir=out_dir, iteration=i)
            self.fit(self.X, self.y, fit_noise_variance=min_variance)

        frames = []
        for path in sorted(out_dir.glob('*.png'), reverse=False):
            if path.is_file() and path.suffix == '.png':
                image = imageio.imread(path)
                frames.append(image)
        imageio.mimsave(out_dir.joinpath('loss_variance.gif'),  # output gif
                        frames,          # array of input frames
                        fps=5)         # optional: frames per second
        return min_variance


def softplus(x) -> ag_np.ndarray:
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


def softplus_inv(x) -> ag_np.ndarray:
    # print('SOFTPLUS INVERSE')
    # print(x)
    # print(ag_np.log(ag_np.exp(x) - 1))
    return ag_np.log1p(-ag_np.exp(-x)) + x


def constrain(*omega_params):
    return list(map(softplus_inv, omega_params))


def constrain_inv(*params):
    return list(map(softplus_inv, params))


def plot_model(model):
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

    plt.show()
