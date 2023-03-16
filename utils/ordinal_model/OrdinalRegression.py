#!/usr/bin/env python3
"""
Maximum-likelihood estimator for ordinal regression.

Usage
-----
TODO
----
* Write additional method for using softplus link function: f(x) = ln(1+e^x) -- Done
* Include parameter on fit to show loss vs epochs--Done
* Include paramter on fit to show traceplots
* Write plotting function for loss vs epochs
* Write plotting function for traceplots
* Remove scaling -- Done
    * Initialize weights accordingly
* Confirm with Preetish/Hughes what exactly needs to be plotted for the traceplots
    * Preetish suggested decision boundries
"""
import pathlib

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
# import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.scipy.stats import norm
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler

from ordinal_model.ordinal_log_likelihood import log_likelihood, proba


class OrdinalRegression:
    """Class to fit and predict ordinal outcomes.
    """

    def __init__(self, noise_variance: float = None, C: float = 0, save_loss: bool = False, random_state: int = None) -> None:
        """Constructer for OrdinalRegression class.

        Parameters
        ----------
        noise_variance : float, optional
            _description_, by default 1
        C : float, optional
            Regularization strength of cutpoints
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
        self.rs = np.random.RandomState(random_state)

        # File path(s)
        # self.directory = pathlib.Path(__file__).parent
        self.directory = pathlib.Path.cwd()

        # Logging
        self.save_loss = save_loss
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model using the training data and ordinal labels.

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
        * What should the parameters for scipy.minimize be?--Done
            * Prof. Hughes suggested to switch back to value_and_grad--Done
        * Should be eventually rewritten in tensorflow/pytorch
        * Rewrite paramaters to be dummy values that get passed through
          softplus in the learning function so that we ensure positive real--DONE
        * Add epsilons additively to the cutpoints (where the first is just a 
          positive real) to ensure that they remain in the same order--DONE
        * Change minimize() method to L-BFGS--DONE
        * Troubleshoot why constraining sigma leads to NaNs
        """
        # Relevant parameters
        N, M = X.shape
        R = y.max()+1

        # Transform
        # * Adds bias feature to X
        X_transformed = self._transform(X)
        self.X = X
        self.y = y

        # Cutpoints
        # ordinal_outcomes = set(range(y.max()+1))
        # self.R = len(ordinal_outcomes)
        # cut_points_start = -self.padding*(self.R-2)/2
        # cut_points_end = -cut_points_start + self.padding
        # base_cut_points = np.arange(
        #     cut_points_start, cut_points_end, self.padding)
        # self.cut_points = np.hstack((-np.inf, base_cut_points, np.inf))

        # Initialized parameters
        # Variance - Noise variance initialized at variance=1
        # * r is unconstrained parameter to represent constrained variance at > 0
        # * Choose whether to learn variance as a parameter or not
        if self.noise_variance is None:
            init_r = softplus_inv(1)
        else:
            init_r = softplus_inv(self.noise_variance)

        # Weights
        # TODO: can't remember how to initialize weights differently (if needed)
        init_w = self.rs.normal(size=M+1)
        print('INIT weights:')
        print(init_w)

        # Cutpoints
        init_cut_points = np.linspace(-3, 3, num=R-1)
        print('INIT cutpoints:')
        print(init_cut_points)
        init_b1 = init_cut_points[0]
        init_epsilons = softplus_inv(np.diff(init_cut_points))
        print('INIT deltas:')
        print(np.diff(init_cut_points))
        print('INIT epsilons:')
        print(init_epsilons)

        # Combine into a single np.ndarray since scipy only accepts arrays
        init_params = np.hstack((init_r, init_b1, init_epsilons, init_w))
        print('INIT PARAMS:')
        print(init_params)

        # MLE Estimate
        # Set up loss function
        def loss_function(params):
            print('PARAMS:')
            print(params)
            # Variance
            # TODO: for some reason, trying to learn sigma is leading to NaNs
            if self.noise_variance is None:
                variance = softplus(np.array(params[0])[np.newaxis])
            else:
                variance = self.noise_variance  # baseline where sigma doesn't change

            # Cutpoints
            deltas = softplus(params[2:R])  # 2+R-2 = R
            # print('deltas:')
            # print(deltas)
            # Use cumsum() to construct cutpoints from b1 and deltas
            b = np.cumsum(np.hstack((params[1], deltas)))
            # print('cutpoints:')
            # print(b)

            # Weights
            w = params[R:]

            # Return negative log-likelihood with complexity penalty (optional)
            return -self.log_likelihood(variance, w, b, X_transformed, y) + self.C * np.sum(b**2)

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
        if self.save_loss == True:
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
        print('best noise:')
        print(self.noise_variance)
        deltas = softplus(params[2:R])  # 2+R-2 = R
        print('best deltas:')
        print(deltas)
        self.b = np.cumsum(np.hstack((params[1], deltas)))
        print('best cutpoints:')
        print(self.b)
        self.w = params[R:]
        print('best weights:')
        print(self.w)
        return None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.

        Uses standard scaler to scale data and add ones column for the bias
        weight.

        Parameters
        ----------
        X : np.ndarray
            Raw feature data

        Returns
        -------
        X_transformed : np.ndarray
            Transformed feature data
        """
        # Add additional column of 1s for bias weight
        N = X.shape[0]
        X_transformed = np.hstack((np.ones((N, 1)), X))
        return X_transformed

    def _inverse_transform(self, X_transformed) -> np.ndarray:
        """Reconstruct the original data.

        Parameters
        ----------
        X_transformed : _type_
            Transformed feature data

        Returns
        -------
        X : np.ndarray
            Raw feature data
        """
        return X_transformed[:, 1:]

    def predict(self, X) -> np.ndarray:
        """Predict ordinal outcomes.

        Parameters
        ----------
        X : _type_
            Raw feature data.

        Returns
        -------
        y_predict : np.ndarray
            Predicted ordinal labels.
        """
        X_transformed = self._transform(X)
        best_proba_NR = self.proba(
            self.noise_variance, self.w, self.b, X_transformed)
        y_predict = np.argmax(best_proba_NR, axis=1)
        return y_predict

    def log_likelihood(self, variance, w, b, X, y) -> float:
        """Compute log-likelihood.

        Parameters
        ----------
        variance : _type_
                Noise variance
        w : _type_
            Latent function feature weights
        b : _type_
            Cutpoints
        X : _type_
            Raw feature data
        y : _type_
            Corresponding ordinal labels

        Returns
        -------
        log_likelihood : float
            Log likelihood given sample data.
        """
        # Useful parameters
        N = X.shape[0]

        # Log Likelihood
        proba_NR = self.proba(variance, w, b, X)
        log_likelihood_N = np.log(proba_NR[np.arange(N), y] + 1e-7)
        # print('NEG LOG LIKELIHOOD:')
        # if self.save_loss == True:
        #     with open(self.directory.joinpath('neg_log_likelihood.csv'), 'a') as f:
        #         print(-np.sum(log_likelihood_N)/N, file=f)
        return np.sum(log_likelihood_N)

    def proba(self, variance, w, b, X) -> np.ndarray:
        """Compute probabilities of each ordinal outcome given a set of weights
        and cut-points.

        Parameters
        ----------
        variance : _type_
                Noise variance
        w : _type_
            Latent function feature weights
        b : _type_
            Cutpoints
        X : _type_
            Raw feature data

        Returns
        -------
        proba : np.ndarray, shape: (NxD)
            Probabilities of being classified as each ordinal label
        """
        # Initialize values
        # print('SIGMA')
        # print(sigma)
        # sigma += 1e-3
        b = np.hstack((-np.inf, b, np.inf))
        # print(b)
        N = X.shape[0]
        R = b.size - 1
        z_matrix_NR_1 = np.zeros((N, 0))
        z_matrix_NR_2 = np.zeros((N, 0))
        f_x = X @ w

        # variance = np.array(variance)
        print('VARIANCE:')
        print(variance)
        # sigma = np.sqrt(np.repeat(variance, N))

        # Iterate through possible ordinal outcomes
        for j in range(R):
            # Note: Indexing is not supported
            # z_matrix_NR2[:, bi, 0] = (
            #     self.cut_points[bi+1] - (X @ w)
            # ) / np.sqrt(self.noise_variance)
            # z_matrix_NR2[:, bi, 1] = (
            #     self.cut_points[bi] - (X @ w)
            # ) / np.sqrt(self.noise_variance)
            z_bj_1 = ((b[j+1] - (f_x)) / np.sqrt(variance))[:, np.newaxis]
            z_bj_2 = ((b[j] - (f_x)) / np.sqrt(variance))[:, np.newaxis]
            # if z_matrix_NR_1 is None and z_matrix_NR_2 is None:
            #     z_matrix_NR_1 = z_bi_1
            #     z_matrix_NR_2 = z_bi_2
            # else:
            # print('SHAPES')
            # print(z_matrix_NR_1.shape)
            # print(z_bj_1.shape)
            # print(z_matrix_NR_2.shape)
            # print(z_bj_2.shape)
            z_matrix_NR_1 = np.hstack((z_matrix_NR_1, z_bj_1))
            z_matrix_NR_2 = np.hstack((z_matrix_NR_2, z_bj_2))
        z_matrix_NR2 = np.concatenate(
            (z_matrix_NR_1[:, :, np.newaxis], z_matrix_NR_2[:, :, np.newaxis]), axis=2)
        gaussian_cdf_NR2 = norm.cdf(z_matrix_NR2)
        proba_NR = gaussian_cdf_NR2[:, :, 0] - gaussian_cdf_NR2[:, :, 1]
        return proba_NR

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Obtain the probabilities of each ordinal outcome given the best
        weights and cut-points.

        Parameters
        ----------
        X : np.ndarray
            Raw feature data

        Returns
        -------
        best_proba : np.ndarray
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


def softplus(x) -> np.ndarray:
    # print('SOFTPLUS')
    # print(x)
    # print(np.log(1 + np.exp(x)))
    # print('X VALUES')
    # print(x)
    # print(x.shape)
    # mask1_N = x > 5
    # mask0_N = np.logical_not(mask1_N)
    # out_N = np.zeros(x.shape, dtype=np.float64)
    # # print('MASK1')
    # # print(mask1_N)
    # # print(mask1_N.shape)
    # # print('MASK0')
    # # print(mask0_N)
    # # print(mask0_N.shape)
    # # print('OUT')
    # # print(out_N)
    # # print(out_N.shape)
    # out_N[mask0_N] = np.log1p(np.exp(x[mask0_N]))
    # out_N[mask1_N] = np.log1p(np.exp(-x[mask1_N])) + x[mask1_N]
    # np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    # out_N = mask0_N * np.log1p(np.exp(x)) + mask1_N * (np.log1p(np.exp(-x))+x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_inv(x) -> np.ndarray:
    # print('SOFTPLUS INVERSE')
    # print(x)
    # print(np.log(np.exp(x) - 1))
    return np.log1p(-np.exp(-x)) + x


def plot_model(model):
    ''' Function to plot the decision boundaries given a model
    model : sklearn like classifier or regressor that outputs decision scores f(x) for any input x
    '''
    y = model.y
    x0 = model.X[:, 0]
    x1 = model.X[:, 1]
    # set the limits on x-axis close to the data
    x0_lims = [x0.min()-1, x0.max()+1]
    # set the limits on y-axis close to the data
    x1_lims = [x1.min()-1, x1.max()+1]
    grid_resolution = 1000
    eps = 1.0
    x0_min, x0_max = x0_lims[0] - eps, x0_lims[1] + eps
    x1_min, x1_max = x1_lims[0] - eps, x1_lims[1] + eps

    # create a grid of 2-d points to plot the decision scores
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )
    print(xx0.shape)
    print(xx1.shape)

    # flatten the grid
    X_grid = np.c_[xx0.ravel(), xx1.ravel()]
    # print(X_grid)

    # predict the scores on the grid of points
    p_grid = model.predict_proba(X_grid)

    # set the colormap
    blue_colors = plt.cm.Blues(np.linspace(0, 1, 201))
    blue_colors[:, 3] = np.linspace(0, 1, 201)
    blue_cmap = matplotlib.colors.ListedColormap(blue_colors)
    orange_colors = plt.cm.Oranges(np.linspace(0, 1, 201))
    orange_colors[:, 3] = np.linspace(0, 1, 201)
    orange_cmap = matplotlib.colors.ListedColormap(orange_colors)
    green_colors = plt.cm.Greens(np.linspace(0, 1, 201))
    green_colors[:, 3] = np.linspace(0, 1, 201)
    green_cmap = matplotlib.colors.ListedColormap(green_colors)
    red_colors = plt.cm.Reds(np.linspace(0, 1, 201))
    red_colors[:, 3] = np.linspace(0, 1, 201)
    red_cmap = matplotlib.colors.ListedColormap(red_colors)

    grey_colors = plt.cm.Greys(np.linspace(0, 1, 201))
    grey_colors[:, 3] = 0.4
    grey_cmap = matplotlib.colors.ListedColormap(grey_colors)

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
    # contour_values_to_plot = [0.0]
    # L = np.maximum(len(contour_values_to_plot), 11)
    # level_colors = plt.cm.Greys(np.linspace(0, 1, L))
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

    # plot the data as scatter plot

    axs.scatter(x0[y == 0], x1[y == 0],
                marker='x', linewidths=1, color='darkblue', alpha=0.9)
    axs.scatter(x0[y == 1], x1[y == 1],
                marker='x', linewidths=1, color='darkorange', alpha=0.9)
    axs.scatter(x0[y == 2], x1[y == 2],
                marker='x', linewidths=1, color='darkgreen', alpha=0.9)
    axs.scatter(x0[y == 3], x1[y == 3],
                marker='x', linewidths=1, color='darkred', alpha=0.9)
    # axs.plot(x_pos_ND[:, 0], x_pos_ND[:, 1], 'r+', alpha=0.6)
    # axs.set_xticks([-3, 0, 3])
    # axs.set_yticks([-3, 0, 3])

    left, right = x0_min, x0_max
    bottom, top = x1_min, x1_max
    im0 = axs.imshow(
        p_grid[:, [0]].reshape(xx0.shape),
        alpha=0.4, cmap=blue_cmap,
        interpolation='nearest',
        origin='lower',  # this is crucial
        extent=(left, right, bottom, top),
        vmin=0.0, vmax=1.0)
    im1 = axs.imshow(
        p_grid[:, [1]].reshape(xx0.shape),
        alpha=0.4, cmap=orange_cmap,
        interpolation='nearest',
        origin='lower',  # this is crucial
        extent=(left, right, bottom, top),
        vmin=0.0, vmax=1.0)
    im2 = axs.imshow(
        p_grid[:, [2]].reshape(xx0.shape),
        alpha=0.4, cmap=green_cmap,
        interpolation='nearest',
        origin='lower',  # this is crucial
        extent=(left, right, bottom, top),
        vmin=0.0, vmax=1.0)
    im3 = axs.imshow(
        p_grid[:, [3]].reshape(xx0.shape),
        alpha=0.4, cmap=red_cmap,
        interpolation='nearest',
        origin='lower',  # this is crucial
        extent=(left, right, bottom, top),
        vmin=0.0, vmax=1.0)
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

    plt.show()
