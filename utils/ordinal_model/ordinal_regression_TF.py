#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ordinal_model.ordered_gaussian import OrderedGaussian, OrderedGaussian2

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


class OrdinalRegression:
    def __init__(self, random_state=10) -> None:
        if random_state is not None:
            tf.random.set_seed(random_state)

    def fit(self, X, y, use_gradient_tape=False, epochs=200, learning_rate=1e-3, dist='og2', verbose='auto'):
        """Fit the data to an orderinal model predictor using TensorFlow.

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        use_gradient_tape : bool, optional
            Whether to optimize using gradient tape (or automatically), by default False
        epochs : int, optional
            Number of desired epochs, by default 200
        learning_rate : _type_, optional
            Learning rate, by default 1e-3
        dist : str, optional
            Type of distribution desired to be used in distribution layer, by default 'og2'
        """
        # Training Set
        self.X = X
        self.y = y

        # Dimensions
        R = y.max() + 1

        # Define cutpoints variable
        # init = tf.sort(tf.random.uniform(shape=[R-1]))
        init_cutpoints = tf.range(R-1, dtype=np.float32) / (R-2)
        cutpoints = tf.Variable(initial_value=init_cutpoints, trainable=True)
        # scale = tf.Variable(initial_value=1.0,
        #                     trainable=True, dtype=tf.float32)

        # Set up model
        if dist == 'og':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(1),
                tfp.layers.DistributionLambda(
                    lambda t: OrderedGaussian(loc=t, cutpoints=cutpoints)),
            ])
        elif dist == 'og2':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(1),
                tfp.layers.DistributionLambda(
                    lambda t: OrderedGaussian2(loc=t, cutpoints=cutpoints)),
            ])
        elif dist == 'ol':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(1),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.OrderedLogistic(loc=t, cutpoints=cutpoints)),
            ])

        # Optimization
        if use_gradient_tape:
            self.losses = self.optimize(
                X, y, epochs=epochs, learning_rate=learning_rate)
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                               loss=OrdinalLoss(),
                               metrics=[])

            self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose)
            self.losses = [[epoch_loss]
                           for epoch_loss in self.history.history['loss']]

    def optimize(self, X, y, epochs=200, learning_rate=1e-3):
        """Optimize the loss function manually using TensorFlow's gradient tape.

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        epochs : int, optional
            _description_, by default 200
        learning_rate : _type_, optional
            _description_, by default 1e-3

        Returns
        -------
        losses
            List of lists of losses by epoch 
        """
        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # Prepare the training dataset.
        batch_size = X.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        train_dataset = train_dataset.shuffle(
            buffer_size=batch_size).batch(batch_size)

        # Prepare object to save losses
        losses = [[] for x in range(epochs)]

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    # Logits for this minibatch
                    logits = self.model(x_batch_train, training=True)
                    print('LOGITS:', logits, sep='\n')

                    # Compute the loss value for this minibatch.
                    loss_fn = OrdinalLoss()
                    loss_value = loss_fn(y_batch_train, logits)

                # Save loss
                losses[epoch].append(loss_value)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_weights))

                # Log every batch.
                if step % 1 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" %
                          ((step + 1) * batch_size))
        return losses

    def predict(self, X):
        """Predict the ordinal outcomes using the fitted model."""
        return self.model.predict(X)

    def plot_losses(self, flatten_epochs=False):
        """Visualize the loss plot."""
        if flatten_epochs:
            losses = [[num for sublist in self.losses for num in sublist]]
        else:
            losses = self.losses

        n_plots = len(losses)
        fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4))
        if flatten_epochs:
            axs.plot(losses[0])
            axs.set_xlabel('Epoch')
            axs.set_ylabel('Negative Log Likelihood per sample')
        else:
            for i, ax in enumerate(axs):
                ax.plot(losses[i])
                ax.set_title(f'Epoch = {i}')
                ax.set_xlabel('Step')
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.show()

################################################################################
# Loss function (negative log likelihood)
################################################################################


class OrdinalLoss(tf.keras.losses.Loss):
    """Ordinal Loss class"""
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ordinal_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)

################################################################################
# Decision Boundary Plot
################################################################################


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
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )

    # Flatten the grid
    X_grid = np.c_[xx0.ravel(), xx1.ravel()]

    # Predict the scores on the grid of points
    p_grid = model.model(X_grid).categorical_probs().numpy().squeeze(axis=1)

    # Set up the shade and marker colors
    shade_colors = [plt.cm.Blues, plt.cm.Oranges,
                    plt.cm.Greens, plt.cm.Reds, plt.cm.Purples]
    marker_colors = ['darkblue', 'darkorange',
                     'darkgreen', 'darkred', 'darkmagenta']

    # blue_colors = plt.cm.Blues(np.linspace(0, 1, 201))
    # blue_colors[:, 3] = np.linspace(0, 1, 201)
    # blue_cmap = matplotlib.colors.ListedColormap(blue_colors)
    # orange_colors = plt.cm.Oranges(np.linspace(0, 1, 201))
    # orange_colors[:, 3] = np.linspace(0, 1, 201)
    # orange_cmap = matplotlib.colors.ListedColormap(orange_colors)
    # green_colors = plt.cm.Greens(np.linspace(0, 1, 201))
    # green_colors[:, 3] = np.linspace(0, 1, 201)
    # green_cmap = matplotlib.colors.ListedColormap(green_colors)
    # red_colors = plt.cm.Reds(np.linspace(0, 1, 201))
    # red_colors[:, 3] = np.linspace(0, 1, 201)
    # red_cmap = matplotlib.colors.ListedColormap(red_colors)

    # Set the Greys colormap (for colorbar)
    grey_colors = plt.cm.Greys(np.linspace(0, 1, 201))
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

    # Iterate through labels and plot data and probability maps
    for label in range(np.max(y)+1):
        # Generate RGB-A values
        rgba_values = shade_colors[label](np.linspace(0, 1, 201))
        # Make opacity values increasingly transparent as colors become lighter
        rgba_values[:, 3] = np.linspace(0, 1, 201)
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
