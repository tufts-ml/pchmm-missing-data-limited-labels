#!/usr/bin/env python3
"""The ordered gaussian distribution class."""

################################################################################
# Imports
################################################################################

# import tensorflow.keras.backend as K
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability import layers as tfpl
from tensorflow_probability import math as tfp_math
from tensorflow_probability import util
from tensorflow_probability.python.internal import dtype_util, tensor_util, prefer_static
import tensorflow as tf
import warnings
# from ..util.util import as_tuple
# from tensorflow.keras.layers import Lambda, Dense
import numpy as np
# from ..third_party.convar import ConvolutionalAutoregressiveNetwork
import tensorflow as tf

from tensorflow_probability.python.distributions.ordered_logistic import _broadcast_cat_event_and_params

################################################################################
# Class
################################################################################


class OrderedGaussian(tfd.Distribution):

    def __init__(
        self,
        cutpoints,
        loc,
        scale=1.,
        dtype=tf.int32,
        validate_args=False,
        allow_nan_stats=True,
        name='OrderedGuassian',
    ):
        """Initialize Ordered Gaussian distributions.

        Args:
          cutpoints: A floating-point `Tensor` with shape `[B1, ..., Bb, K]` where
            `b >= 0` indicates the number of batch dimensions. Each entry is then a
            `K`-length vector of cutpoints. The vector of cutpoints should be
            non-decreasing, which is only checked if `validate_args=True`.
          loc: A floating-point `Tensor` with shape `[B1, ..., Bb]` where `b >=
            0` indicates the number of batch dimensions. The entries represent the
            mean(s) of the latent logistic distribution(s). Different batch shapes
            for `cutpoints` and `loc` are permitted, with the distribution
            `batch_shape` being `tf.shape(loc[..., tf.newaxis] -
            cutpoints)[:-1]` assuming the subtraction is a valid broadcasting
            operation.
          scale: Desired scale (stdev) of the gaussian noise applied on the latent function (default: 1)
          dtype: The type of the event samples (default: int32).
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
            (e.g. mode) use the value "`NaN`" to indicate the result is
            undefined. When `False`, an exception is raised if one or more of the
            statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.
        TODO
        ----
        * Update doctstring to be meaningful for gaussian
        """
        parameters = dict(locals())

        with tf.name_scope(name) as name:

            # Ordinal Gaussian Distributions specific parameters
            float_dtype = dtype_util.common_dtype(
                [cutpoints, loc, scale],
                dtype_hint=tf.float32,
            )
            self._cutpoints = tensor_util.convert_nonref_to_tensor(
                cutpoints, dtype_hint=float_dtype, name='cutpoints')
            self._loc = tensor_util.convert_nonref_to_tensor(
                loc, dtype_hint=float_dtype, name='loc')
            self._scale = tensor_util.convert_nonref_to_tensor(
                scale, dtype_hint=float_dtype, name='scale')

            super().__init__(
                dtype=dtype,
                reparameterization_type=tfd.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name
            )

    # @classmethod
    # def _parameter_properties(self, dtype, num_classes=None):
    #     # pylint: disable=g-long-lambda
    #     return dict(
    #         cutpoints=util.ParameterProperties(
    #             event_ndims=1,
    #             shape_fn=lambda sample_shape: tf.concat(
    #                 [sample_shape, [num_classes]], axis=0),
    #             default_constraining_bijector_fn=lambda: tfb.ascending.Ascending()),  # pylint:disable=unnecessary-lambda
    #         loc=util.ParameterProperties(),
    #         scale=util.ParameterProperties(),
    #     )
    #     # pylint: enable=g-long-lambda

    @classmethod
    def _params_event_ndims(cls):
        return dict(cutpoints=1, loc=0)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(('loc', 'scale'),
                ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

    @property
    def cutpoints(self):
        """Input argument `cutpoints`."""
        return self._cutpoints

    @property
    def loc(self):
        """Input argument `loc`."""
        return self._loc

    @property
    def scale(self):
        """Input argument `loc`."""
        return self._scale

    def ordinal_log_probs(self):
        """Log probabilities for the `K+1` ordered categories."""
        norm = tfd.Normal(loc=0, scale=self.scale)
        z_values = (self._augmented_cutpoints() -
                    self.loc[..., tf.newaxis]) / self.scale
        log_cdfs = norm.log_cdf(z_values)
        return tfp_math.log_sub_exp(log_cdfs[..., :-1], log_cdfs[..., 1:])

    def ordinal_probs(self):
        """Probabilities for the `K+1` ordered categories."""
        return tf.math.exp(self.ordinal_log_probs())

    def _log_prob(self, values):
        """Associated log probabilities of true output labels.

        Parameters
        ----------
        values : _type_
            Values of desired labels to predict
        """
        num_categories = self._num_categories()
        x_safe = tf.where((values > num_categories - 1)
                          | (values < 0), 0, values)
        log_probs = tfd.categorical.Categorical(
            logits=self.ordinal_log_probs()).log_prob(x_safe)
        inf = tf.constant(np.inf, dtype=log_probs.dtype)
        return tf.where((values > num_categories - 1) | (values < 0), -inf, log_probs)

    def _augmented_cutpoints(self):
        cutpoints = tf.convert_to_tensor(self.cutpoints)
        inf = tf.fill(
            cutpoints[..., :1].shape,
            tf.constant(np.inf, dtype=cutpoints.dtype))
        return tf.concat([-inf, cutpoints, inf], axis=-1)

    def _num_categories(self):
        return tf.shape(self.cutpoints, out_type=self.dtype)[-1] + 1

    def _sample_n(self, n, seed=None):
        return tfd.categorical.Categorical(
            logits=self.ordinal_log_probs()).sample(n, seed)

    def _batch_shape_tensor(self, cutpoints=None, loc=None):
        cutpoints = self.cutpoints if cutpoints is None else cutpoints
        loc = self.loc if loc is None else loc
        return prefer_static.broadcast_shape(
            prefer_static.shape(cutpoints)[:-1],
            prefer_static.shape(loc))

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self.loc.shape, self.cutpoints.shape[:-1])

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    # def _cdf(self, x):
    #     return tfd.categorical.Categorical(logits=self.ordinal_log_probs()).cdf(x)

    # def _entropy(self):
    #     return tfd.categorical.Categorical(
    #         logits=self.ordinal_log_probs()).entropy()

    # def _mode(self):
    #     log_probs = self.ordinal_log_probs()
    #     mode = tf.argmax(log_probs, axis=-1, output_type=self.dtype)
    #     tensorshape_util.set_shape(mode, log_probs.shape[:-1])
    #     return mode

    def _default_event_space_bijector(self):
        return

#     def _parameter_control_dependencies(self, is_init):
#         assertions = []

#         # In init, we can always build shape and dtype checks because
#         # we assume shape doesn't change for Variable backed args.
#         if is_init:

#             if not dtype_util.is_floating(self.cutpoints.dtype):
#                 raise TypeError(
#                     'Argument `cutpoints` must having floating type.')

#             if not dtype_util.is_floating(self.loc.dtype):
#                 raise TypeError('Argument `loc` must having floating type.')

#             cutpoint_dims = tensorshape_util.rank(self.cutpoints.shape)
#             msg = 'Argument `cutpoints` must have rank at least 1.'
#             if cutpoint_dims is not None:
#                 if cutpoint_dims < 1:
#                     raise ValueError(msg)
#             elif self.validate_args:
#                 cutpoints = tf.convert_to_tensor(self.cutpoints)
#                 assertions.append(
#                     assert_util.assert_rank_at_least(cutpoints, 1, message=msg))

#         if not self.validate_args:
#             return []

#         if is_init != tensor_util.is_ref(self.cutpoints):
#             cutpoints = tf.convert_to_tensor(self.cutpoints)
#             assertions.append(distribution_util.assert_nondecreasing(
#                 cutpoints, message='Argument `cutpoints` must be non-decreasing.'))

#         return assertions

#     def _sample_control_dependencies(self, x):
#         assertions = []
#         if not self.validate_args:
#             return assertions
#         assertions.append(distribution_util.assert_casting_closed(
#             x, target_dtype=tf.int32))
#         assertions.append(assert_util.assert_non_negative(x))
#         assertions.append(
#             assert_util.assert_less_equal(
#                 x, tf.cast(self._num_categories(), x.dtype),
#                 message=('OrderedLogistic samples must be `>= 0` and `<= K` '
#                          'where `K` is the number of cutpoints.')))
#         return assertions


# @kullback_leibler.RegisterKL(OrderedLogistic, OrderedLogistic)
# def _kl_ordered_logistic_ordered_logistic(a, b, name=None):
#     """Calculate the batched KL divergence KL(a || b), a and b OrderedLogistic.

#     This function utilises the `OrderedLogistic` `categorical_log_probs` member
#     function to implement KL divergence for discrete probability distributions as
#     described in
#     e.g. [Wikipedia](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence).

#     Args:
#       a: instance of a OrderedLogistic distribution object.
#       b: instance of a OrderedLogistic distribution object.
#       name: Python `str` name to use for created operations.
#         Default value: `None` (i.e., `'kl_ordered_logistic_ordered_logistic'`).

#     Returns:
#       Batchwise KL(a || b)
#     """
#     with tf.name_scope(name or 'kl_ordered_logistic_ordered_logistic'):
#         a_log_probs = a.categorical_log_probs()
#         b_log_probs = b.categorical_log_probs()
#         return tf.reduce_sum(
#             tf.math.multiply_no_nan(
#                 a_log_probs - b_log_probs, tf.math.exp(a_log_probs)),
#             axis=-1)

class OrderedGaussian2(tfd.OrderedLogistic):
    def __init__(
        self,
        cutpoints,
        loc,
        scale=1.,
        dtype=tf.int32,
        validate_args=False,
        allow_nan_stats=True,
        name='OrderedGuassian',
    ):
        with tf.name_scope(name) as name:

            self._scale = tensor_util.convert_nonref_to_tensor(
                scale, dtype_hint=tf.float32, name='scale')

            super().__init__(
                cutpoints=cutpoints,
                loc=loc,
                dtype=dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name
            )

    @property
    def scale(self):
        """Input argument `loc`."""
        return self._scale

    def categorical_log_probs(self):
        """Log probabilities for the `K+1` ordered categories."""
        norm = tfd.Normal(loc=0, scale=self.scale)
        z_values = (self._augmented_cutpoints() -
                    self.loc[..., tf.newaxis]) / self.scale
        log_cdfs = norm.log_cdf(z_values)
        # log_cdfs = norm.log_cdf(z_values)
        return tfp_math.log_sub_exp(log_cdfs[..., :-1], log_cdfs[..., 1:])

    # def _log_prob(self, values):
    #     """Associated log probabilities of true output labels.

    #     Parameters
    #     ----------
    #     values : _type_
    #         Values of desired labels to predict
    #     """
    #     num_categories = self._num_categories()
    #     x_safe = tf.where((values > num_categories - 1)
    #                       | (values < 0), 0, values)
    #     log_probs = tfd.categorical.Categorical(
    #         logits=self.categorical_log_probs()).log_prob(x_safe)
    #     inf = tf.constant(np.inf, dtype=log_probs.dtype)
    #     return tf.where((values > num_categories - 1) | (values < 0), -inf, log_probs)

    def _log_prob(self, x):
        # TODO(b/149334734): Consider using QuantizedDistribution for the log_prob
        # computation for better precision.
        eps = 1e-8
        num_categories = self._num_categories()

        z = (self._augmented_cutpoints() - self.loc[..., tf.newaxis]) / self.scale
        z = tf.where(tf.math.is_inf(z), tf.sign(z) * (1/eps), z) # replace inf/-inf with very large or very small value

        # from IPython import embed
        # embed()
        # tf.print(self.scale)
        # tf.print(
        #     tf.gather((self._augmented_cutpoints() - self.loc[..., tf.newaxis]) / self.scale,
        #               0)
        # )

        x, augmented_log_survival = _broadcast_cat_event_and_params(
            event=x,
            params=tfd.Normal(loc=0, scale=1).log_cdf(z),
            base_dtype=dtype_util.base_dtype(self.dtype))
        x_flat = tf.reshape(x, [-1, 1])
        augmented_log_survival_flat = tf.reshape(
            augmented_log_survival, [-1, num_categories + 1])
        log_survival_flat_xm1 = tf.gather(
            params=augmented_log_survival_flat,
            indices=tf.clip_by_value(x_flat, 0, num_categories),
            batch_dims=1)
        log_survival_flat_x = tf.gather(
            params=augmented_log_survival_flat,
            indices=tf.clip_by_value(x_flat + 1, 0, num_categories),
            batch_dims=1)
        log_prob_flat = tfp_math.log_sub_exp(
            log_survival_flat_xm1, log_survival_flat_x)
        # Deal with case where both survival probabilities are -inf, which gives
        # `log_prob_flat = nan` when it should be -inf.
        minus_inf = tf.constant(-np.inf, dtype=log_prob_flat.dtype)
        log_prob_flat = tf.where(
            x_flat > num_categories - 1, minus_inf, log_prob_flat)
        return tf.reshape(log_prob_flat, shape=tf.shape(x))
