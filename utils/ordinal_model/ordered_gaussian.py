#!/usr/bin/env python3
"""The ordered gaussian distribution class."""

################################################################################
# distributions.py
################################################################################

# import tensorflow.keras.backend as K
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability import layers as tfpl
from tensorflow_probability import math
import tensorflow as tf
import warnings
# from ..util.util import as_tuple
# from tensorflow.keras.layers import Lambda, Dense
import numpy as np
# from ..third_party.convar import ConvolutionalAutoregressiveNetwork
import tensorflow as tf

################################################################################
# ordered_logistic.py
################################################################################

# import numpy as np
# import tensorflow.compat.v2 as tf

# from tensorflow_probability.python.bijectors import ascending
# from tensorflow_probability.python.distributions import categorical
# from tensorflow_probability.python.distributions import distribution
# from tensorflow_probability.python.distributions import kullback_leibler
# from tensorflow_probability.python.internal import assert_util
# from tensorflow_probability.python.internal import distribution_util
# from tensorflow_probability.python.internal import dtype_util
# from tensorflow_probability.python.internal import parameter_properties
# from tensorflow_probability.python.internal import prefer_static as ps
# from tensorflow_probability.python.internal import reparameterization
# from tensorflow_probability.python.internal import tensor_util
# from tensorflow_probability.python.internal import tensorshape_util
# from tensorflow_probability.python.math import generic


class OrderedGaussian(tfd.Distribution):

    def __init__(
        self,
        cutpoints,
        loc,
        scale=1,
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
          scale: TODO (default: 1)
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

        # Ordinal Gaussian Distributions specific parameters
        self.cutpoints = cutpoints
        self.loc = loc
        self.scale = scale

        # super().__init__(
        #     dtype=dtype,
        #     reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        #     validate_args=validate_args,
        #     allow_nan_stats=allow_nan_stats,
        #     parameters=parameters,
        #     name=name)

    # @classmethod
    # def _parameter_properties(cls, dtype, num_classes=None):
    #     # pylint: disable=g-long-lambda
    #     return dict(
    #         cutpoints=parameter_properties.ParameterProperties(
    #             event_ndims=1,
    #             shape_fn=lambda sample_shape: ps.concat(
    #                 [sample_shape, [num_classes]], axis=0),
    #             default_constraining_bijector_fn=lambda: ascending.Ascending()),  # pylint:disable=unnecessary-lambda
    #         loc=parameter_properties.ParameterProperties())
    #     # pylint: enable=g-long-lambda

    # @property
    # def cutpoints(self):
    #     """Input argument `cutpoints`."""
    #     return self._cutpoints

    # @property
    # def loc(self):
    #     """Input argument `loc`."""
    #     return self._loc

    def log_probs(self):
        """Log probabilities for the `K+1` ordered categories."""
        norm = tfd.Normal(loc=0, scale=self.scale)
        log_cdfs = norm.log_cdf(
            self._augmented_cutpoints() - self.loc[..., tf.newaxis]
        )
        return math.log_sub_exp(log_cdfs[..., :-1], log_cdfs[..., 1:])

    def probs(self):
        """Probabilities for the `K+1` ordered categories."""
        return tf.math.exp(self.log_probs())

    def _augmented_cutpoints(self):
        cutpoints = tf.convert_to_tensor(self.cutpoints)
        inf = tf.constant(np.inf, dtype=cutpoints.dtype)
        return tf.concat([-inf, cutpoints, inf], axis=-1)

    def _num_categories(self):
        return tf.shape(self.cutpoints, out_type=self.dtype)[-1] + 1

#     def _sample_n(self, n, seed=None):
#         return categorical.Categorical(
#             logits=self.categorical_log_probs()).sample(n, seed)

#     def _event_shape_tensor(self):
#         return tf.constant([], dtype=tf.int32)

#     def _event_shape(self):
#         return tf.TensorShape([])

#     def _log_prob(self, x):
#         num_categories = self._num_categories()
#         x_safe = tf.where((x > num_categories - 1) | (x < 0), 0, x)
#         log_probs = categorical.Categorical(
#             logits=self.categorical_log_probs()).log_prob(x_safe)
#         neg_inf = dtype_util.as_numpy_dtype(log_probs.dtype)(-np.inf)
#         return tf.where((x > num_categories - 1) | (x < 0), neg_inf, log_probs)

#     def _cdf(self, x):
#         return categorical.Categorical(logits=self.categorical_log_probs()).cdf(x)

#     def _entropy(self):
#         return categorical.Categorical(
#             logits=self.categorical_log_probs()).entropy()

#     def _mode(self):
#         log_probs = self.categorical_log_probs()
#         mode = tf.argmax(log_probs, axis=-1, output_type=self.dtype)
#         tensorshape_util.set_shape(mode, log_probs.shape[:-1])
#         return mode

#     def _default_event_space_bijector(self):
#         return

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
