# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from typing import Tuple

from lightonml.internal import utils
import numpy as np
from attr import attrs, attrib


def int_tuple(value) -> Tuple[int, ...]:
    """Return a tuple of ints whatever the input is"""
    if not isinstance(value, tuple):
        value = (value,)
    return tuple(int(elem) for elem in value)


@attrs(frozen=True)
class InputTraits:
    """Attributes of user input: number of features, and whether it's bit-packed"""
    n_features = attrib(type=Tuple[int, ...], converter=int_tuple)
    packed = attrib(type=bool)

    @property
    def n_features_s(self):
        """Scalar number of features"""
        return int(np.prod(self.n_features))


def numpy_arr(array):
    # we don't want to import tensor only for type-checking,
    # as it might not be present
    is_tensor = type(array).__name__ == 'Tensor'
    # np is guaranteed to be a numpy array
    np_arr = array.cpu().numpy() if is_tensor else np.asarray(array)
    return is_tensor, np_arr


# noinspection PyPep8Naming
class OpuUserInput:
    """This class handles the vectors fed by users of the OPU transform

    2 ways of creating it, one from existing traits, and one that guesses traits,
    used for "user input".
    """

    def __init__(self, X, _2d_batch: bool, input_traits: InputTraits,
                 n_samples: Tuple[int, ...]):
        self.is_tensor, X_np = numpy_arr(X)
        # If input array isn't C contiguous, will make a copy of it
        self.X = np.require(X_np, requirements=['C_CONTIGUOUS'])
        # Sanity checks
        if input_traits.packed and self.X.dtype != np.uint8:
            raise ValueError("Bit-packed array type must be uint8")
        self._traits = input_traits
        self._n_samples = n_samples
        self._2d_batch = _2d_batch

    @classmethod
    def from_traits(cls, array, traits: InputTraits):
        is_2d_features = len(traits.n_features) == 2
        packed = traits.packed
        # plain_2d means that the 2d features can be read from the last 2 dimensions
        plain_2d = not packed and is_2d_features
        _, np_arr = numpy_arr(array)
        _2d_batch, n_samples = cls.batch_size(np_arr, plain_2d)
        return cls(array, _2d_batch, traits, n_samples)

    @classmethod
    def from_input(cls, array, packed=False, is_2d_features=False, n_2d_features: tuple = None):
        if n_2d_features is not None:
            assert packed, "n_2d_features can only be specified with packed input"
        # packed_2d means user says is_2d_features, with packed input
        if (packed and is_2d_features) and n_2d_features is None:
            raise ValueError("When packed and 2D features, number of features"
                             " must be provided.")
        # plain_2d means that the 2d features can be read from the last 2 dimensions
        plain_2d = not packed and is_2d_features

        _, np_arr = numpy_arr(array)
        _2d_batch, n_samples = cls.batch_size(np_arr, plain_2d)

        # Determine number of features
        if is_2d_features:
            if not packed:
                n_features = np_arr.shape[-2:]
            else:
                n_features = n_2d_features
        else:
            n_features_ = np_arr.shape[-1]
            if packed:
                # If packed, number of features is 8 times the input's shape
                n_features_ *= 8
            # when 1d, it's still a tuple
            n_features = (n_features_, )
        traits = InputTraits(n_features, packed, )
        return cls(array, _2d_batch, traits, n_samples)

    @classmethod
    def batch_size(cls, np_arr, plain_2d):
        # Determine input shape
        if np_arr.ndim == 1:
            assert not plain_2d, "1D input can't be 2D features"
            _2d_batch = False
            n_samples = (1,)
        elif np_arr.ndim == 2:
            _2d_batch = False
            # if plain 2d features, it's a single 2d batch.
            # Else, it's N-size of 1d batch
            n_samples = (1,) if plain_2d else np_arr.shape[:1]
        elif np_arr.ndim == 3:
            # if plain 2d features, it's 1d batch of 2d vectors,
            # else it's a 2D batch of vectors (1d or 2d)
            _2d_batch = not plain_2d
            n_samples = np_arr.shape[:2] if _2d_batch else np_arr.shape[:1]
        elif np_arr.ndim == 4:
            _2d_batch = True
            assert plain_2d, "4D input must be 2D features, unpacked"
            n_samples = np_arr.shape[:2]
        else:
            raise ValueError("The input array np_arr.ndims should be 1, 2, 3 or 4")
        return _2d_batch, n_samples

    @property
    def traits(self):
        return self._traits

    @property
    def n_features(self) -> tuple:
        """Can be 1d our 2d"""
        return self._traits.n_features

    @property
    def n_features_s(self):
        """Scalar number of features"""
        return self._traits.n_features_s

    @property
    def n_samples(self) -> tuple:
        """Number of samples"""
        return self._n_samples

    @property
    def n_samples_s(self):
        return int(np.prod(self._n_samples))

    @property
    def is_batch(self):
        return self.n_samples_s > 1

    def __len__(self):
        """Returns scalar number of samples"""
        return self.n_samples_s

    def binary_check(self):
        """Raises exception if input isn't binary"""
        if not self._traits.packed:
            # Do the check over 1000 samples randomly chosen to avoid lightonml#61
            sampled_X = utils.random_samples(self.X, 1000)
            if not ((sampled_X == 0) | (sampled_X == 1)).all():
                raise ValueError('The input array should be binary - contain only 0s and 1s.')

    def reshape_input(self, raveled_features=False, leave_single_dim=False):
        """
        Returns input with shape suited for formatting and transform.

        2D samples are reshaped to 1D.
        2D features are kept as-is, unless raveled_features is True.
        Also, if it's a batch of a single vector (first dimension being one),
        returns a view without the first dimension, unless leave_single_dim is True
        """
        if self.is_batch or leave_single_dim:
            # When 2D batch, this will transform it into a 1D batch.
            sample_shape = (self.n_samples_s,)
        else:
            # Remove the single dimension when not batch, and not leave_single_dim
            sample_shape = ()

        if self._traits.packed:
            # when packed, just take the last dimension
            feature_shape = (self.X.shape[-1], )
        elif raveled_features:
            feature_shape = (self.n_features_s, )
        else:
            feature_shape = self.n_features

        # do the reshape, concatenating both tuples
        ret = self.X.reshape(sample_shape + feature_shape)
        ret.setflags(write=False)  # make sure it's read-only
        return ret

    def reshape_output(self, Y):
        """Do the necessary reshape of output vector if number of samples are 2D"""
        # return to a 3d shape by taking the first 2 dims of the initial shape
        if self._2d_batch:
            out_shape = self._n_samples + (-1,)
        else:
            out_shape = Y.shape

        # If X is a single vector with explicit first dimension to 1, add it back
        # to output shape
        if not self.is_batch and self.X.shape[0] == 1 and out_shape[0] != 1:
            out_shape = (1,) + out_shape

        return Y.reshape(out_shape)

    def unravel_features(self, X):
        """Return features unraveled, but keeping n_samples.
        Also if self is single, remove the first dimension.

        For use after encoding the batches in linear, before formatting+transform
        X has the shape of self.reshape_input(raveled_features=True, leave_single_dim=True)

        Since the batch encoding changes nb of samples, we expect them to be different that self,
         but number of features need to be the same
        """
        assert X.ndim == 2
        n_samples = X.shape[0]
        n_features = X.shape[1]
        assert n_features == self.n_features_s
        assert not self._traits.packed  # currently unsupported

        if n_samples > 1:
            sample_shape = (n_samples,)
        else:
            # Remove the single dimension when not batch
            sample_shape = ()
        feature_shape = self.n_features
        return X.reshape(sample_shape + feature_shape)

    def __str__(self):
        bit_pack_str = "bit-packed" if self._traits.packed else "not bit-packed"
        return "Input is a {} batch of {} features, {}."\
            .format(self.n_samples, self.n_features, bit_pack_str)
