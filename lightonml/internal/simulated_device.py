# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import warnings
from contextlib import contextmanager

from lightonml.internal.types import OutputRoiStrategy, Tuple2D
import numpy as np


# noinspection PyPep8Naming
class SimulatedOpuDevice(object):
    """
    Provides a simulation of the OPU with synthetic random matrix

    Parameters
    ----------

    frametime_us: int
    exposure_us: int
    verbose: bool
        These parameters are provided only for API compatibility with
        lightonml.internal.device.OpuDevice
    linear: bool
        Whether the OPU simulates a linear operation or not

    Attributes
    ----------
    _random_matrix : np.ndarray
        The random matrix used for transformation
        @see build_random_matrix to generate it
    """

    def __init__(self, frametime_us=500, exposure_us=400,
                 verbose=False, linear=False):
        self.active = False
        self.frametime_us = int(frametime_us)
        self.exposure_us = int(exposure_us)
        self.output_roi = (0, 0), (2040, 1088)
        self._verbose = verbose
        self._random_matrix = None
        self._seed = None
        self._linear = linear

    def build_random_matrix(self, n_features, n_components, seed=0):
        """
        Generate the random matrix that simulates the OPU.

        The matrix is complex, drawning elements at random from a
        normal gaussian distribution
        @see self.random_matrix

        Parameters
        ----------
        n_features: int
            number of features of the generated matrix
        n_components: int
            number of components of the generated matrix
        seed : {None, int, array_like}, optional
            Initializer for the pseudo random number generator of the matrix
            Can be any integer between 0 and 2**32 - 1 inclusive,
            an array (or other sequence) of such integers, or None.
            If seed is None, then RandomState will try to read data from
            /dev/urandom if available or seed from the clock otherwise.
        """

        rng = np.random.RandomState(seed)
        std = 1. / np.sqrt(n_features)
        matrix_shape = (n_features, n_components)
        real_comp = rng.normal(loc=0.0, scale=std, size=matrix_shape).astype(np.float32)
        imag_comp = rng.normal(loc=0.0, scale=std, size=matrix_shape).astype(np.float32)
        self._random_matrix = real_comp + 1.0j * imag_comp
        self._seed = seed

    def __enter__(self):
        self.open()
        return self

    def open(self):
        self.active = True

    def close(self):
        self.active = False

    def __exit__(self, *args):
        self.close()

    def reserve(self, _):
        pass

    @contextmanager
    def acquiring(self, *args, **kwargs):
        try:
            yield
        finally:
            pass

    def transform_single(self, X):
        if X.ndim == 1:
            return self.transform1(np.expand_dims(X, axis=0))[0]
        else:
            assert X.ndim == 2 and X.shape[0] == 1
            return self.transform1(X)[0]

    def transform1(self, X):
        assert X.ndim == 2
        n_rows = self._random_matrix.shape[0]
        n_features = X.shape[1]
        if n_features > n_rows:
            raise ValueError("X must have {} columns".format(n_rows))
        elif n_features < n_rows:
            random_matrix = self._random_matrix[:n_features]
            warnings.warn("The number of rows of random matrix ({}) are bigger than "
                          "X number of features ({}), consider reducing it for performance"
                          .format(n_rows, n_features))
        else:
            random_matrix = self._random_matrix
        if self._linear:
            return np.dot(X, random_matrix).real
        else:
            return np.abs(np.dot(X, random_matrix)) ** 2

    def transform2(self, X, Y, _=None):
        Y[:] = self.transform1(X)

    @property
    def random_matrix(self):
        return self._random_matrix

    @property
    def input_shape(self):
        """tuple(int), Shape of the input device, in elements and cartesian coordinates
        """
        return 1140, 912

    @property
    def nb_features(self):
        """int: Total number of features supported by the OPU"""
        if self._random_matrix is not None:
            return self._random_matrix.shape[0]
        else:
            return None

    @property
    def input_size(self):
        """int: Input size of the input device, in bytes"""
        return self.nb_features // 8

    @property
    def acq_state(self):
        return None

    @property
    def output_dtype(self):
        return np.float32

    @property
    def output_roi_strategy(self):
        return OutputRoiStrategy.mid_square

    @property
    def output_roi_increment(self):
        return 1

    @property
    def output_shape_max(self):
        return 1920, 1080

    @property
    def output_shape(self) -> Tuple2D:
        return self.output_roi[1]

    @property
    def output_readout_us(self):
        return self.exposure_us

    @property
    def gain_dB(self):
        return 0.

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove random_matrix from state, and replace it with what's needed
        state.pop("_random_matrix")
        has_matrix = self._random_matrix is not None
        state["matrix"] = {'has_matrix': has_matrix,
                           'seed': self._seed}
        if has_matrix:
            state["matrix"]["shape"] = self._random_matrix.shape

        return state

    def __setstate__(self, state):
        """Restore object with given state"""
        self.__dict__ = state
        # Restore random matrix
        if state['matrix']['has_matrix']:
            shape = state["matrix"]["shape"]
            self.build_random_matrix(*shape, state["matrix"]["seed"])
        else:
            self._random_matrix = None
