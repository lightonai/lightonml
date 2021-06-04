# -*- coding: utf8
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import lightonml
from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice


class OPUMap(BaseEstimator, TransformerMixin):
    """Adapter of the OPU to scikit-learn.
    Transform method is mapped to `transform <lightonml.opu.OPU.transform>` of
    the `OPU <lightonml.opu.OPU>` class.

    .. seealso:: `lightonml.opu.OPU`

    Parameters
    ----------
    n_components: int,
        dimensionality of the target projection space.
    opu : lightonml.opu.OPU,
        optical processing unit instance (created at init if not provided)
    ndims : int,
        number of dimensions of an input. Can be 1 or 2.
        if ndims is 1, transform accepts 1d vector or batch of 1d vectors.
        if ndims is 2, transform accepts 2d vector or batch of 2d vectors.
    packed: bool, optional
        whether the input data is in bit-packed representation
        if packed is True and ndims is 2, each input vector is assumed
        to be a 1d array, and the "real" number of features must be provided
        using n_2d_features parameter
        defaults to False
    n_2d_features: list(int) or tuple(int) or np.ndarray (optional)
        number of 2d features if the input is packed
    simulated: bool, default False,
        use real or simulated OPU
    linear: bool, default False,
        use the linear version of the OPU transform (lightonml.opu.OPU.linear_transform)
    max_n_features: int, optional
        maximum number of binary features that the OPU will transform
        used only if simulated=True, in order to initiate the random matrix
    verbose_level: int, optional
        Levels are 0: nothing, 1: print info, 2: debug info, 3: trace info
        deprecated, use lightonml.set_verbose_level instead

    Attributes
    ----------
    opu : lightonml.opu.OPU,
        optical processing unit instance
    n_components : int,
        dimensionality of the target projection space.
    ndims : int,
        number of dimensions of an input. Can be 1 or 2.
        if ndims is 1, transform accepts 1d vector or batch of 1d vectors.
        if ndims is 2, transform accepts 2d vector or batch of 2d vectors.
    packed: bool, optional
        whether the input data is in bit-packed representation
        if packed is True and ndims is 2, each input vector is assumed
        to be a 1d array, and the "real" number of features must be provided
        using n_2d_features parameter
        defaults to False
    n_2d_features: list(int) or tuple(int) or np.ndarray (optional)
        number of 2d features if the input is packed
    simulated: bool, default False,
        use real or simulated OPU
    linear: bool, default False,
        use the linear version of the OPU transform (lightonml.opu.OPU.linear_transform)
    max_n_features: int, optional
        maximum number of binary features that the OPU will transform
        used only if simulated=True, in order to initiate the random matrix
    """
    def __init__(self, n_components, opu=None, ndims=1, n_2d_features=None, packed=False,
                 simulated=False, max_n_features=None, verbose_level=-1, linear=False):
        # verbose_level shouldn't be used anymore, but put it as attributes
        # in order to comply with sklearn estimator
        if verbose_level >= 0:
            lightonml.set_verbose_level(verbose_level)
        self.verbose_level = lightonml.get_verbose_level()

        if opu is None:
            if simulated:
                simulated_opu_device = SimulatedOpuDevice()
                if max_n_features is None:
                    raise ValueError("When using simulated=True, you need to provide max_n_features.")
                self.opu = OPU(opu_device=simulated_opu_device, max_n_features=max_n_features,
                               n_components=n_components)
            else:
                self.opu = OPU(n_components=n_components)
        else:
            self.opu = opu
            self.opu.n_components = n_components
            if simulated and not isinstance(opu.device, SimulatedOpuDevice):
                warnings.warn("You provided a real OPU object but set simulated=True."
                              " Will use the real OPU.")
            if isinstance(opu.device, SimulatedOpuDevice) and not simulated:
                warnings.warn("You provided a simulated OPU object but set simulated=False."
                              " Will use simulated OPU.")

        if ndims not in [1, 2]:
            raise ValueError("Number of input dimensions must be 1 or 2")
        self.ndims = ndims
        self.n_2d_features = n_2d_features
        self.packed = packed
        self.simulated = simulated
        self.linear = linear
        self.max_n_features = max_n_features
        self.fitted = False

    @property
    def n_components(self):
        return self.opu.n_components

    @n_components.setter
    def n_components(self, value):
        self.opu.n_components = value

    def fit(self, X=None, y=None, n_features=None, packed=False, online=False):
        """
        Configure OPU transform for 1d or 2d vectors

        The function can be either called with input vector, for fitting OPU
        parameters to it, or just vector dimensions, with `n_features`.

        When input is bit-packed the packed flag must be set to True.

        When input vectors must be transformed one by one, performance will
        be improved with the online flag set to True.

        Parameters
        ----------
        X: np.ndarray,
            Fit will be made on this vector to optimize transform parameters
        y: np.ndarray,
            For sklearn interface compatibility
        n_features: int or tuple(int),
            Number of features for the input, necessary if X parameter isn't provided
        packed: bool, optional
            Set to true if the input vectors will be already bit-packed
            defaults to False
        online: bool, optional
            Set to true if the transforms will be made one vector after the other
            defaults to False

        Returns
        -------
        self
        """
        if self.ndims == 1:
            self.opu.fit1d(X, n_features=n_features, packed=packed, online=online)
        elif self.ndims == 2:
            self.opu.fit2d(X, n_features=n_features, packed=packed, online=online)
        else:
            assert False, "SklearnOPU.ndims={}; expected 1 or 2.".format(self.ndims)
        self.fitted = True
        return self

    def transform(self, X, y=None):
        """Performs the nonlinear random projections.

            .. seealso:: `lightonml.opu.OPU.transform`
        """
        if self.opu.n_components != self.n_components:
            self.opu.n_components = self.n_components
        if not self.fitted:
            print("OPUMap was not fit to data. Performing fit on the input with default parameters...")
            self.fit(X)
        transform = self.opu.linear_transform if self.linear else self.opu.transform
        return np.array(transform(X))

    def open(self):
        self.opu.open()

    def close(self):
        self.opu.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
