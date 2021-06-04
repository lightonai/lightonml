# -*- coding: utf8
import warnings

import torch
import torch.nn as nn

import lightonml
from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice


class OPUMap(nn.Module):
    """Adapter of the OPU to the Pytorch interface.

    Forward method is mapped to `transform <lightonml.opu.OPU.transform>` of
    the `OPU <lightonml.opu.OPU>` class

    .. seealso:: `lightonml.opu.OPU`

    Parameters
    ----------
    n_components: int,
        dimensionality of the target projection space.
    opu : lightonml.opu.OPU,
        optical processing unit instance
    ndims : int,
        number of dimensions of an input. Can be 1, 2 or 3.
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
        use the linear version of the OPU transform
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
        number of dimensions of an input. Can be 1, 2 or 3.
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
    fitted: bool
        if the OPU parameters have already been chosen.
    """
    def __init__(self, n_components, opu=None, ndims=1, n_2d_features=None, packed=False,
                 simulated=False, max_n_features=None, verbose_level=-1, linear=False):
        if verbose_level >= 0:
            lightonml.set_verbose_level(verbose_level)
        self.verbose_level = lightonml.get_verbose_level()
        super(OPUMap, self).__init__()
        if opu is None:
            if simulated:
                simulated_opu = SimulatedOpuDevice()
                if max_n_features is None:
                    raise ValueError("When using simulated=True, you need to provide max_n_features.")
                self.opu = OPU(opu_device=simulated_opu, max_n_features=max_n_features,
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
                warnings.warn("You provided a simulated OPU object but set simulated=False. "
                              "Will use simulated OPU.")
        self.n_components = self.opu.n_components
        if ndims not in [1, 2]:
            raise ValueError("Number of input dimensions must be 1 or 2")
        self.ndims = ndims
        self.n_2d_features = n_2d_features
        self.packed = packed
        self.simulated = simulated
        self.linear = linear
        self.max_n_features = max_n_features

        self.fitted = False
        self.online = False
        if lightonml.get_verbose_level() >= 1:
            print("OPU output is detached from the computational graph.")

    @property
    def n_components(self):
        return self.opu.n_components

    @n_components.setter
    def n_components(self, value):
        self.opu.n_components = value

    def forward(self, input):
        """Performs the nonlinear random projections.

        .. seealso:: `lightonml.opu.OPU.transform`
        """
        if not self.fitted:
            print("OPUMap was not fit to data. Performing fit on the first batch with default parameters...")
            self.fit(input)
        transform = self.opu.linear_transform if self.linear else self.opu.transform

        if self.online:
            output = torch.empty((len(input), self.n_components), dtype=torch.uint8)
            for i in range(len(input)):
                output[i] = transform(input[i])
            return output.detach()
        else:
            output = transform(input)
        return output.detach()

    def reset_parameters(self, input, y, n_features, packed, online):
        if online:
            self.online = True
        if self.ndims == 1:
            self.opu.fit1d(input, n_features=n_features, packed=packed, online=self.online)
        elif self.ndims == 2:
            self.opu.fit2d(input, n_features=n_features, packed=packed, online=self.online)
        else:
            assert False, "OPUMap.ndims={}; expected 1 or 2.".format(self.ndims)
        self.fitted = True
        return

    def fit(self, X=None, y=None, n_features=None, packed=False, online=False):
        """Configure OPU transform for 1d or 2d vectors

        The function can be either called with input vector, for fitting OPU
        parameters to it, or just vector dimensions, with `n_features`.

        When input is bit-packed the packed flag must be set to True.

        When input vectors must be transformed one by one, performance will
        be improved with the online flag set to True.

        Parameters
        ----------
        X: np.ndarray or torch.Tensor, optional,
            Fit will be made on this vector to optimize transform parameters
        y: np.ndarray or torch.Tensor, optional,
            For consistence with Sklearn API.
        n_features: int or tuple(int)
            Number of features for the input, necessary if X parameter isn't provided
        packed: bool
            Set to true if the input vectors will be already bit-packed
        online: bool, optional
            Set to true if the transforms will be made one vector after the other
            defaults to False

            .. seealso:: `lightonml.opu.OPU.fit1d`
            .. seealso:: `lightonml.opu.OPU.fit2d`
        """
        return self.reset_parameters(X, y, n_features, packed, online)

    def extra_repr(self):
        return 'out_features={}, n_dims={}, packed={} simulated={}'.format(
            self.n_components, self.n_dims, self.packed, self.simulated
        )

    def open(self):
        self.opu.open()

    def close(self):
        self.opu.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
