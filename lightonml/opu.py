# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

"""
This module contains the OPU class
"""
import warnings
from typing import Optional, Union, Tuple
from warnings import warn

import numpy as np
from contextlib import ExitStack
import attr

import lightonml
from lightonml.internal.config import get_host_option
from lightonml.internal import config, output_roi, utils, types
from lightonml.internal.user_input import OpuUserInput, InputTraits
from lightonml.internal.simulated_device import SimulatedOpuDevice
from lightonml.internal.device import OpuDevice, AcqState
from lightonml.context import ContextArray
from lightonml.internal.settings import OpuSettings, TransformSettings
from lightonml.internal.runner import TransformRunner, FitTransformRunner
from lightonml.internal.types import InputRoiStrategy, IntOrTuple, TransformOutput


# noinspection PyPep8Naming
class OPU:
    """Interface to the OPU.

    .. math:: \\mathbf{y} = \\lvert \\mathbf{R} \\mathbf{x} \\rvert^2

    Main methods are `transform`, `fit1d` and `fit2d`,
    and accept NumPy arrays or PyTorch tensors.

    Acquiring/releasing hardware device resources is done by open/close and a
    context-manager interface.

    Unless `open_at_init=False`, these resources are acquired automatically at init.
    If another process or kernel has not released the resources, an error will be
    raised, call `close()` or shutdown the kernel on the OPU object to release it.

    Parameters
    ----------
    n_components : int,
        dimensionality of the target projection space.
    opu_device : OpuDevice or SimulatedOpuDevice, optional
        optical processing unit instance linked to a physical or simulated device.
        If not provided, a device is properly instantiated.
        If opu_device is of type SimulatedOpuDevice, the random matrix is generated
        at __init__, using max_n_features and n_components
    max_n_features: int, optional
        maximum number of binary features that the OPU will transform
        used only if opu_device is a SimulatedOpuDevice,
        in order to initiate the random matrix
    config_file : str, optional
        path to the configuration file (for dev purpose)
    config_override: dict, optional
        for override of the config_file (for dev purpose)
    verbose_level: int, optional
        deprecated, use lightonml.set_verbose_level instead
    input_roi_strategy: types.InputRoiStrategy, optional
        describes how to display the features on the input device
        .. seealso:: `lightonml.types.InputRoiStrategy`
    open_at_init: bool, optional
        forces the setting of acquiring hardware resource at init. If
        not provided, follow system's setting (usually True)

    Attributes
    ----------
    n_components: int
        dimensionality of the target projection space.
    max_n_features: int
        maximum number of binary features that the OPU will transform
        writeable only if opu_device is a SimulatedOpuDevice,
        in order to initiate or resize the random matrix
    device: OpuDevice or SimulatedOpuDevice
        underlying hardware that performs transformation (read-only)
    input_roi_strategy: types.InputRoiStrategy, optional
        describes how to display the features on the input device
    verbose_level: int, optional
        0, 1 or 2. 0 = no messages, 1 = most messages, and 2 = messages
    """

    def __init__(self, n_components: int = 200000,
                 opu_device: Optional[Union[OpuDevice, SimulatedOpuDevice]] = None,
                 max_n_features: int = 1000, config_file: str = "",
                 config_override: dict = None, verbose_level: int = -1,
                 input_roi_strategy: types.InputRoiStrategy = types.InputRoiStrategy.auto,
                 open_at_init: bool = None, disable_pbar=False):

        self.__opu_config = None
        self.__config_file = config_file
        self.__config_override = config_override
        self._max_n_features = max_n_features
        self.disable_pbar = disable_pbar

        # Get trace and print functions
        if verbose_level != -1:
            warnings.warn("Verbose level arg will removed in 1.3, "
                          "Use lightonml.set_verbose_level instead",
                          DeprecationWarning)
            lightonml.set_verbose_level(verbose_level)
        else:
            verbose_level = lightonml.get_verbose_level()
        self._debug = lightonml.get_debug_fn()
        self._trace = lightonml.get_trace_fn()
        self._print = lightonml.get_print_fn()

        # Device init, or take the one passed as input
        if not opu_device:
            opu_type = self.config["type"]
            self._base_frametime_us = self.config["input"]["frametime_us"]
            self._base_exposure_us = self.config["output"]["exposure_us"]
            seq_nb_prelim = self.config.get("sequence_nb_prelim", 0)
            name = self.config["name"]
            self.device = OpuDevice(opu_type, self._base_frametime_us,
                                    self._base_exposure_us, seq_nb_prelim,
                                    None, verbose_level, name)
        else:
            if not isinstance(opu_device, (SimulatedOpuDevice, OpuDevice)):
                raise TypeError("opu_device must be of type {} or {}"
                                .format(SimulatedOpuDevice.__qualname__,
                                        OpuDevice.__qualname__))
            self.device = opu_device
            self._base_frametime_us = self.device.frametime_us
            self._base_exposure_us = self.device.exposure_us

        if self._s.simulated:
            # build the random matrix if not done already
            self._resize_rnd_matrix(max_n_features, n_components)

        self._output_roi = output_roi.OutputRoi(self.device.output_shape_max,
                                                self.device.output_roi_strategy,
                                                self._s.allowed_roi, self._s.min_n_components)
        # This also sets the output ROI
        self.n_components = n_components
        self.input_roi_strategy = input_roi_strategy
        # Runner initialized when entering fit
        self._runner = None  # type: Optional[TransformRunner]
        # ExitStack for device acquisition, initialized when entering fit
        self._acq_stack = ExitStack()
        self._trace("OPU initialized")

        # Open at init, unless relevant host.json option is False
        if open_at_init is None:
            open_at_init = get_host_option("lightonml_open_at_init", True)
        if open_at_init:
            self.open()

    def _tr_settings(self, no_input=False, **override) -> TransformSettings:
        """Returns transform settings for feeding to TransformRunner"""
        init = TransformSettings(self.input_roi_strategy, self.n_components)
        settings = attr.evolve(init, **override)
        if no_input:
            # If no input_roi, just choose a full strategy
            settings.input_roi_strategy = InputRoiStrategy.full
            assert settings.input_roi is None
        return settings

    def fit1d(self, X=None, n_features: int = None,
              packed: bool = False, online=False, **override):
        """
        Configure OPU transform for 1d vectors

        The function can be either called with input vector, for fitting OPU
        parameters to it, or just vector dimensions, with ``n_features``.

        When input is bit-packed the packed flag must be set to True.

        When input vectors must be transformed one by one, performance will
        be improved with the online flag set to True.

        Parameters
        ----------
        X: np.ndarray or torch.Tensor
            Fit will be made on this vector to optimize transform parameters
        n_features: int
            Number of features for the input, necessary if X parameter isn't provided
        packed: bool
            Set to true if the input vectors will be already bit-packed
        online: bool, optional
            Set to true if the transforms will be made one vector after the other
            defaults to False
        override: keyword args for overriding transform settings (advanced parameters)
        """
        return self.__fit(X, n_features, packed, online, False, **override)

    def fit2d(self, X=None, n_features: Tuple[int, int] = None,
              packed: bool = False, online=False, **override):
        """
        Configure OPU transform for 2d vectors

        The function can be either called with input vector, for fitting OPU
        parameters to it, or just vector dimensions, with `n_features`.

        When input is bit-packed the packed flag must be set to True.
        Number of features must be then provided with `n_features`

        When input vectors must be transformed one by one, performance will
        be improved with the online flag set to True.

        Parameters
        ----------
        X: np.ndarray or torch.Tensor
            a 2d input vector, or batch of 2d input_vectors, binary encoded, packed or not
        n_features: tuple(int)
            Number of features for the input, necessary if X parameter isn't provided, or
            if input is bit-packed
        packed: bool, optional
            whether the input data is in bit-packed representation
            if True, each input vector is assumed to be a 1d array, and the "real" number
            of features must be provided as n_features
            defaults to False
        online: bool, optional
            Set to true if the transforms will be made one vector after the other
            defaults to False
        override: keyword args for overriding transform settings (advanced parameters)
        """
        return self.__fit(X, n_features, packed, online, True, **override)

    def transform(self, X) -> TransformOutput:
        """
        Performs the nonlinear random projections of one or several input vectors.

        The `fit1d` or `fit2d` method must be called before, for setting vector dimensions
        or online option.
        If you need to transform one vector after each other,

        Parameters
        ----------
        X:  np.ndarray or torch.Tensor
            input vector, or batch of input vectors.
            Each vector must have the same dimensions as the one given in `fit1d` or `fit2d`.

        Returns
        -------
        Y: np.ndarray or torch.Tensor
             complete array of nonlinear random projections of X,
             of size self.n_components
             If input is an ndarray, type is actually ContextArray,
             with a context attribute to add metadata
        """
        assert self._runner, "Call fit1d or fit2d before transform"
        assert self.device.active, "OPU device isn't active, use opu.open() or \"with opu:\""

        user_input = OpuUserInput.from_traits(X, self._runner.traits)
        self._debug(str(user_input))

        if user_input.is_batch:
            # With batch input start acquisition first
            assert self.device.acq_state != AcqState.online, \
                "Can't transform a batch of vectors when acquisition is" \
                " in online mode, only single vectors"
            with self.device.acquiring(n_images=self._s.n_samples_by_pass):
                out = self._runner.transform(user_input)
        else:
            out = self._runner.transform(user_input)
        Y = user_input.reshape_output(out)
        # if the input is a tensor, return a tensor in CPU memory
        if user_input.is_tensor:
            # noinspection PyPackageRequirements
            import torch
            return torch.from_numpy(Y)
        else:
            return Y

    # noinspection PyIncorrectDocstring
    def transform1d(self, *args, **kwargs) -> TransformOutput:
        """Performs the nonlinear random projections of one 1d input vector, or
        a batch of 1d input vectors.

        This function is only for backwards compatibility, prefer using `fit1d` followed
        by `transform`, or `fit_transform1d`

        .. warning:: when making several transform calls, prefer calling `fit1d`
            and then `transform`, or you might encounter an inconsistency in the
            transformation matrix.

        The input data can be bit-packed, where ``n_features = 8*X.shape[-1]``
        Otherwise ``n_features = X.shape[-1]``

        .. deprecated:: 1.2

        Parameters
        ----------
        X: np.ndarray or torch.Tensor
            a 1d input vector, or batch of 1d input_vectors, binary encoded, packed or not
            batch can be 1d or 2d. In all cases ``output.shape[:-1] = X.shape[:-1]``
        packed: bool, optional
            whether the input data is in bit-packed representation
            defaults to False
        override: keyword args for overriding transform settings (advanced parameters)

        Returns
        -------
        Y: np.ndarray or torch.Tensor
             complete array of nonlinear random projections of X,
             of size self.n_components
             type is actually ContextArray, with a context attribute to add metadata
        """
        warn("As of version 1.2 prefer calling fit1d then transform")
        return self.fit_transform1d(*args, **kwargs)

    # noinspection PyIncorrectDocstring
    def transform2d(self, *args, **kwargs) -> TransformOutput:
        """Performs the nonlinear random projections of one 2d input vector, or
        a batch of 2d input vectors.

        .. warning:: when making several `transform` calls, prefer calling `fit2d`
            and then `transform`, or you might encounter an inconsistency in the
            transformation matrix.

        This function is only for backwards compatibility, prefer using `fit2d` followed
        by transform, or `fit_transform2d`.

        .. deprecated:: 1.2

        Parameters
        ----------
        X: np.ndarray or torch.Tensor
            a 2d input vector, or batch of 2d input_vectors, binary encoded, packed or not
        packed: bool, optional
            whether the input data is in bit-packed representation
            if True, each input vector is assumed to be a 1d array, and the "real" number
            of features must be provided as n_2d_features
            defaults to False
        n_2d_features: list, tuple or np.ndarray of length 2
            If the input is bit-packed, specifies the shape of each input vector.
            Not needed if the input isn't bit-packed.
        override: keyword args for overriding transform settings (advanced parameters)

        Returns
        -------
        Y: np.ndarray or torch.Tensor
             complete array of nonlinear random projections of X,
             of size self.n_components
             If input is an ndarray, type is actually ContextArray,
             with a context attribute to add metadata
        """
        warn("As of version 1.2 prefer calling fit2d then transform, or fit_transform2d")
        return self.fit_transform2d(*args, **kwargs)

    def fit_transform1d(self, X, packed: bool = False,
                        **override) -> ContextArray:
        """Performs the nonlinear random projections of 1d input vector(s).

        This function is the one-liner equivalent of `fit1d` and `transform` calls.

        .. warning:: when making several transform calls, prefer calling `fit1d`
            and then `transform`, or you might encounter an inconsistency in the
            transformation matrix.

        The input data can be bit-packed, where ``n_features = 8*X.shape[-1]``
        Otherwise ``n_features = X.shape[-1]``

        If tqdm module is available, it is used for progress display

        Parameters
        ----------
        X: np.ndarray or torch.Tensor
            a 1d input vector, or batch of 1d input_vectors, binary encoded, packed or not
            batch can be 1d or 2d. In all cases ``output.shape[:-1] = X.shape[:-1]``
        packed: bool, optional
            whether the input data is in bit-packed representation
            defaults to False
        override: keyword args for overriding transform settings (advanced parameters)

        Returns
        -------
        Y: np.ndarray or torch.Tensor
             complete array of nonlinear random projections of X,
             of size self.n_components
             If input is an ndarray, type is actually ContextArray,
             with a context attribute to add metadata

        """
        self.fit1d(X, None, packed, False, **override)
        return self.transform(X)

    def fit_transform2d(self, X, packed: bool = False, n_2d_features=None,
                        **override) -> ContextArray:
        """Performs the nonlinear random projections of 2d input vector(s).

        This function is the one-liner equivalent of `fit2d` and `transform` calls.

        .. warning:: when making several transform calls, prefer calling `fit2d`
            and then `transform`, or you might encounter an inconsistency in the
            transformation matrix.

        If tqdm module is available, it is used for progress display

        Parameters
        ----------
        X: np.ndarray or torch.Tensor
            a 2d input vector, or batch of 2d input_vectors, binary encoded, packed or not
        packed: bool, optional
            whether the input data is in bit-packed representation
            if True, each input vector is assumed to be a 1d array, and the "real" number
            of features must be provided as n_2d_features
            defaults to False
        n_2d_features: list, tuple or np.ndarray of length 2
            If the input is bit-packed, specifies the shape of each input vector.
            Not needed if the input isn't bit-packed.
        override: keyword args for overriding transform settings (advanced parameters)

        Returns
        -------
        Y: np.ndarray or torch.Tensor
             complete array of nonlinear random projections of X,
             of size self.n_components
             If input is an ndarray, type is actually ContextArray,
             with a context attribute to add metadata
        """
        self.fit2d(X, n_2d_features, packed, False, **override)
        return self.transform(X)

    def __fit(self, X, n_features: IntOrTuple,
              packed: bool, online: bool, is_2d_features: bool,
              **override):
        """Internal working of the fitXd calls

        Instantiates a TransformRunner, and start online acq if needs be.
        """
        if X is not None:
            # Input is provided, do the fit with user input
            user_input = OpuUserInput.from_input(X, packed, is_2d_features, n_features)
            tr_settings = self._tr_settings(no_input=False, **override)
            self._runner = FitTransformRunner(self._s, tr_settings, user_input,
                                              device=self.device,
                                              disable_pbar=self.disable_pbar)
        else:
            # Only dimensions are provided, no fitting happens on input
            assert n_features, "either input vector or n_features must be specified"
            # tr_settings has no input_roi, since it uses X to compute it
            tr_settings = self._tr_settings(no_input=True, **override)
            traits = InputTraits(n_features, packed)
            self._runner = TransformRunner(self._s, tr_settings, traits,
                                           device=self.device,
                                           disable_pbar=self.disable_pbar)
        self._acq_stack.close()
        if online:
            if self._s.no_single_transform:
                raise RuntimeError("Online transform isn't available with this OPU")
            # Start acquisition only if online. Batch transform start their own.
            self._acq_stack.enter_context(self.device.acquiring(online=True))

    def __enter__(self):
        """Context manager interface that acquires hardware resources
        used by the OPU device."""
        self.__active_before_enter = self.device.active
        self.open()
        return self

    def __exit__(self, *args):
        # Don't close if OPU was already active
        if not self.__active_before_enter:
            self.close()

    def open(self):
        """Acquires hardware resources used by the OPU device

        .. seealso:: `close()` or use the context manager interface for
            closing at the end af an indent block
        """
        if self.device.active:
            return
        self.device.open()
        # initial reservation for giving batch transforms a buffer ready to use
        self.device.reserve(self._s.n_samples_by_pass)
        if self._s.detect_trigger:
            # Detect trigger issue, and take action if needed
            issue = utils.detect_trigger_issue(self.device)
            if issue:
                # noinspection PyProtectedMember,PyUnresolvedReferences
                self.device._OpuDevice__opu.nb_prelim = 1
                self._debug("trigger issue detected, workaround applied")
            else:
                self._debug("trigger issue not detected")

        self._debug("OPU opened")

    def close(self):
        """Releases hardware resources used by the OPU device"""
        self._acq_stack.close()
        self.device.close()
        self._debug("OPU closed")

    @property
    def config(self):
        """Returns the internal configuration object"""
        # Load it when asked first time
        if not self.__opu_config:
            self.__opu_config = config.load_config(self.__config_file, self._trace)
            if self.__config_override is not None:
                utils.recurse_update(self.__opu_config, self.__config_override)

        return self.__opu_config

    @property
    def max_n_components(self):
        return self._output_roi.max_components

    @property
    def n_components(self) -> int:
        return self._n_components

    @n_components.setter
    def n_components(self, value: int):
        self.device.output_roi = self._output_roi.compute_roi(value)
        if self._s.simulated:
            self._resize_rnd_matrix(self.max_n_features, value)
        # We used to call device.reserve here, but moved to device.acquiring()
        self._n_components = value

    @property
    def max_n_features(self) -> int:
        return self._s.max_n_features

    @max_n_features.setter
    def max_n_features(self, value: int):
        if not self._s.simulated:
            raise AttributeError("max_n_feature can't be set if device is real")
        self._resize_rnd_matrix(value, self._n_components)
        self._max_n_features = value

    @property
    def _s(self) -> OpuSettings:
        """Returns immutable settings associated with the OPU
        Settings are immutable (attrs frozen), so generate it at
        each call. Performance impact is negligible"""
        # Get default value
        pass_default = attr.fields(OpuSettings).n_samples_by_pass.default

        # Common settings to both simulated and base
        kwargs = {"input_shape": self.device.input_shape,
                  "output_max_shape": self.device.output_shape_max,
                  "frametime_us": self._base_frametime_us,
                  "exposure_us": self._base_exposure_us}

        if isinstance(self.device, SimulatedOpuDevice):
            # Notice we never query self.config here, in order not to
            # need a configuration file for simulated device
            return OpuSettings(max_n_features=self._max_n_features,
                               n_samples_by_pass=pass_default,
                               simulated=True, **kwargs
                               )

        return OpuSettings(
            max_n_features=int(np.prod(self.device.input_shape)),
            # Will use defaults of OpuSettings if not found
            n_samples_by_pass=self.config.get("n_samples_by_pass", pass_default),
            min_batch_size=self.config["input"].get("minimum_batch_size", 0),
            allowed_roi=self.config["output"].get("allowed_roi"),
            # min_n_components is linked to the minimum output size
            min_n_components=self.config["output"].get("minimum_output_size", 0),
            ones_range=self.config["ones_range"],
            n_tries=self.config.get("n_transform_tries", 5),
            detect_trigger=self.config.get("detect_trigger_issue", False),
            no_single_transform=self.config.get("no_single_transform", False),
            **kwargs)

    def _resize_rnd_matrix(self, n_features: int, n_components: int):
        """Resize device's random matrix"""
        rnd_mat = self.device.random_matrix
        if rnd_mat is None or rnd_mat.shape != (n_features, n_components):
            self._print("OPU: computing the random matrix... ", end='', flush=True)
            self.device.build_random_matrix(n_features, n_components)
            self._print("OK")

    def version(self):
        """Returns a multi-line string containing name and versions of the OPU"""
        from lightonml import __version__ as lgversion
        version = []

        # Build OPU name
        opu_name = self.__opu_config['name']
        opu_version = self.__opu_config['version']
        opu_location = self.__opu_config['location']
        version.append('OPU ' + opu_name+'-'+opu_version+'-'+opu_location)

        # module version
        version.append("lightonml version " + lgversion)
        version.append(self.device.versions())
        return '\n'.join(version)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove logging functions, they can't be pickled
        state.pop("_debug")
        state.pop("_trace")
        state.pop("_print")
        # acq stack can't be pickled, will be restored
        state.pop("_acq_stack")
        # If acquisition is ongoing, close it
        state["__online_acq"] = self.device.acq_state == AcqState.online
        self._acq_stack.close()
        # Device itself is closed on pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore logging functions removed at getstate
        self._debug = lightonml.get_debug_fn()
        self._trace = lightonml.get_trace_fn()
        self._print = lightonml.get_print_fn()
        self._acq_stack = ExitStack()
        # Restore online acquisition if it was the case
        if state["__online_acq"]:
            self._acq_stack.enter_context(self.device.acquiring(online=True))
