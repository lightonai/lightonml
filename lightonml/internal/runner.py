# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

# noinspection PyProtectedMember
import time
import datetime as dt
import warnings
from functools import partial
from typing import TYPE_CHECKING, Callable
import numpy as np
import pkg_resources

from lightonml import get_verbose_level
from lightonml.context import Context, ContextArray
from lightonml.internal.device import OpuDevice, AcqState
from lightonml.internal import utils
from lightonml.internal.input_roi import InputRoi
from lightonml.internal.formatting import model1_formatter, model1_plain_formatter, FlatFormatter
from lightonml.internal.user_input import OpuUserInput, InputTraits
from lightonml.internal.progress import Progress

if TYPE_CHECKING:
    from lightonml.internal.settings import OpuSettings, TransformSettings


class TransformRunner:
    """Internal class for use with OPU transform

    The runner is responsible for initiating and running transform according
    to OPU input traits and setting.

    It runs the transform directly by talking to the OPU device.

    This class is short-lived for each OPU fit, and bound to an InputTraits
    (calling transform on it with different traits will raise an error) 
    After Init with opu_input or features_shape, it can be used to run a transform

    When fit is called on real data, use FitTransformRunner instead (child class below)
    """

    def __init__(self, opu_settings: "OpuSettings", settings: "TransformSettings",
                 traits: InputTraits, device: OpuDevice = None,
                 roi_compute: Callable = None, disable_pbar=False):
        self.s = opu_settings   # OPU settings
        self.t = settings  # User's transform settings
        self.device = device  # Device for running the transform
        self._traits = traits

        # Whether input's n_feature matches the max number of features
        # If yes, no need to format (use plain formatter)
        self.input_matches_max = self._traits.n_features_s == self.s.max_n_features

        from lightonml import get_print_fn, get_trace_fn, get_debug_fn
        self._print = get_print_fn()
        self._trace = get_trace_fn()
        self._debug = get_debug_fn()
        self.disable_pbar = disable_pbar

        # ones_info says whether the input has sufficient ones ratio
        # can be None if not automatic input_roi, actually filled in FitTransformRunner
        self.ones_info = {}

        # Do input traits checks
        if self._traits.n_features_s > self.s.max_n_features:
            raise ValueError("input's number of features ({}) can't be greater than {}"
                             .format(self._traits.n_features_s, self.s.max_n_features))

        # Get the function used to compute ROI, uses self if not coming from child FitTransformRunner
        if roi_compute is None:
            roi_compute = self._roi_compute

        # Get a formatter
        if not self.s.simulated:
            self.formatter = self._configure_formatting(roi_compute)
        else:
            # Simulated OPU just needs a formatter that flattens 2D features
            self.formatter = FlatFormatter(self._traits.n_features)

        self._debug("Formatter {} with element size {}. Input ROI: {}, {} "
                    .format(self.formatter.fmt_type.name, self.formatter.factor,
                            self.formatter.roi_offset, self.formatter.roi_size))

        # allocate intermediate buffer for batch
        # batch_size = self._traits.n_samples_s
        # Buffer size must be 0 to allocate a single target
        # buffer_size = batch_size if batch_size > 1 else 0
        # buffer for online mode
        self.buffer = self.formatter.empty_target(0)
        # First adjust exposure
        self._adjust_exposure()
        self._adjust_frametime()

    @property
    def traits(self):
        """Traits of user input fon which the runner is fit"""
        return self._traits

    def _roi_compute(self):
        """Computes input ROI when Runner is instantiated with only traits, not data"""
        roi = InputRoi(self.s.input_shape, self.s.ones_range)
        return roi.compute_roi(self.t.input_roi_strategy,
                               self._traits.n_features)

    def _configure_formatting(self, roi_compute):
        """
        Computes input ROI (unless manually defined), and returns a Formatter object
        roi_compute is an caller that provides input ROI computation, used unless input_roi
        is manually specified.

        If opu is fit with user data, roi_compute is function of child FitTransformRunner
        Else it is computed
        """
        if self.input_matches_max:
            # Formatter in this case is straight-forward, plain formatter
            self._debug("Plain formatter")
            return model1_plain_formatter(self.s.input_shape)

        # Compute input ROI unless user provided one
        offset, size = self.t.input_roi or roi_compute()
        # Instantiate formatter
        return model1_formatter(self.s.input_shape, self._traits.n_features, offset, size)

    def transform(self, input_vectors: OpuUserInput, linear=False):
        """Do the OPU transform of input
        If batch transform, device acquisition must be started
        """
        assert self.device.active and input_vectors.traits == self._traits
        if not self.s.simulated:
            input_vectors.binary_check()
        else:
            # In simulated mode, SimulatedDevice must whether the transform is linear or not
            self.device._linear = linear
        context = self._get_context()
        self._pre_print(self.device, input_vectors.n_samples)
        X = input_vectors.reshape_input()

        if input_vectors.is_batch:
            t0 = time.time()
            nb_retries = 0
            # Batch transform, allocate the result, progress bar, and start acquisition
            # allocation of empty vector for iteration
            n_samples = input_vectors.n_samples_s
            Y = np.empty((n_samples, self._out_size), dtype=self.device.output_dtype)
            self._trace("Y allocated")
            indices = self.indices(input_vectors.n_samples_s)
            with Progress(n_samples, "OPU: transform", self.disable_pbar) as p:
                # iterate over consecutive pairs of indices
                # (https://stackoverflow.com/a/21303286)
                for i, (start, end) in enumerate(zip(indices, indices[1:])):
                    c = self.__batch_transform(X[start:end], Y[start:end], i)
                    p.update(end - start)
                    nb_retries += c
            t1 = time.time()
            # Fill context from opu settings at the end of the transform
            context.from_opu(self.device, dt.datetime.fromtimestamp(t0),
                             dt.datetime.fromtimestamp(t1))
        else:
            Y, nb_retries = self.__single_transform(X)
            context.from_opu(self.device, dt.datetime.now())

        if nb_retries:
            self._print("OPU number of retries: ", nb_retries)
        return ContextArray(Y, context)

    @property
    def _out_size(self) -> int:
        # determine the output size of Y; you want the real one, use output ROI size
        if self.t.raw_output_size and not self.s.simulated:
            return int(np.prod(self.device.output_shape))
        else:
            return self.t.n_components

    def __single_transform(self, X):
        """
        Format and transform a single vector, called from transform.
        Returns tuple: output vector, number of retries
        """

        # do the formatting
        self.formatter.apply(X, self.buffer, self._traits.packed)
        # Check what method to call, depending if we're in an online mode or not

        if self.s.no_single_transform:
            # Check proper version of the drivers to include the
            # appropriate transform1 binding
            pkg_resources.require("lightonopu>=1.3b3")
            # We need to do a batch transform, single transform isn't available
            batch_buffer = np.expand_dims(self.buffer, axis=0)
            # Pad the buffer to match minimum batch size
            padding = ((0, self.s.min_batch_size - 1), (0, 0))
            padded_buffer = np.pad(batch_buffer, padding, 'constant', constant_values=0)
            output, nb_retries = self.__try_or_giveup(self.device.transform1,
                                                      padded_buffer, -1, 1)
            return output[0, :self._out_size], nb_retries
        else:
            if self.device.acq_state == AcqState.online:
                transform = self.device.transform_online
            else:
                transform = self.device.transform_single
            output, n_retries = self.__try_or_giveup(transform, self.buffer)

            # reshape output to match n_components
            return output[:self._out_size], n_retries

    def __batch_transform(self, ins: np.ndarray, output: np.ndarray, batch_index):
        """Format and transform a single chunk of encoded vectors, called from transform"""
        # TODO use self.buffer
        X_format = self.formatter.format(ins, self._traits.packed)
        batch_size = ins.shape[0]
        if batch_size <= self.s.min_batch_size:
            # if batch size is too small, pad it with empty vectors
            padding = ((0, self.s.min_batch_size - batch_size), (0, 0))
            X_padded = np.pad(X_format, padding, 'constant', constant_values=0)
        else:
            X_padded = X_format
        # OPU transform, try it up to 5 times if it fails
        _, nb_retries = self.__try_or_giveup(self.device.transform2,
                                             X_padded, output, batch_index)

        return nb_retries

    def __try_or_giveup(self, func, *args, **kwargs):
        # in verbose mode, print the exception
        print_exc = get_verbose_level() >= 2
        return utils.try_or_giveup(func, self.s.n_tries, print_exc, *args, **kwargs)

    def _get_context(self):
        context = Context()
        if self.formatter:
            context.input_roi_upper = self.formatter.roi_offset
            context.input_roi_shape = self.formatter.roi_size
            context.fmt_type = self.formatter.fmt_type
            context.fmt_factor = self.formatter.factor
        context.n_ones = self.ones_info.get("total", None)
        return context

    def _pre_print(self, device, shape):
        self._print(f'OPU: random projections of an array of'
                    f' size {shape}')
        self._print(f"OPU: using frametime {device.frametime_us} μs, exposure {device.exposure_us} μs, "
                    f"output ROI {device.output_roi}")

    def _adjust_frametime(self):
        """frametime can be lower than output readout in large ROI"""
        # Changing ROI, or raising exposure, can change minimum frame-time
        if not self.device:
            return
        min_frametime_us = max(self.device.output_readout_us,
                               self.device.exposure_us) + 50
        if self.t.frametime_us != 0:
            # Frametime overridden by transform settings
            self.device.frametime_us = self.t.frametime_us
        elif self.s.frametime_us < min_frametime_us:
            # If minimum frametime not reached, set it
            self.device.frametime_us = min_frametime_us
        else:
            # Nothing special, use opu settings
            self.device.frametime_us = self.s.frametime_us

    def _adjust_exposure(self):
        if not self.device:
            return
        ones_factor = self.ones_info.get("value", None)
        # Exposure overridden by transform settings
        if self.t.exposure_us != 0:
            self.device.exposure_us = self.t.exposure_us
            self._debug(f"Base exposure overridden at {self.t.exposure_us} µs")
            if ones_factor is not None and ones_factor > 1:
                warnings.warn("Exposure is overridden, while it would need to "
                              f"be reduced by a factor {ones_factor}")
        elif ones_factor is not None and ones_factor > 1:
            # check_ones > 1 means too much ones on the input device
            # we have to lower exposure in order not to saturate excessively
            self.device.exposure_us = self.s.exposure_us / ones_factor
            self._debug(f"Reducing exposure by a factor {ones_factor}")
        else:
            self.device.exposure_us = self.s.exposure_us

    def indices(self, n_samples: int) -> np.ndarray:
        # Divide indices in batches of n_samples_by_pass
        return self._get_batch_indices(n_samples, self.s.n_samples_by_pass)

    @staticmethod
    def _get_batch_indices(total_size: int, slice_size: int) -> np.ndarray:
        """Given total_size, return an array with intermediate slices
        e.g. 55 with a slice size of 10 gives [0, 10, 20, 30, 40, 50, 55]
        """
        (nb_runs, remainder) = divmod(total_size, slice_size)
        indices = np.arange(nb_runs + 1) * slice_size
        if remainder:
            indices = np.append(indices, total_size)
        return indices

    def _count_ones(self, X: np.ndarray, packed: bool):
        """count avg number of ones in binary array
        X: 2D ndarray
        """
        return utils.count_ones(X, self.s.n_count, packed)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove logging functions, they can't be pickled
        state.pop("_debug")
        state.pop("_trace")
        state.pop("_print")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore logging functions removed at getstate
        import lightonml
        self._debug = lightonml.get_debug_fn()
        self._trace = lightonml.get_trace_fn()
        self._print = lightonml.get_print_fn()


class FitTransformRunner(TransformRunner):
    """
    TransformRunner that uses input vector for ROI computation
    """

    def __init__(self, opu_settings: "OpuSettings", settings: "TransformSettings",
                 user_input: OpuUserInput, device: OpuDevice = None,
                 disable_pbar=False):
        # Send the roi_compute method as the ROI compute function
        comp_func = partial(self.roi_compute, user_input)
        super().__init__(opu_settings, settings, user_input.traits,
                         device, comp_func, disable_pbar)

    def roi_compute(self, user_input: OpuUserInput):
        # with automatic input roi, compute number of ones before
        # This case is met only when input is provided to runner creation
        # (i.e. opu.fit with an input)
        #
        # Count number of ones, and then compute ROI with "auto" strategy
        roi = InputRoi(self.s.input_shape, self.s.ones_range)
        n_ones = self._count_ones(user_input.reshape_input(), user_input.traits.packed)
        n_features_s = self._traits.n_features_s
        self._debug("Counted an average of"
                    " {:.2f} ones in input of size {} ({:.2f}%)."
                    .format(n_ones, n_features_s, 100 * n_ones / n_features_s))
        return roi.compute_roi(self.t.input_roi_strategy,
                               self._traits.n_features, n_ones, self.ones_info)
