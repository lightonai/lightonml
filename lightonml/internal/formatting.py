# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numbers
from typing import Union, Tuple
import numpy as np
from operator import xor

from lightonml.internal import types
from lightonml.internal.types import Tuple2D
from lightonml.internal.utils import rev


class OpuFormatterBase:
    """
    Base class for formatting arbitrary size input into an OPU device input
    of fixed shape

    Parameters
    ----------
    n_features: int or tuple(int)
        Number of features in the input to be formatted. Can be 1d or 2d
    target_size: int
        Size of the OPU device input, in bytes

    Methods
    ------
    apply:
        formats src (bit-packed or not) into target
    format:
        same as apply, but allocate and returns target
    """
    def __init__(self, n_features, target_size):
        if isinstance(n_features, (list, tuple, np.ndarray)):
            self.n_features = np.asarray(n_features)
            assert issubclass(self.n_features.dtype.type, np.integer)
        elif isinstance(n_features, numbers.Integral):
            self.n_features = np.asarray([n_features])
        else:
            raise ValueError("n_features must be array-like or int")
        assert self.features_ndim in [1, 2], "n_features should be 1d or 2d"

        self.target_size = target_size
        self.fmt_type = types.FeaturesFormat.none
        self.factor = 0
        self.roi_size = self.roi_offset = (0, 0)
        from lightonml import get_trace_fn
        self._trace = get_trace_fn()

    @property
    def n_features_s(self):
        return np.prod(self.n_features)

    @property
    def features_ndim(self):
        return len(self.n_features)

    def apply(self, src: np.ndarray, target: np.ndarray, packed: bool):
        self._check_dims(src, target, packed)
        self._apply(src, target, packed, *self._get_src_dims(src, packed))
        self._trace("(formatting) input formatted")

    def format(self, src: np.ndarray, packed: bool):
        batch_size, _ = self._get_src_dims(src, packed)
        target = self.empty_target(batch_size)
        self.apply(src, target, packed)
        return target

    def _apply(self, src: np.ndarray, target: np.ndarray, packed: bool,
               batch_size: int, n_features: Tuple[int, ...]):
        pass

    def _check_dims(self, src, target, packed):
        """Checks dimensions on src and target"""
        batch_size, n_features = self._get_src_dims(src, packed)
        if batch_size:
            assert target.ndim == 2
        else:
            assert target.ndim == 1
        assert target.shape[-1] == self.target_size

    def _get_src_dims(self, src, packed) -> (int, Tuple[int, ...]):
        """Get batch size of src, or 0 if a single vector, and number of features"""
        if packed:
            if src.ndim == 2:
                # batch
                return src.shape[0], None
            if src.ndim == 1:
                # single
                return 0, None
            else:
                raise ValueError("Incorrect number of input dimensions, {}".format(src.shape))
        else:
            if src.ndim == self.features_ndim + 1:
                # batch
                return src.shape[0], src.shape[1:]
            elif src.ndim == self.features_ndim:
                # single
                return 0, src.shape
            else:
                raise ValueError("Incorrect number of input dimensions, {}".format(src.shape))

    def _flat_shape(self, batch_size):
        """Returns 1d or 2d shape with a scalar number of features"""
        if batch_size:
            return batch_size, self.n_features_s
        else:
            return self.n_features_s,

    def empty_target(self, batch_size: int = 0):
        """Returns numpy array of the correct size, with batch_size number of samples
        If batch_size is 0, returns a single vector
        """
        if batch_size:
            shape = (batch_size, self.target_size)
        else:
            shape = self.target_size
        out = np.empty(shape, dtype=np.uint8)
        self._trace("(formatting) format output allocated")
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove logging functions, they can't be pickled
        del state["_trace"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore logging functions removed at getstate
        from lightonml import get_trace_fn
        self._trace = get_trace_fn()


class PackedFormatterBase(OpuFormatterBase):
    """Base class for formatting input to a bit-packed array"""
    def __init__(self, n_features, device_shape, column_order, reverse_bits_order):
        self.device_shape = tuple(device_shape)
        assert len(device_shape) == 2
        self.device_size = np.prod(device_shape)
        assert self.device_size % 8 == 0, "Device size must be multiple of 8"
        super().__init__(n_features, target_size=self.device_size//8)
        self.device_shape_r = rev(device_shape)
        self.column_order = column_order
        self.reverse_bits_order = reverse_bits_order
        self.roi_size = self.device_shape


class PackedFormatter(PackedFormatterBase):
    """This class handles formatting of encoded (binary) data into the input device

    Input can be either vector, or batch of vectors, bit-packed or not.
    The input be C-Contiguous.

    (The class is actually an interface to formatting_v2 python extension.)

    Parameters
    ----------
    device_shape: tuple(int)
        Device shape, in cartesian coordinates
    n_features: int or tuple(int)
        Number of features accepted by the formatting.
        If 1D, formatting is lined
        If 2D, formatting is macro-2D
    roi_offset: tuple(int) (optional)
        Offset for input ROI, in cartesian coordinates (defaults to (0,0))
    roi_size: tuple(int) (optional)
        Size for input ROI, in cartesian coordinates (defaults to device_shape)
    column_order: bool
        True if the formatted vector is shaped in column order (vs row)
    reverse_bits_order: bool
        True if formatting must reverse the bits from native architecture
    """
    def __init__(self, device_shape: Tuple2D, n_features: Union[int, Tuple2D],
                 roi_offset: Tuple2D = None, roi_size: Tuple2D = None,
                 column_order: bool = False, reverse_bits_order=False):
        # this is pybind formatting module
        # import at init only, allowing to avoid pybind import
        # (as this module isn't totally portable)
        super().__init__(n_features, device_shape, column_order, reverse_bits_order)
        from lightonopu import formatting_v2
        # formatter must be given shape in contiguous rows, so reversed(cartesian_shape),
        # or column-major, so cartesian_shape
        # (reversed(reversed(cartesian_shape) = cartesian_shape)
        self.formatter_shape = self.device_shape if column_order else self.device_shape_r
        self.internal = formatting_v2.Formatter(self.formatter_shape)
        self._configure(roi_offset, roi_size)

    def _apply(self, src: np.ndarray, target: np.ndarray, packed: bool,
               batch_size: int, n_features: Tuple[int, ...]):
        """src can be single 1d/2d vector, or array of 1d/2d vectors.
        vectors should 1d or 2d depending on n_features"""
        assert src.flags['C_CONTIGUOUS']
        if packed:
            src2 = src
        else:
            assert n_features == tuple(self.n_features)
            # possibly reshape if 2d features
            shape = self._flat_shape(batch_size)
            # call packbits along the last dimension if not packed
            src2 = np.packbits(src.reshape(shape), axis=len(shape) - 1)
            self._trace("input packed")
        assert src2.dtype == np.uint8
        # signature for apply_format: (source: ndarray, target:ndarray, n_features: int)
        self.internal.apply_format(src2, target, self.n_features_s, self.reverse_bits_order)

    def _configure(self, roi_offset=None, roi_size=None):
        """Initialize formatting mapping
        roi_offset and roi_size to be given in cartesian coordinates
        """
        self.roi_offset = roi_offset or (0, 0)
        self.roi_size = roi_size or self.device_shape

        if np.greater(np.add(self.roi_offset, self.roi_size), self.device_shape).any():
            raise ValueError("source doesn't fit in target")

        # Reverse cartesian coordinates accordingly before internal functions
        if not self.column_order:
            roi_offset_ = rev(self.roi_offset)
            roi_size_ = rev(self.roi_size)
        else:
            roi_offset_ = self.roi_offset
            roi_size_ = self.roi_size

        # Call the actual functions from internal formatter
        # When n_features is 2d, use macro-2d, or lined if 1d
        if self.features_ndim == 2:
            if np.greater(self.n_features, roi_size_).any():
                raise ValueError("source doesn't fit in target")
            self.internal.define_macro_2d_mapping(self.n_features, roi_size_, roi_offset_)
            self.factor = self.internal.macro_2d_size
            self.fmt_type = types.FeaturesFormat.macro_2d
        elif self.features_ndim == 1:
            if self.n_features > np.prod(self.roi_size):
                raise ValueError("source doesn't fit in target")

            self.internal.define_lined_mapping(self.n_features, roi_size_, roi_offset_)
            self.factor = self.internal.lined_factor
            self.fmt_type = types.FeaturesFormat.lined

    def __getstate__(self):
        """Called at pickle"""
        state = self.__dict__.copy()
        # Internal object must drop, since it's a pybind object it's not pickeable
        del state["internal"]
        return state

    def __setstate__(self, state):
        """Called at unpickle"""
        # Recreate the formatting and configure it
        self.__dict__.update(state)
        from lightonopu import formatting_v2
        self.internal = formatting_v2.Formatter(self.formatter_shape)
        self._configure(self.roi_offset, self.roi_size)


class PackedPlainFormatter(PackedFormatterBase):
    """
    Formatting for data that matches device size or shape

    input must be provided in numpy-order (rows, columns)
    so input.shape = device_shape_r = reversed(device_shape)

    apply, format:
    allows to format 1d or 2d-shaped
    vector or batch of vectors matching device size, into a bit-packed with correct orientation
    If 2d-shaped, must be of shape device_shape_r
    if packed, can't be 2d-shaped
    """
    def __init__(self, device_shape: Tuple2D, column_order: bool, reverse_bits_order=False):
        super().__init__(device_shape, device_shape, column_order, reverse_bits_order)

    def _apply(self, src: np.ndarray, target: np.ndarray, packed: bool,
               batch_size: int, n_features: Tuple[int, ...]):
        """src can be:
         * single 1d vector
         * single (unpacked) 2d vector matching device shape
         * array of 1d vectors
         * array of (unpacked) 2d vectors matching device shape"""
        if packed:
            np.copyto(target, src)
            self._trace("(formatting) input copied")
        else:
            self._apply_unpacked(src, target, batch_size, n_features)

    def _check_dims(self, src, target, packed):
        if packed:
            assert src.dtype == np.uint8
            assert src.shape[-1] == self.target_size, \
                "bit-packed vector size must match device size: {}".format(self.target_size)
            assert target.shape[-1] == self.target_size

    def _apply_unpacked(self, src: np.ndarray, target: np.ndarray,
                        batch_size: int, n_features: Tuple[int, ...]):
        """Format a single, or array of, non bit-packed vector(s)
         that matches device shape (2d) or size (1d)"""
        if len(n_features) == 2:
            # If 2D, ravel it following the correct order
            assert n_features in [self.device_shape_r, self.device_shape], \
                "2D vector size must be {} (or transposed)".format(self.device_shape_r)
            # Handle transposed shape, in this case change the ravel order
            # Truth table (that's a XOR):
            # column_order  && shape_r -> 'F' (native shape)
            # column_order  && shape   -> 'C' (transpose shape)
            # !column_order && shape_r -> 'C' (native shape)
            # !column_order && shape   -> 'F' (transpose shape)
            if xor(self.column_order, n_features == self.device_shape):
                order_2d = 'F'
            else:
                order_2d = 'C'
        else:
            assert n_features[0] == self.device_size, \
                "1D vector size must be {}".format(self.device_size)
            order_2d = ''

        # Reverse bit order means little endian in the packbits call
        bit_order = 'little' if self.reverse_bits_order else 'big'
        if batch_size:
            for src_vector, trg_vector in zip(src, target):
                self._ravel_to_target(src_vector, trg_vector, order_2d, bit_order)
        else:
            self._ravel_to_target(src, target, order_2d, bit_order)
        self._trace("(formatting) input raveled")

    def _get_src_dims(self, src, packed) -> (int, Tuple[int, ...]):
        if packed:
            if src.ndim == 2:
                # batch
                return src.shape[0], None
            if src.ndim == 1:
                # single
                return 0, None
            else:
                raise ValueError("Incorrect number of input dimensions, {}".format(src.shape))
        else:
            if src.ndim == 1:
                return 0, src.shape
            elif src.ndim == 2:
                # src can be either a device shape, or
                if src.shape[-1] == self.device_size:
                    return src.shape[0], (self.device_size,)
                elif src.shape in [self.device_shape_r, self.device_shape]:
                    return 0, src.shape
                else:
                    raise ValueError("apply_plain function must be provided device shape (2D) "
                                     "or size (1D) input")
            elif src.ndim == 3:
                return src.shape[0], src.shape[1:]
            else:
                raise ValueError("plain function input number of dimensions must be 1, 2 or 3")

    @staticmethod
    def _ravel_to_target(src_vector, trg_vector, order_2d, bit_order):
        # format into target, with ravel if 2D
        if order_2d:
            src_vector_ = np.ravel(src_vector, order=order_2d)
        else:
            src_vector_ = src_vector
        np.copyto(trg_vector, np.packbits(src_vector_, bitorder=bit_order))


def model1_plain_formatter(device_shape: Tuple2D):
    """Returns PackedPlainFormatter configured for Model1 controllers"""
    return PackedPlainFormatter(device_shape, column_order=True, reverse_bits_order=True)


def model1_formatter(device_shape: Tuple2D, n_features,
                     roi_offset: Tuple2D = None, roi_size: Tuple2D = None):
    """Returns PackedFormatter configured for Model1 controllers"""
    return PackedFormatter(device_shape, n_features=n_features, roi_offset=roi_offset,
                           roi_size=roi_size, column_order=True, reverse_bits_order=True)


class FlatFormatter(OpuFormatterBase):
    """Formatter class that does unpacking and flattening 2D features
    For use with SimulatedOpuDevice
    """
    def __init__(self, n_features, dtype: np.dtype = np.uint8):
        # Target size is n_features_s, but not available before init
        target_size = int(np.prod(np.asarray(n_features))) * np.dtype(dtype).itemsize
        super().__init__(n_features, target_size)

    def _apply(self, src: np.ndarray, target: np.ndarray, packed: bool,
               batch_size: int, n_features: Tuple[int, ...]):
        """Unpack if needed, or reshape if 2D features"""
        if packed:
            target[:] = np.unpackbits(src, axis=src.ndim - 1)
        else:
            assert (self.n_features == n_features).all()
            shape = self._flat_shape(batch_size)
            target[:] = src.reshape(shape)
