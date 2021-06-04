# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numpy as np
from lightonml.internal import utils
from lightonml.internal import formatting


class InputDevice:
    """Utility class intended to use an input device independently,
    without an output device

    Example, to display full ones:
    device = InputDevice()
    device.display1d([1, 1, 1, 1l)
    """
    def __init__(self, frametime_us=500, verbose=False):
        # noinspection PyUnresolvedReferences
        from lightonopu import inputdev_pybind
        self.device = inputdev_pybind.InputDevice(verbose)
        # Currently only Model1 is supported (column order)
        self.plain_fmt = formatting.model1_plain_formatter(self.device.shape)
        self.frametime_us = frametime_us

    @property
    def shape(self):
        """Input shape, in cartesian coordinates (width, height) """
        return tuple(self.device.shape)

    @property
    def shape_r(self):
        """Input shape, in "numpy" coordinates (nrows, ncols) """
        return utils.rev(self.device.shape)

    @property
    def input_size_bytes(self):
        """size of raw bit-packed input"""
        return self.device.input_size

    @property
    def input_size(self):
        """number of elements"""
        return np.prod(self.device.shape)

    def display_raw(self, input_data: np.ndarray):
        """
        Display data as raw array (no formatting, already bit-packed)
        input_data size must match device.input_size_bytes
        """
        assert input_data.size == self.input_size_bytes
        self.device.display(input_data, self.frametime_us)

    def display1d(self, input_data: np.ndarray):
        """Display a 1d array, formatting it if it's not the right shape
        Display is raw if input array has device input size
        """
        assert input_data.ndim == 1
        if input_data.size == self.input_size:
            # if it's already at the device shape, use the plain formatter
            fmt_data = self.plain_fmt.format(input_data, packed=False)
        else:
            fmt = formatting.model1_formatter(self.device.shape, input_data.shape)
            fmt_data = fmt.format(input_data, packed=False)

        self.display_raw(fmt_data)

    def display2d(self, input_data: np.ndarray):
        """
        Display a 2d array, formatting it if it's not the right shape
        Display is raw if input array has input shape (or transposed)
        """
        assert input_data.ndim == 2
        # if it's already at the input shape, use the plain formatter
        if input_data.shape in [self.shape_r, self.shape]:
            fmt_data = self.plain_fmt.format(input_data, packed=False)
        else:
            fmt = formatting.model1_formatter(self.device.shape, input_data.shape)
            fmt_data = fmt.format(input_data, packed=False)

        self.display_raw(fmt_data)
