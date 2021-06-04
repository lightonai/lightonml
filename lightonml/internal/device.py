# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
import sys
from builtins import object, TypeError
from contextlib import contextmanager, ExitStack
from enum import Enum
from typing import Union, Tuple, List
import numpy as np

from lightonml.internal.pidfile import PidFile, ProcessRunningException
from lightonml.internal.types import OutputRoiStrategy, Roi, Tuple2D

# Default values for hardware parameters
default_frametime_us = 500
default_exposure_us = 400
default_gain_dB = 0.

# for constant. checked for truth when hardware is opened
_input1_shape = (912, 1140)
_output1_shape_max = (1920, 1080)
_output2_shape_max = (2040, 1088)
_output3_shape_max = (2048, 1088)


class OpuAlreadyUsedException(Exception):
    """Exception raised when hardware resources for an OPU are already taken"""
    pass


class AcqState(Enum):
    """Acquisition state (see common/include/opu_types.hpp)"""
    stopped = 1
    batch = 2
    online = 3


# noinspection PyPep8Naming
class OpuDevice(object):
    """
    Class for hardware interface with a LightOn OPU.

    Implements a context manager interface for acquiring access to the OPU,
    but most properties are gettable and settable even though the OPU isn't active.

    """

    # noinspection PyUnresolvedReferences
    def __init__(self, opu_type: str, frametime_us: int,
                 exposure_us: int, sequence_nb_prelim=0,
                 output_roi: Roi = None, verbose=0, name="opu"):

        if opu_type == "type1":
            from lightonopu import opu1_pybind
            self.__opu = opu1_pybind.OPU()
            self._output_shape_max = _output1_shape_max
            # With Model1 we just get the ROI in the middle
            self._output_roi_strategy = OutputRoiStrategy.mid_square
            self._output_roi_increment = 8
        elif opu_type == "type2":
            from lightonopu import opu2_pybind
            self.__opu = opu2_pybind.OPU()
            self._output_shape_max = _output2_shape_max
            # With Model2 we get max width in the middle
            self._output_roi_strategy = OutputRoiStrategy.mid_width
            self._output_roi_increment = 1
        elif opu_type == "type3":
            from lightonopu import opu3_pybind
            self.__opu = opu3_pybind.OPU()
            self._output_shape_max = _output3_shape_max
            # With Model2 we get max width in the middle
            self._output_roi_strategy = OutputRoiStrategy.mid_width
            self._output_roi_increment = 1
        else:
            raise TypeError("Don't know this OPU type: " + opu_type)

        # context for pid file
        self.pidfile = ExitStack()
        self.opu_type = opu_type
        # "off" fields allow to know what to send at resource acquisition
        # force to int if input is e.g. a float
        self._frametime_us_off = int(frametime_us)
        self._exposure_us_off = int(exposure_us)
        self._gain_dB_off = default_gain_dB
        self._output_roi_off = output_roi
        self._reserved_off = 0
        self._sequence_nb_prelim = sequence_nb_prelim
        self.name = name

        self.verbose = verbose
        from lightonml import get_trace_fn, get_debug_fn
        self._trace = get_trace_fn()
        self._debug = get_debug_fn()

        # forward opu interface to class
        self.transform1 = self.__opu.transform1
        self.transform2 = self.__opu.transform2
        self.transform_single = self.__opu.transform_single
        self.transform_online = self.__opu.transform_online
        if hasattr(self.__opu, "transform_online_test"):
            self.transform_online_test = self.__opu.transform_online_test

    def open(self):
        """
        Acquires hardware resource.

        Do nothing if the resource is already acquired in the current object.
        If the resource isn't available, do 4 retries until it is available,
        or raise a OPUUsedByOther exception

        Equivalent is to use context manager interface::

            with opu:
                outs = opu.transform(ins)

        Raises
        ------
        self.OPUUsedByOther
            if hardware has already been acquired in another process or object
        """
        # Just return if it's already active
        if self.__opu.active:
            return
        # A PID file allows to check whether an OPU of this name already runs on the system,
        # and provide the user with a helpful message
        try:
            log = sys.stdout if self.verbose >= 2 else None
            warn = sys.stderr if self.verbose >= 2 else None
            self.pidfile.enter_context(PidFile("/tmp/lighton/lighton-{}.pid"
                                               .format(self.name), log, warn))
        except ProcessRunningException:
            # "from None" to disable exception chaining, not useful here
            raise OpuAlreadyUsedException(_already_used_text) from None

        if self.verbose:
            print("Opening OPU... ", end='', flush=True)

        # noinspection PyPep8
        try:
            self.__opu.open(self._frametime_us_off, self._exposure_us_off,
                            self._sequence_nb_prelim, self.verbose)

            # check that input&output constants is conform to what the hardware tells
            if self.input_shape != tuple(self.__opu.input_shape):
                raise Exception("Incorrect values for internal input shape")
            if self.output_shape_max != tuple(self.__opu.output_shape_max):
                raise Exception("Incorrect values for internal max output shape")
            if self.input_size != self.__opu.input_size:
                raise Exception("Incorrect values for internal input size")
            if default_gain_dB != self.__opu.gain_dB:
                raise Exception("Incorrect defaults for output gain")

            # sets hardware parameter back
            self.gain_dB = self._gain_dB_off
            if self._output_roi_off is not None:
                self.output_roi = self._output_roi_off
            if self._reserved_off != 0:
                self.reserve(self._reserved_off)
            if self.verbose:
                print("OK")
        except:
            # Cleanup if an exception was thrown
            try:
                self.__opu.close()
            finally:
                self.pidfile.close()
            raise

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        # grab hardware values before closing
        try:
            if self.active:
                self._frametime_us_off = self.__opu.frametime_us
                self._exposure_us_off = self.__opu.exposure_us
                self._gain_dB_off = self.__opu.gain_dB
                self._output_roi_off = to_roi(self.__opu.output_roi)

            self.__opu.close()
            self._debug("OPU device closed")
        finally:
            # In the end make sure the pidfile is released
            self.pidfile.close()

    @contextmanager
    def acquiring(self, triggered=True, online=False, n_images=0):
        try:
            if n_images:
                self.reserve(n_images)
            if online:
                self.__opu.start_online_acq(triggered)
                self._trace("Online acq started")
            else:
                self.__opu.start_acq(triggered)
                self._trace("Normal acq started")
            yield
        finally:
            self.__opu.stop_acq()
            self._trace("Acquisition stopped")

    @property
    def active(self):
        """bool, whether the hardware resources have been acquired"""
        return self.__opu.active

    @property
    def input_shape(self) -> Tuple2D:
        """tuple(int), Shape of the input, in elements and cartesian coordinates
        """
        return _input1_shape

    @property
    def output_shape_max(self) -> Tuple2D:
        """tuple(int): Shape of the whole output (no ROI),
        in elements and cartesian coordinates"""
        return self._output_shape_max

    @property
    def output_roi_strategy(self) -> OutputRoiStrategy:
        return self._output_roi_strategy

    @property
    def output_roi_increment(self) -> int:
        return self._output_roi_increment

    @property
    def nb_features(self) -> int:
        """int: Total number of features supported by the OPU"""
        return self.input_shape[0] * self.input_shape[1]

    @property
    def input_size(self) -> int:
        """int: Input size, in bytes"""
        return self.nb_features // 8

    @property
    def exposure_us(self) -> int:
        """int: exposure for output, in microseconds"""
        return self.__opu.exposure_us if self.active else self._exposure_us_off

    @property
    def output_readout_us(self) -> int:
        """int: time given for readout, in microseconds"""
        if not self.active:
            raise RuntimeError("The Opu must be active to get this value")
        return self.__opu.output_readout_us

    @property
    def frametime_us(self) -> int:
        """int: time for which each input display, in microseconds"""
        return self.__opu.frametime_us if self.active else self._frametime_us_off

    @frametime_us.setter
    def frametime_us(self, value):
        if self.active:
            self.__opu.frametime_us = int(value)
        else:
            self._frametime_us_off = int(value)

    @exposure_us.setter
    def exposure_us(self, value):
        if self.active:
            self.__opu.exposure_us = int(value)
        else:
            self._exposure_us_off = int(value)

    def reserve(self, n_images):
        """Does internal allocation of a number of images, necessary for transform2 calls"""
        if self.active:
            self.__opu.reserve(int(n_images))
        else:
            self._reserved_off = int(n_images)

    @property
    def output_shape(self) -> Tuple2D:
        """
        list(int): Shape of the current output ROI, in elements
        and cartesian coordinates
        """
        return self.output_roi[1]

    @property
    def acq_state(self) -> Union[AcqState, None]:
        return AcqState(self.__opu.acq_state) if self.active else None

    @property
    def output_dtype(self):
        return np.uint8

    @property
    def output_roi(self) -> Roi:
        """ tuple(list(int)): offset and size of the current output ROI"""
        return to_roi(self.__opu.output_roi) if self.active else self._output_roi_off

    @output_roi.setter
    def output_roi(self, value: Roi):
        if self.active:
            # Binding accepts tuple(list(int), list(inst))
            self.__opu.output_roi = (list(value[0]), list(value[1]))
        else:
            self._output_roi_off = value

    @property
    def gain_dB(self) -> float:
        """ Gain of the output (not implemented in every device)"""
        return self.__opu.gain_dB if self.active else self._gain_dB_off

    @gain_dB.setter
    def gain_dB(self, value):
        if self.active:
            self.__opu.gain_dB = value
        else:
            self._gain_dB_off = value

    def versions(self):
        """Returns multi-line string with device and libraries versions"""
        version = []
        # device version needs an active OPU
        if self.active:
            version.append(self.__opu.device_versions())
        # this is static so works even if non active
        version.append(self.__opu.library_versions())
        return '\n'.join(version)

    def __deepcopy__(self, memo):
        """
        Deep copy of an object

        Can't make a deep copy of an OPU that represents a hardware
        resource, However sklearn needs to clone it as an estimator
        """
        return self

    def __getstate__(self):
        """Closes and return current state"""
        state = {"active": self.active,
                 "opu_type": self.opu_type,
                 "frametime_us": self.frametime_us,
                 "exposure_us": self.exposure_us,
                 "gain_dB": self.gain_dB,
                 "output_ROI": self.output_roi,
                 "reserved": self._reserved_off,
                 "sequence_nb_prelim": self._sequence_nb_prelim,
                 "verbose": self.verbose,
                 "name": self.name}
        self.close()
        return state

    def __setstate__(self, state):
        """Restore object with given state"""
        self.__init__(state["opu_type"], state["frametime_us"], state["exposure_us"],
                      state["sequence_nb_prelim"], state["output_ROI"], state["verbose"],
                      state["name"])
        self.gain_dB = state["gain_dB"]
        self._reserved_off = state["reserved"]
        # If state was active then open OPU
        if state["active"]:
            self.open()

    def __str__(self):
        active = "Active" if self.active else "Inactive"
        return '{} OPU, frametime {} μs, exposure {} μs, output ROI {}'.format(
            active, self.frametime_us, self.exposure_us, self.output_roi)


_already_used_text = """
The OPU hardware resources have already been reserved by another OPU or OPUMap object.
The options to recover from this are:
 * shutdown the kernel if it's in a notebook;
 * call close() on the object when you don't need it anymore, or use the Python "with" statement.
"""


def to_roi(roi: Tuple[List[int], List[int]]) -> Roi:
    # Binding returns tuple(list, list), we want tuple(tuple, tuple)
    # noinspection PyTypeChecker
    return tuple(roi[0]), tuple(roi[1])
