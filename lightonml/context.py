# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import datetime as dt

import numpy as np

from lightonml.internal import types
from lightonml.internal.types import Roi, Tuple2D


def from_epoch(datetime_or_epoch):
    """Convert to datetime if argument is an epoch, otherwise return same"""
    if isinstance(datetime_or_epoch, dt.datetime):
        # already a datetime
        return datetime_or_epoch
    if isinstance(datetime_or_epoch, float):
        return dt.datetime.fromtimestamp(datetime_or_epoch)
    raise ValueError("Need an epoch or datetime")


class Context:
    """Describes the context of an OPU transform

    Attributes
    ----------
    exposure_us: int
        Exposure time of the output device (µs)
    frametime_us: int
        Exposure time of the input device (µs)
    output_roi: tuple(tuple(int))
        (offset, size) of the output device region of interest
    input_roi:
        (offset, size) of the input device region of interest
    start: float
        epoch of the start time of the transform
    end: float
        epoch of the end time of the transform
    n_ones: int
        average number of ones displayed on the input device
    self.fmt_type: lightonml.types.FeaturesFormat
        type of formatting used to map features to the input device
    self.fmt_factor: int
        size of the macro-elements used when formatting
    """
    def __init__(self, frametime: int = None, exposure: int = None,
                 output_roi: Roi = None,
                 start: dt.datetime = None, end: dt.datetime = None,
                 gain: float = None,
                 input_roi: Roi = None,
                 n_ones: int = None,
                 fmt_type: types.FeaturesFormat = None,
                 fmt_factor: int = None):
        self.info = None
        self.exposure_us = exposure    # type: int
        self.frametime_us = frametime  # type: int
        if output_roi:
            self.output_roi_upper = output_roi[0]  # type: Tuple2D # coordinates of upper output ROI
            self.output_roi_shape = output_roi[1]  # type: Tuple2D # shape of output ROI
        else:
            self.output_roi_shape = self.output_roi_upper = None
        if input_roi:
            self.input_roi_upper = input_roi[0]   # type: Tuple2D # coordinates of upper input ROI
            self.input_roi_shape = input_roi[1]   # type: Tuple2D # shape of output ROI
        else:
            self.input_roi_shape = self.input_roi_upper = None
        self.start = start         # type: dt.datetime # timestamp at start
        self.end = end             # type: dt.datetime # timestamp at end
        self.gain = gain           # output gain
        self.n_ones = n_ones
        self.fmt_type = fmt_type
        self.fmt_factor = fmt_factor

    def from_opu(self, opu, start: dt.datetime, end: dt.datetime = None):
        """Takes context from an OPU device, namely frametime, exposure,
        cam_roi and gain.
        With optional end time"""
        self.frametime_us = opu.frametime_us
        self.exposure_us = opu.exposure_us
        self.output_roi_upper, self.output_roi_shape = opu.output_roi
        self.gain = opu.gain_dB
        self.start = start         # type: dt.datetime # timestamp at start
        self.end = end             # type: dt.datetime # timestamp at end

    @staticmethod
    def from_timestamps(start, end=None):
        if end:
            end = from_epoch(end)
        return Context(-1, -1, ((-1, -1), (-1, -1)), from_epoch(start), end)

    @staticmethod
    def from_dict(d):
        """Create a context from a dict (flat or not)"""
        output_roi = d.get("output_roi") or d.get("ROI")
        if not output_roi and "roi_position_0" in d.keys():
            output_roi = ((d["roi_position_0"], d["roi_position_1"]),
                          (d["roi_shape_0"], d["roi_shape_1"]))

        if not output_roi:
            if "output_roi_position_0" in d.keys():
                output_roi = ((d["output_roi_position_0"], d["output_roi_position_1"]),
                              (d["output_roi_shape_0"], d["output_roi_shape_1"]))
            if "roi_position_0" in d.keys():
                output_roi = ((d["roi_position_0"], d["roi_position_1"]),
                              (d["roi_shape_0"], d["roi_shape_1"]))

        input_roi = d.get("input_roi")

        if not input_roi:
            if "input_roi_position_0" in d.keys():
                # From flat_dict
                input_roi = ((d["input_roi_position_0"], d["input_roi_position_1"]),
                             (d["input_roi_shape_0"], d["input_roi_shape_1"]))

        if 'fmt_type' in d.keys():
            # Convert back from string to enum
            fmt_type = types.FeaturesFormat[d["fmt_type"]]
        else:
            fmt_type = None
        context = Context(d['frametime_us'], d['exposure_us'], output_roi, d["start"],
                          d.get("end"), d.get("gain_db"), input_roi, d.get("n_ones"),
                          fmt_type, d.get("fmt_factor"))
        context.add_info(d.get("info"))
        return context

    def add_info(self, info):
        self.info = info

    def as_dict(self):
        result = {"exposure_us": self.exposure_us, "frametime_us": self.frametime_us,
                  "output_roi": (self.output_roi_upper, self.output_roi_shape),
                  "start": self.start}
        if self.gain is not None:
            result["gain_dB"] = self.gain
        if self.info is not None:
            result["info"] = self.info
        if self.end is not None:
            result["end"] = self.end
        if self.input_roi_shape is not None and self.input_roi_upper is not None:
            result["input_roi"] = (self.input_roi_upper, self.input_roi_shape)
        if self.n_ones is not None:
            result["n_ones"] = self.n_ones
        if self.fmt_type is not None:
            result["fmt_type"] = self.fmt_type.name
        if self.fmt_factor is not None:
            result["fmt_factor"] = self.fmt_factor

        return result

    def as_flat_dict(self):
        result = self.as_dict()
        output_roi = result.pop("output_roi", None)
        input_roi = result.pop("input_roi", None)
        if output_roi is not None:
            result["output_roi_position_0"] = output_roi[0][0]
            result["output_roi_position_1"] = output_roi[0][1]
            result["output_roi_shape_0"] = output_roi[1][0]
            result["output_roi_shape_1"] = output_roi[1][1]

        if input_roi is not None:
            result["input_roi_position_0"] = input_roi[0][0]
            result["input_roi_position_1"] = input_roi[0][1]
            result["input_roi_shape_0"] = input_roi[1][0]
            result["input_roi_shape_1"] = input_roi[1][1]

        return result

    def __str__(self):
        return str(self.as_dict())

    def __eq__(self, other):
        if not isinstance(other, Context):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.__dict__ == other.__dict__


class ContextArray(np.ndarray):
    """Array with additional 'context' attribute"""
    def __new__(cls, input_array, context: Context):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.context = context
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(ContextArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. ContextArray():
        #    obj is None
        #    (we're in the middle of the ContextArray.__new__
        #    constructor, and self.context will be set when we return to
        #    ContextArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(ContextArray):
        #    obj is arr
        #    (type(obj) can be ContextArray)
        # From new-from-template - e.g contextarr[:3]
        #    type(obj) is ContextArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'context', because this
        # method sees all creation of default objects - with the
        # ContextArray.__new__ constructor, but also with
        # arr.view(ContextArray).
        self.context = getattr(obj, 'context', None)
        # We do not need to return anything
