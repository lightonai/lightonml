# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from attr import attrs, attrib

from lightonml.internal import types
from lightonml.internal.types import Roi


@attrs(frozen=True)
class OpuSettings(object):
    """OpuSettings hold immutable settings of an OPU, like input and output shape"""
    # Maximum number of features
    max_n_features = attrib(type=int)

    # Shapes are in cartesian coordinates
    # (e.g. (912, 1140) for Model1)
    """Shape of the input device"""
    input_shape = attrib(type=tuple)  # Shape of the input device
    output_max_shape = attrib(type=tuple)   # Maximum shape of the output device

    # base frametime and exposure, usually in lighton.json.
    # can be overridden by manual timings, or automatic transform setup
    frametime_us = attrib(type=int, default=0)
    exposure_us = attrib(type=int, default=0)

    # Allowed ROI on the output device: None if non-existent
    allowed_roi = attrib(type=Roi, default=None)
    # Is it a real or simulated OPU
    simulated = attrib(type=bool, default=False)
    # Number of samples passed at each iteration of a batch run
    n_samples_by_pass = attrib(type=int, default=3000)

    # number of samples to take in a batch when counting ones
    n_count = attrib(type=int, default=1000)
    # Minimum roi size for the output device
    min_n_components = attrib(type=int, default=0)

    # Minimum number of vectors for the batch operation
    min_batch_size = attrib(type=int, default=0)

    # list representing [min, max] number of ones on the input device
    # in order to get satisfying signal on the output device
    ones_range = attrib(factory=list)
    # number of times a transform must be retried before returning error
    n_tries = attrib(type=int, default=1)
    # Detect trigger issues at OPU open
    detect_trigger = attrib(type=bool, default=False)
    # If single transform isn't working, use batch instead
    no_single_transform = attrib(type=bool, default=False)
    # stdev of the random features, for the output rescaling
    stdev = attrib(type=float, default=1.)


@attrs
class TransformSettings(object):
    """Settings for a transform"""
    input_roi_strategy = attrib(type=types.InputRoiStrategy)
    n_components = attrib(type=int)

    # override input ROI as (offset, size)
    input_roi = attrib(type=Roi, default=None)
    # If True, don't cut output size at n_components
    raw_output_size = attrib(type=bool, default=False)
    # If non-zero, override exposure time for a transform
    exposure_us = attrib(type=int, default=0)
    # If non-zero, override frametime time for a transform
    frametime_us = attrib(type=int, default=0)
