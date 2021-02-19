"""
Copyright (c) 2020 LightOn, All Rights Reserved.
This file is subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.

Module containing enums used with opu.OPU class
"""

from enum import Enum
from typing import List, Tuple, Union
import numpy as np


class OutputRoiStrategy(Enum):
    """Strategy used for computing the output ROI"""
    mid_width = 1
    """Area in the middle & max_width, to have max speed (Zeus, Vulcain)"""
    mid_square = 2
    """Area in the middle & square (Saturn)"""


class InputRoiStrategy (Enum):
    """Strategy used for computing the input ROI"""
    full = 1
    """Apply zoom on elements to fill the whole display"""
    small = 2
    """Center the features on the display, with one-to-one element mapping"""
    auto = 3
    """Try to find the most appropriate between these two modes"""


class FeaturesFormat (Enum):
    """Strategy used for the formatting of data on the input device"""
    lined = 1
    """Features are positioned in line"""
    macro_2d = 2
    """Features are zoomed into elements"""
    auto = 3
    """Automatic choice
    
    `lined` if features are 1d, `macro_2d` if 2d
    """
    none = 4
    """No formatting
    
    input is displayed as-is, but it must match the same number of 
    elements of the input device"""


IntList = Union[List[int], np.ndarray]
Tuple2D = Tuple[int, int]
Roi = Tuple[Tuple2D, Tuple2D]
# Either a int, or a tuple or ints
IntOrTuple = Union[int, Tuple[int, ...]]

from lightonml.context import ContextArray

TransformOutput = Union[ContextArray, "Tensor"]
