# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from lightonml.internal.types import OutputRoiStrategy, Roi, Tuple2D
import numpy as np


class OutputRoi:
    """
    Provides functions to determine the ROI to choose for the output device
    """
    def __init__(self, output_shape: Tuple2D,
                 strategy: OutputRoiStrategy, allowed_roi: Roi = None,
                 minimum_components: int = 0):
        """
        Constructor

        Parameters
        ----------
        output_shape: tuple(int)
            cartesian shape for the output
        strategy: OutputRoiStrategy
        allowed_roi: tuple(tuple(int))
            (offset, size) for an allowed ROI
        """
        self.strategy = strategy
        self.allowed_roi = allowed_roi or ((0, 0), output_shape)
        shape = self.allowed_roi[1]
        self.aspect_ratio = shape[0] / shape[1]
        self.min_components = minimum_components
        self.max_components = int(np.prod(self.allowed_roi[1]))  # type: int

        # check that boundaries of allowed ROI don't out fit output shape
        total_allowed = np.add(self.allowed_roi[0], self.allowed_roi[1])
        assert np.less_equal(total_allowed, output_shape).all()
        # If we fill the whole output width, we must be sure that
        # X offset is 0
        if self.strategy is OutputRoiStrategy.mid_width:
            assert self.allowed_roi[0][0] == 0
        assert self.min_components < self.max_components

    def compute_roi(self, n_components) -> Roi:
        """Compute the output offset and ROI, with ROI being large enough
        to contain n_components

        Returns
        -------
        answer: tuple(list(int))
            tuple containing offset and size, in cartesian coordinates
        """
        allowed_offset = np.array(self.allowed_roi[0])
        allowed_size = np.array(self.allowed_roi[1])

        # minimum value for n_components
        n_components = max(self.min_components, n_components)

        if n_components > self.max_components:
            raise IndexError("n_components is beyond maximum ({})"
                             .format(self.max_components))

        if self.strategy is OutputRoiStrategy.mid_square:
            # find the dimensions with the same form factor
            height = np.sqrt(n_components / self.aspect_ratio)
            width = self.aspect_ratio * height

            # get ceil and convert to int
            size = np.ceil(np.array([width, height])).astype(int)

            # center the ROI
            offset = allowed_offset + (allowed_size//2 - size//2)
        elif self.strategy is OutputRoiStrategy.mid_width:
            height = int(np.ceil(n_components/allowed_size[0]))
            size = [allowed_size[0], height]
            offset = [0, int(allowed_size[1]/2 - height/2)]
        else:
            assert False
        # check if it fits
        assert np.less_equal(size, allowed_size).all()
        # check if it's sufficient
        assert np.prod(size) >= n_components
        return tuple(offset), tuple(size)
