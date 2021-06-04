# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from lightonml.internal.types import InputRoiStrategy, Roi, IntList, Tuple2D
import numpy as np
from typing import Tuple, Union, Optional
import warnings


class InputRoi:
    """
    Provides functions to determine the ROI to choose for the OPU input device
    """
    def __init__(self, input_shape: Tuple2D, ones_range: IntList):
        """
        Constructor

        Parameters
        ----------
        input_shape: tuple(int)
            shape of the input, in cartesian coordinates
        ones_range: tuple(int)
            list representing [min, max] number of ones in order to get
            satisfying signal on the output
        """
        self.input_shape = np.asarray(input_shape)
        self.input_size = np.prod(self.input_shape)
        self.shape_ratio = input_shape[0] / input_shape[1]
        assert len(input_shape) == 2
        assert len(ones_range) == 2 \
            and ones_range[0] < ones_range[1] <= self.input_size
        self.ones_range = ones_range

    # noinspection PyDefaultArgument
    def check_nb_ones(self, n_ones: int, 
                      size: Union[int, Tuple2D, np.ndarray],
                      checked_out: dict = {}) -> float:
        """
        Find if the number of ones to be displayed will give proper signal
        on the output
i
        Parameters
        ----------
        n_ones: int
             number of ones in the input data
        size: int or tuple(int) or ndarray(int)
             either total number of elements, or the size of the ROI
        checked_out: dict
            dict filled with additional info (value and total)

        Returns
        -------
        answer: float
             * -1 if not enough ones
             * 0 if enough
             * ratio over the maximum allowed, if too much (will saturate)
             (this allows to lower exposure in the TransformRunner)
        """
        # number of ones must be added to the size of the outside ROI,
        # since it's padded with ones
        total_n_ones = self.input_size - np.prod(size) + int(n_ones)
        # range[0] is lower range, range[1] is higher range
        if total_n_ones < self.ones_range[0]:
            value = -1
        elif total_n_ones <= self.ones_range[1]:
            value = 0
        # case total_n_ones > range[1]: return the ratio of excess
        else:
            value = total_n_ones/self.ones_range[1]
        checked_out["value"] = value
        checked_out["total"] = total_n_ones
        return value

    # noinspection PyDefaultArgument
    def compute_roi(self, strategy: InputRoiStrategy, n_features,
                    n_ones: Optional[int] = None, checked_out: dict = {}) -> Roi:
        """
        This method *tries* to find the best ROI for the given features

        Parameters
        ----------
        strategy: InputRoiStrategy
             enum for the strategy to use
                * full: use macro-elements to fill the whole display
                * small: center the features on the display, with one-to-one mapping
                * auto: try to find the most appropriate between these two modes
        n_features: int or tuple(int)
             number of elements in the input data (can be 2D)
        n_ones: int, optional
             number of ones in the input data, useful for auto strategy
        checked_out: dict, optional
            will be filled with the result of self.check_nb_ones if strategy is auto

        Returns
        -------
        roi: offset and size for the chosen ROI
        """
        if isinstance(n_features, (tuple, np.ndarray)):
            n_features = np.asarray(n_features)
        else:
            n_features = np.asarray([n_features])
        # sanity checks
        assert n_features.ndim == 1
        assert len(n_features) <= 2
        assert (n_features > 0).all()

        if strategy is InputRoiStrategy.auto:
            if n_ones is None:
                raise ValueError("In auto mode should be provided the number of ones")
            if np.prod(n_features) > self.input_size/2:
                # in this case we can't use macro-elements
                # use "small" strategy, which is just mapping features
                # one-to-one on the display
                roi = self._small_roi(n_features)
                self.check_nb_ones(n_ones, n_features, checked_out)
                return self._check_roi(roi)
            else:
                # use macro-elements to try to have higher number of ones
                roi, factor = self._macro_roi(n_features)
                # check that nb_ones is ok: the new count is multiplied by factor
                if self.check_nb_ones(n_ones * factor, roi[1], checked_out) >= 0:
                    return self._check_roi(roi)
                # if not ok, reduce factor progressively, in order to get more ones
                else:
                    for factor in reversed(range(factor)):
                        roi, _ = self._macro_roi(n_features, factor)
                        if self.check_nb_ones(n_ones * factor, roi[1], checked_out) >= 0:
                            return self._check_roi(roi)
                    raise ValueError("No satisfying roi could be found, input data"
                                     " might be too sparse.")

        elif strategy is InputRoiStrategy.small:
            roi = self._small_roi(n_features)
            # Check the number of ones
            if n_ones is not None and self.check_nb_ones(n_ones, roi[1], checked_out) < 0:
                warnings.warn("The input data is too sparse for the input device.\n"
                              "However the linear transform isn't impacted by data sparsity.")
            return self._check_roi(roi)

        elif strategy is InputRoiStrategy.full:
            roi, factor = self._macro_roi(n_features, -1)
            # Check the number of ones
            if n_ones is not None and self.check_nb_ones(n_ones*factor, roi[1], checked_out) < 0:
                warnings.warn("The input data is too sparse for the input device.\n"
                              "However the linear transform isn't impacted by data sparsity.")
            return self._check_roi(roi)

        else:
            assert False

    def _macro_roi(self, n_features, factor=-1) -> Tuple[Roi, int]:
        """Get a ROI using macro-elements, 2D or 1D.
        factor at -1 will get the biggest macro-elements
        Returns (roi, macro-element size)
        """
        # Determine macro-elements size
        if len(n_features) == 2:
            # 2D macro-elements, determine macro-element's side size
            if factor == -1:
                # take the minimum between 2D ratios
                factor = min(self.input_shape // n_features)
            roi_size = np.multiply(n_features, factor)
            offset = self.input_shape // 2 - roi_size // 2
        else:
            n_features = n_features[0]
            if factor == -1:
                # lined formatting, just use the whole shape
                roi_size = self.input_shape
                offset = [0, 0]
                factor = self.input_size // n_features
            else:
                # find ROI with original aspect ratio that fits the number of features
                height = np.sqrt(n_features * factor / self.shape_ratio)
                width = self.shape_ratio * height

                # get ceil and convert to int
                roi_size = np.ceil(np.array([width, height])).astype(int)
                offset = self.input_shape // 2 - roi_size // 2

        return (offset, roi_size), factor

    def _check_roi(self, roi: Roi) -> Roi:
        if (np.add(roi[0], roi[1]) > self.input_shape).any():
            raise ValueError(f"ROI {roi} doesn't fit in input size {self.input_shape}")
        if np.any(np.asarray(roi[0]) < 0):
            raise ValueError(f"ROI {roi} has negative offset")
        return tuple(roi[0]), tuple(roi[1])

    def _small_roi(self, n_features) -> Roi:
        size = self._get_2d_size(n_features, self.input_shape)
        # offset is set so that the ROI is centered
        offset = self.input_shape // 2 - size // 2
        return tuple(offset), tuple(size)

    @staticmethod
    def _get_2d_size(n_elements: IntList, shape: IntList) -> IntList:
        """
        Return shape of a square that can contain n_elements

        if n_elements is already 2D, return it as is
        """
        if len(n_elements) == 2:
            if (n_elements > shape).any():
                raise ValueError(f"2D features {n_elements} don't fit in input shape {shape}")
            return n_elements
        else:
            if n_elements > np.prod(shape):
                raise ValueError(f"1D features {n_elements} don't fit in input shape {shape}")
            # if 1D input, make it a square with correct size
            side = int(np.ceil(np.sqrt(n_elements)))
            small_side = np.min(shape)
            if side <= small_side:
                return np.array([side, side])
            else:
                # if a square doesn't fit shape's smaller side,
                # use a rectangular shape whose small side is the smaller side
                small_index = np.argmin(shape)
                large_index = (small_index+1) % 2
                ret = np.zeros(2, dtype=int)
                ret[small_index] = small_side
                ret[large_index] = np.ceil(n_elements / small_side)
                return ret
