# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

from lightonml.internal import utils
import numpy as np
import sys
import os


# noinspection PyUnresolvedReferences,PyPackageRequirements
def plot_hist_opu_output(x):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    n, bins, patches = plt.hist(x.flatten(), bins=range(256), density=True)
    plt.title('Distribution of output values')
    return fig, n, bins


def detect_low_intensity(x):
    mean_intensity = x.mean()
    print('Average intensity: {:.2f}'.format(mean_intensity))
    if mean_intensity < 11:
        print('Signal intensity is too low.')
    return mean_intensity


def detect_saturation(x):
    saturation_percent = (x == 255).mean() * 100
    print('Saturation: {:.2f} %'.format(saturation_percent))
    if saturation_percent > 0.5:
        print('Signal at the output is saturating.')
    return saturation_percent


def profile_opu_output(x):
    """Runs different checks on the output of the OPU, to detect if there is
    enough signal or the presence of saturation.

    Parameters
    ----------
    x: np.ndarray,
        output array from the OPU
    """
    if np.any(x < 0) or np.any(x > 255):
        raise ValueError('x is not the output of an OPU. Some values are '
                         'outside of the [0, 255] range.')
    if x.ndim == 1:
        x = x[np.newaxis, :]

    print('A histogram of values is going to be displayed. If it is too squashed '
          'close to the origin, you probably need to increase the signal. '
          'If there is a peak at x=255, then you need to decrease it.')
    if display_available():
        fig, _, _ = plot_hist_opu_output(x)
        fig.show()

    detect_low_intensity(x)
    detect_saturation(x)


def show_output_image(opu, x):
    """From a given x input vector, show the raw image transformed by the OPU.

    Parameters
    ----------
    opu: OPU,
        instance of OPU.
    x: np.ndarray of [0, 1], dtype uint8,
        sample to display on the input device.

    Returns
    -------
    y: np.ndarray of uint8,
        output image.
    """
    # don't cut output to n_components
    if x.ndim == 1:
        y = opu.fit_transform1d(x, raw_output_size=True)
    elif x.ndim == 2:
        y = opu.fit_transform2d(x, raw_output_size=True)
    else:
        raise ValueError("Input must be a 1d or 2d vector")

    # Then reshape (invert coordinate systems)
    y.shape = utils.rev(y.context.output_roi_shape)
    print('Showing an output image; if you see artifacts, it could be an effect of '
          'a low signal level. ')

    if display_available():
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import matplotlib.pyplot as plt
        plt.imshow(y)
        plt.colorbar()
        plt.show()
    return y


def in_notebook():
    # True if the module is running in IPython kernel, False if in shell
    return 'ipykernel' in sys.modules


def display_available():
    if 'DISPLAY' in os.environ or in_notebook():
        return True
    else:
        print('NO $DISPLAY environment available. Open ssh connection with '
              'the -Y option or run from Jupyter notebook.')
        return False
