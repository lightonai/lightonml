# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lightonml.internal.device import OpuDevice


def rev(tuple_in):
    """convert back and forth between cartesian
    and (row, col) coordinates"""
    return tuple(reversed(tuple_in))


def try_or_giveup(func, n_tries, print_exc, *args, **kwargs):
    """Returns func() with trying n_times """
    count = 0
    while True:
        try:
            return func(*args, **kwargs), count
        except Exception as e:
            count += 1
            if count == n_tries:
                raise
            if print_exc:
                print(e)


def recurse_update(d, u):
    """
    Update a dictionary recursively
    https://stackoverflow.com/a/3233356/11343
    """
    import collections
    for k, v in u.items():
        # noinspection PyUnresolvedReferences
        if isinstance(v, collections.abc.Mapping):
            d[k] = recurse_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def count_ones(X, n_count, packed=False):
    """count avg number of ones in binary or bitpacked array
    X: 2D ndarray
    n_count: number of samples to take over the whole range
    """
    from lightonml.internal.popcount_u8 import popcount_u8

    sampled_X = random_samples(X, n_count)
    if packed:
        # if packed, return bitcounts
        total_count = popcount_u8(sampled_X)
    else:
        total_count = np.count_nonzero(sampled_X)
    if sampled_X.ndim == 1:
        return total_count
    return total_count / len(sampled_X)


def random_samples(X: np.ndarray, n_count):
    """Returns n_counts samples, randomly selected over X"""
    if X.ndim == 1:
        return X
    n_samples = X.shape[0]
    # count on arrays randomly selected over the whole range
    if n_samples > n_count:
        count_idx = np.random.randint(0, n_samples, n_count)
    else:
        count_idx = np.arange(0, n_samples)
    return X[(count_idx,)]


# noinspection PyPackageRequirements,PyUnresolvedReferences
def format_show(formatter, x, formatted_x, packed, plot=True):
    """Utility method to visualize a single vector with its formatted counterpart"""
    if formatter.n_features_s == 0:
        raise RuntimeError("Call configure first")
    if packed:
        x = np.unpackbits(x)
    x = x.reshape(formatter.n_features)
    if formatter.n_features.ndim == 1:
        x = [x]
    formatted_x = np.unpackbits(formatted_x).reshape(formatter.formatter_shape)
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(x, cmap='gray')
        axes[0].set_title('source')
        axes[1].imshow(formatted_x, cmap='gray')
        axes[1].set_title('formatted')
        plt.show()
    else:
        print("This is source array:\n", x)
        print("This is translated array into {} space: \n"
              .format(formatted_x.shape), formatted_x)


def binarize_output(out):
    """Get (0, 1) array from a batch of vectors, using threshold at half their mean"""
    out_mean = np.mean(out, axis=out.ndim - 1)
    # noinspection PyArgumentList
    threshold = (out_mean.max() - out_mean.min()) / 2
    return np.where(out_mean > threshold, 1, 0)


def random_bin(shape, dtype=bool):
    """Generates a random vector of 0s and 1s"""
    return np.random.randint(2, size=shape, dtype=dtype)


def alternate_full_sequence(n_vecs, n_features, fill_value=1):
    """Generates a batch of randomly alternating full/empty vectors
    n_vecs is batch size, n_features is number of features

    Returns tuple with the whole batch, and a vector of 0/1 meaning which
    vector is full or empty
    """
    # base pattern with 0 and 1s
    pattern_in = random_bin(n_vecs, dtype=np.uint8)
    # make a column vector
    pattern = pattern_in[:, np.newaxis] * fill_value
    # create 2D vector out of it for input to the OPU, with 100 features
    ins = np.repeat(pattern, n_features, axis=1)
    return ins, pattern_in


def gen_full_sequence(input_seq, n_features=100, fill_value=1):
    """Generates a batch of full/empty vectors based on a binary input sequence
    """
    # make it a column vector first
    pattern = input_seq[:, np.newaxis] * fill_value
    # create vectors of n_features out of it for input to the OPU
    return np.repeat(pattern, n_features, axis=1)


def blank_fn(*args, **kwargs):
    pass


class DeprecatedParam:
    def __init__(self, deprecated_args, version, reason):
        self.deprecated_args = set(deprecated_args.split())
        self.version = version
        self.reason = reason

    def __call__(self, a_callable):
        def wrapper(*args, **kwargs):
            found = self.deprecated_args.intersection(kwargs)
            if found:
                raise TypeError("Parameter(s) %s deprecated since version %s; %s" % (
                    ', '.join(map("'{}'".format, found)), self.version, self.reason))
            return a_callable(*args, **kwargs)

        return wrapper


def detect_trigger_issue(device: "OpuDevice"):
    """This function detects presence of the trigger issue in an OPU device"""
    # Generate batch of input frames, alternating 2 1s and 2 0s.
    n_samples = 200
    in_b = np.ones(n_samples, dtype=np.uint8)
    in_b[::4] = in_b[1::4] = 0
    # pattern = [0 0 1 1 0 0 1 1 0 0...], make frames out of it
    ins = gen_full_sequence(in_b, device.input_size, 0xff)
    out_size = int(np.prod(device.output_shape))
    out = np.empty((n_samples, out_size), dtype=device.output_dtype)

    # Acquire only n-1 output vectors, to avoid the timeout in the last acquisition
    device.transform2(ins, out, batch_index=-1, nb_acq_output=n_samples-1)
    out_b = binarize_output(out)
    # For comparison first remove last element from both arrays
    out_b = out_b[:-1]
    in_b = in_b[:-1]
    # No problem means arrays are equal, minus the last element
    if np.all(out_b == in_b):
        return False
    # Detection is true if we have out == [0 1 1 0 0 1 1 0 0...] (notice first one missing)
    if np.all(out_b[:-1] == in_b[1:]):
        return True
    # If here we're nowhere, so raise error
    raise RuntimeError("Trigger detection failed")
