# -*- coding: utf8
"""
Encoders
--------------------
These modules contains implementations of Encoders that can transform data
in the binary `uint8` format required by the OPU. Compatible with numpy.ndarray
and torch.Tensor.
"""
import numexpr as ne
import numpy as np
import functools


def _tensor_to_array(X):
    is_tensor = type(X).__name__ == 'Tensor'
    X = X.numpy() if is_tensor else X
    return X, is_tensor


def _array_to_tensor(X, is_tensor):
    if is_tensor:
        import torch
        return torch.from_numpy(X)
    else:
        return X


def preserve_type(transform):
    @functools.wraps(transform)
    def wrapper(instance, X, *args, **kwargs):
        X, is_tensor = _tensor_to_array(X)
        Z = transform(instance, X, *args, **kwargs)
        return _array_to_tensor(Z, is_tensor)
    return wrapper


class BaseTransformer:
    """
    Base class for all basic encoders and decoders.
    Mainly for avoiding empty fit methods and provide an automatic fit_transform method."""

    def fit(self, X, y=None):
        """
        No-op, exists for compatibility with the scikit-learn API.

        :param X: 2D np.ndarray or torch.Tensor
        :param y: 1D np.ndarray or torch.Tensor
        :return: Encoder object 
        """
        return self

    def transform(self, X):
        """
        Function to encode or decode an array X.

        :param X: 2D np.ndarray or torch.Tensor
        :return: 2D np.ndarray or torch.Tensor of uint8
        """
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ConcatenatedBitPlanEncoder(BaseTransformer):
    """Implements an encoding that works by concatenating bitplanes along the feature dimension.

    ``n_bits + starting_bit`` must be lower than the bitwidth of data that are going to be fed
    to the encoder.
    E.g. if ``X.dtype`` is ``uint8``, then ``n_bits + starting_bit`` must be lower than 8.
    If instead ``X.dtype`` is ``uint32``, then ``n_bits + starting_bit`` must be lower than 32.

    Read more in the Examples section.

    Parameters
    ----------
    n_bits: int, defaults to 8,
        number of bits to keep during the encoding. Must be positive.
    starting_bit: int, defaults to 0,
        bit used to start the encoding, previous bits will be thrown away. Must be positive.

    Attributes
    ----------
    n_bits: int,
        number of bits to keep during the encoding.
    starting_bit: int,
        bit used to start the encoding, previous bits will be thrown away.

    """
    def __init__(self, n_bits=8, starting_bit=0):
        if n_bits <= 0:
            raise ValueError('n_bits must be a positive integer.')
        if starting_bit < 0:
            raise ValueError('starting_bit must be 0 or a positive integer.')
        super(ConcatenatedBitPlanEncoder, self).__init__()
        self.n_bits = n_bits
        self.starting_bit = starting_bit

    @preserve_type
    def transform(self, X):
        """Performs the encoding.

        Parameters
        ----------
        X : 2D np.ndarray of uint8, 16, 32 or 64 [n_samples, n_features],
            input data to encode.

        Returns
        -------
        X_enc: 2D np.ndarray of uint8 [n_samples, n_features*n_bits]
            encoded input data. A line is arranged as [bits_for_first_feature, ..., bits_for_last_feature].

        """

        bitwidth = X.dtype.itemsize*8

        if self.n_bits+self.starting_bit > bitwidth:
            raise ValueError('n_bits + starting_bit is greater than bitwidth of input data: '
                             '{}+{} > {}'.format(self.n_bits, self.starting_bit, bitwidth))

        n_samples, n_features = X.shape
        # add a dimension [n_samples, n_features, 1] and returns a view of the data as uint8
        X_uint8 = np.expand_dims(X, axis=2).view(np.uint8)

        # Unpacks the bits along the auxiliary axis
        X_uint8_unpacked = np.unpackbits(X_uint8, axis=2)

        X_enc = X_uint8_unpacked[:, :, self.starting_bit:self.n_bits + self.starting_bit]\
            .reshape((n_samples, self.n_bits*n_features))

        return X_enc


class ConcatenatingBitPlanDecoder(BaseTransformer):
    """Implements a decoding that works by concatenating bitplanes.

    ``n_bits`` MUST be the same value used in SeparatedBitPlanEncoder.
    Read more in the Examples section.

    Parameters
    ----------
    n_bits: int, defaults to 8,
        number of bits used during the encoding.
    decoding_decay: float, defaults to 0.5,
        decay to apply to the bits during the decoding.

    Attributes
    ----------
    n_bits: int,
        number of bits used during the encoding.
    decoding_decay: float, defaults to 0.5,
        decay to apply to the bits during the decoding.

    """
    def __init__(self, n_bits=8, decoding_decay=0.5):
        self.n_bits = n_bits
        self.decoding_decay = decoding_decay

    @preserve_type
    def transform(self, X):
        """Performs the decoding.

        Parameters
        ----------
        X : 2D np.ndarray of uint8 or uint16,
            input data to decode.

        Returns
        -------
        X_dec : 2D np.ndarray of floats
            decoded data.
        """
        n_out, n_features = X.shape
        n_dim_0 = int(n_out / self.n_bits)
        X_dec = np.zeros(shape=(n_dim_0, self.n_bits*n_features), dtype='float32')

        if n_dim_0*self.n_bits*n_features != X.size:
            raise ValueError('Check that you used the same number of bits in encoder and decoder.')

        X = np.reshape(X, (n_dim_0, self.n_bits, n_features))
        decay_factors = np.reshape(self.decoding_decay ** np.arange(self.n_bits), (1, self.n_bits, 1))
        X = X * decay_factors
        X_dec[:] = np.reshape(X, (n_dim_0, self.n_bits * n_features))
        return X_dec


# noinspection PyPep8Naming
class Float32Encoder(BaseTransformer):
    """Implements an encoding that works by separating bitplans and selecting how many bits
    to keep for sign, mantissa and exponent of the float32.

    Parameters
    ----------
    sign_bit: bool, defaults to True,
        if True keeps the bit for the sign.
    exp_bits: int, defaults to 8,
        number of bits of the exponent to keep.
    mantissa_bits: int, defaults to 23,
        number of bits of the mantissa to keep.
    Attributes
    ----------
    sign_bit: bool, defaults to True,
        if True keeps the bit for the sign.
    exp_bits: int, defaults to 8,
        number of bits of the exponent to keep.
    mantissa_bits: int, defaults to 23,
        number of bits of the mantissa to keep.
    n_bits: int,
        total number of bits to keep.
    indices: list,
        list of the indices of the bits to keep.
    """

    def __init__(self, sign_bit=True, exp_bits=8, mantissa_bits=23):
        if exp_bits < 0 or exp_bits > 8:
            raise ValueError('exp_bits must be in the range [0, 8]')
        if mantissa_bits < 0 or mantissa_bits > 23:
            raise ValueError('mantissa_bits must be in the range [0, 23]')
        self.sign_bit = sign_bit
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.n_bits = int(sign_bit) + exp_bits + mantissa_bits
        indices = list(range(1, self.exp_bits + 1)) + list(range(9, self.mantissa_bits + 9))
        if self.sign_bit:
            indices = indices + [0]
        self.indices = sorted(indices)

    @preserve_type
    def transform(self, X):
        """Performs the encoding.

        Parameters
        ----------
        X : 2D np.ndarray of float32 [n_samples, n_features],
            input data to encode.

        Returns
        -------
        X_enc: 2D np.ndarray of uint8 [n_samples*n_bits, n_features],
            encoded input data.
        """
        n_samples, n_features = X.shape
        # create a new axis and separate the binary representation into 4 uint8
        X_uint8 = np.expand_dims(X, axis=2).view('uint8')
        # reverse the ordering of the uint8 before unpacking
        # it is not done the other way around because float32 and uint8 don't read the bits in the
        # same direction
        X_uint8_reversed = np.flip(X_uint8, axis=2)
        X_bits = np.unpackbits(X_uint8_reversed, axis=2)
        # select the bits we asked to keep
        X_enc = np.transpose(X_bits, [0, 2, 1])
        X_enc = X_enc[:, self.indices, :].reshape(n_samples * self.n_bits, n_features)
        return X_enc


class ConcatenatedFloat32Encoder(BaseTransformer):
    """Implements an encoding that works by concatenating bitplanes and selecting how many bits
    to keep for sign, mantissa and exponent of the float32.

    Parameters
    ----------
    sign_bit: bool, defaults to True,
        if True keeps the bit for the sign.
    exp_bits: int, defaults to 8,
        number of bits of the exponent to keep.
    mantissa_bits: int, defaults to 23,
        number of bits of the mantissa to keep.
    Attributes
    ----------
    sign_bit: bool, defaults to True,
        if True keeps the bit for the sign.
    exp_bits: int, defaults to 8,
        number of bits of the exponent to keep.
    mantissa_bits: int, defaults to 23,
        number of bits of the mantissa to keep.
    n_bits: int,
        total number of bits to keep.
    indices: list,
        list of the indices of the bits to keep.
    """

    def __init__(self, sign_bit=True, exp_bits=8, mantissa_bits=23):
        if exp_bits < 0 or exp_bits > 8:
            raise ValueError('exp_bits must be in the range [0, 8]')
        if mantissa_bits < 0 or mantissa_bits > 23:
            raise ValueError('mantissa_bits must be in the range [0, 23]')
        super(ConcatenatedFloat32Encoder, self).__init__()
        self.sign_bit = sign_bit
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.n_bits = int(sign_bit) + exp_bits + mantissa_bits
        indices = list(range(1, self.exp_bits + 1)) + list(range(9, self.mantissa_bits + 9))
        if self.sign_bit:
            indices = indices + [0]
        self.indices = sorted(indices)

    @preserve_type
    def transform(self, X):
        """Performs the encoding.

        Parameters
        ----------
        X : 2D np.ndarray of float32 [n_samples, n_features],
            input data to encode.

        Returns
        -------
        X_enc: 2D np.ndarray of uint8 [n_samples*n_bits, n_features],
            encoded input data.
        """
        n_samples, n_features = X.shape
        # create a new axis and separate the binary representation into 4 uint8
        X_uint8 = np.expand_dims(X, axis=2).view('uint8')
        # reverse the ordering of the uint8 before unpacking
        # it is not done the other way around because the 4 bytes of float32 are stored
        # one after the other, it's not one block of 32 bits
        X_uint8_reversed = np.flip(X_uint8, axis=2)
        X_bits = np.unpackbits(X_uint8_reversed, axis=2)
        # select the bits we asked to keep
        X_enc = X_bits[:, :, self.indices].reshape(n_samples, self.n_bits * n_features)
        return X_enc


# noinspection PyPep8Naming
class BinaryThresholdEncoder(BaseTransformer):
    """Implements binary encoding using a threshold function.

    Parameters
    ----------
    threshold_enc : int or str
        Threshold for the binary encoder. Default 0.
        'auto' will set threshold_enc to feature-wise median of the data passed to the fit function.
    greater_is_one : bool
        If True, above threshold is 1 and below 0. Vice versa if False.

    Attributes
    ----------
    threshold_enc : int or str
        Threshold for the binary encoder.
    greater_is_one : bool
        If True, above threshold is 1 and below 0. Vice versa if False.

    """
    def __init__(self, threshold_enc='auto', greater_is_one=True):
        if not (isinstance(threshold_enc, float) or isinstance(threshold_enc, int) or threshold_enc == 'auto'):
            raise ValueError("Argument threshold_enc should be a number or 'auto'.")
        self.threshold_enc = threshold_enc
        self.greater_is_one = greater_is_one

    def fit(self, X, y=None):
        """
        When threshold_enc is 'auto', this method sets it to a vector containing the median of each column of X.
        Otherwise, it does nothing except print a warning in case threshold_enc is not in the range covered by X.

        Parameters
        ----------
        X : np.ndarray,
            the input data to encode.
        y : np.ndarray,
            the targets data.

        Returns
        -------
        self : BinaryThresholdEncoding
        """
        if isinstance(self.threshold_enc, str):
            # noinspection PyTypeChecker
            self.threshold_enc = np.median(X, axis=0)  # the median is feature-wise
        else:
            if self.threshold_enc < X.min() or self.threshold_enc > X.max():
                print('WARNING: encoder threshold is outside data range')
        return self

    @preserve_type
    def transform(self, X):
        """Transforms any numpy array in a uint8 binary array of [0, 1].

        Parameters
        ----------
        X : np.ndarray  or torch.Tensor
            the input data to encode.

        Returns
        -------
        X_enc : np.ndarray  or torch.Tensor
                uint8 containing only zeros and ones the encoded data.
        """
        if isinstance(self.threshold_enc, str):
            raise RuntimeError("If threshold_enc is 'auto', fit must be called before transform.")

        if self.greater_is_one:
            X_enc = (X > self.threshold_enc)
        else:
            X_enc = (X < self.threshold_enc)
        return X_enc.astype(np.uint8)


class MultiThresholdEncoder(BaseTransformer):
    """Implements binary encoding using multiple thresholds.

    Parameters
    ----------
    thresholds : list, np.ndarray or str
        thresholds for the binary encoder. If a list or an array is passed, the thresholds will be used unmodified.
        If thresholds='linspace', the values will be evenly distributed along the data range.
        If thresholds='quantile', the values will be set to the quantiles corresponding to n_bins.
        If n_bins=4, the thresholds will be the 1st, 2nd and 3rd quartiles.
    columnwise: bool,
        whether to use different thresholds for each column or a common set of thresholds for everything.
    n_bins: int,
        if `thresholds` is 'linspace' or 'quantiles', `n_bins - 1` thresholds will be created?

    Attributes
    ----------
    thresholds : np.ndarray,
        thresholds for the binary encoder.
    columnwise: bool,
        whether to use different thresholds for each column or a common set of thresholds for everything.
    n_bins: int,
        number of different values the encoding can take. A value is encoded into n_bins-1 bits.
    """

    def __init__(self, thresholds='linspace', n_bins=8, columnwise=False):

        if isinstance(thresholds, list) or isinstance(thresholds, np.ndarray):
            thresholds = np.array(thresholds)
            if columnwise and thresholds.ndim != 2:
                raise ValueError("""
                You set columnwise to True but thresholds is 1D.
                Pass a 2D array or set columnwise to False.
                """)
            self.n_bins = thresholds.shape[-1] + 1
        elif thresholds in ['linspace', 'quantiles']:
            self.n_bins = n_bins
        else:
            raise ValueError("Argument thresholds must be a list, a numpy array or be in ['linspace', 'quantiles'].")

        self.thresholds = thresholds
        self.columnwise = columnwise

    def fit(self, X, y=None):
        """If thresholds is not None, this method doesn't do anything.
        If thresholds is `None`, computes `n_bins` thresholds equally spaced on the range of `X`.
        The range of `X` is determined column-wise but the number of bins is the same for all features.

        Parameters
        ----------
        X : 2D np.ndarray
        y: 1D np.ndarray

        Returns
        -------
        self : MultiThresholdEncoder
        """
        def set_thresholds(array):
            if self.thresholds == 'linspace':
                return np.linspace(array.min(), array.max(), self.n_bins + 1)[1:-1]
            elif self.thresholds == 'quantiles':
                k = 1 / self.n_bins
                quantiles = [np.quantile(array, i*k, interpolation='midpoint') for i in range(1, self.n_bins)]
                return np.array(quantiles)

        if self.thresholds in ['linspace', 'quantiles']:
            if self.columnwise:
                thresholds = np.empty((X.shape[1], self.n_bins - 1), dtype='float32')
                for col in range(X.shape[1]):
                    column = X[:, col]
                    column_thresholds = set_thresholds(column)
                    thresholds[col] = column_thresholds
                self.thresholds = thresholds
            else:
                self.thresholds = set_thresholds(X)
        return self

    @preserve_type
    def transform(self, X):
        """Transforms an array to a uint8 binary array of [0, 1].

        The bins defined by the thresholds are not mutually exclusive, i.e a value x will activate
        all the bins corresponding to thresholds lesser than x.

        Parameters
        ----------
        X : np.ndarray of size n_sample x n_features
            The input data to encode.

        Returns
        -------
        X_enc : np.ndarray of uint8, of size n_samples x (n_features x n_bins)
            The encoded data.
        """
        n_thres = self.thresholds.shape[-1]
        # we don't modify self.thresholds directly to make sure multiple calls will not cause a problem
        if self.thresholds.ndim == 1:
            thresholds = self.thresholds[:, None, None]
        elif self.thresholds.ndim == 2:
            thresholds = self.thresholds.T[:, None, :]
        else:
            raise RuntimeError("Unexpected threshold.ndim " + self.thresholds.ndim)
        X = np.repeat(X[None, ...], n_thres, axis=0)
        X = X > thresholds
        X_enc = np.concatenate([X[i] for i in range(thresholds.shape[0])], axis=-1).astype('uint8')
        return X_enc


# noinspection PyPep8Naming
class SequentialBaseTwoEncoder(BaseTransformer):
    """Implements a base 2 encoding.

    E.g. :math:`5` is written :math:`101` in base 2: :math:`1 * 2^2 + 0 * 2^1 + 1 * 2^0` = (1)*4 +(0)*2 +(1)*1, so the
    encoder will give 1111001.

    Parameters
    ----------
    n_gray_levels : int,
        number of values that can be encoded. Must be a power of 2.

    Attributes
    ----------
    n_gray_levels : int,
        number of values that can be encoded. Must be a power of 2.
    n_bits : int,
        number of bits needed to encode n_gray_levels values.
    offset : float,
        value to subtract to get the minimum to 0.
    scale : float,
        scaling factor to normalize the data.

    """
    def __init__(self, n_gray_levels=16):
        assert type(n_gray_levels) == int, 'n_gray_levels must be an integer power of 2'
        assert ((n_gray_levels & (n_gray_levels - 1)) == 0) and n_gray_levels > 0, \
            'n_gray_levels must be an integer power of 2'
        self.n_gray_levels = n_gray_levels
        self.n_bits = np.uint8(np.log2(self.n_gray_levels))
        self.n_bits_type = 8
        self.indices_axis_2 = np.arange(self.n_bits_type - self.n_bits, self.n_bits_type)
        self.offset = None
        self.scale = None

    def fit(self, X, y=None):
        """Computes parameters for the normalization.

        Must be run only on the training set to avoid leaking information to the dev/test set.

        Parameters
        ----------
        X : np.ndarray of uint [n_samples, n_features],
            the input data to encode.
        y : np.ndarray,
            the targets data.

        Returns
        -------
        self : SequentialBaseTwoEncoder.

        """
        X, is_tensor = _tensor_to_array(X)
        self.offset = np.min(X)
        self.scale = np.max(X - self.offset)
        return self

    def normalize(self, X):
        """Normalize the data in the right range before the integer casting.

        Parameters
        ----------
        X : np.ndarray of uint [n_samples, n_features],
            the input data to normalize.

        Returns
        -------
        X_norm : np.ndarray of uint8 [n_samples, n_features],
            normalized data.

        """
        assert_msg = 'You have to call fit on the training data before calling transform.'
        assert self.offset is not None, assert_msg
        assert self.scale is not None, assert_msg
        # Data normalization
        X_norm = ((self.n_gray_levels - 1) * (X - self.offset)) / self.scale
        X_norm = np.round(X_norm)
        # Force the data is in the good range
        X_norm[X_norm < 0] = 0
        X_norm[X_norm > (self.n_gray_levels - 1)] = (self.n_gray_levels - 1)
        # Cast to uint8
        X_norm = X_norm.astype(np.uint8)
        return X_norm

    @preserve_type
    def transform(self, X):
        """Performs the encoding.

        Parameters
        ----------
        X : 2D np.ndarray of uint [n_samples, n_features],
            input data to encode.

        Returns
        -------
        X_enc: 2D np.ndarray of uint8 [n_samples, n_features*(n_gray_levels-1)
            encoded input data.

        """
        n_samples, n_features = X.shape
        X = self.normalize(X)
        # Expand bits along auxiliary axis
        X_bits = np.unpackbits(np.expand_dims(X, axis=2), axis=2)
        # Repeat each bit value for the corresponding power of 2
        X_enc = np.repeat(X_bits[:, :, self.indices_axis_2], 2 ** np.arange(self.n_bits)[::-1], axis=2)
        X_enc = X_enc.reshape((n_samples, n_features * (2 ** self.n_bits - 1)))
        return X_enc


class NoEncoding(BaseTransformer):
    """Implements a No-Op Encoding class for API consistency."""
    def transform(self, X):
        return X


class NoDecoding(BaseTransformer):
    """Implements a No-Op Decoding class for API consistency."""
    def transform(self, X):
        return X


class SeparatedBitPlanEncoder(BaseTransformer):
    """
    Implements an encoder for floating point input

    Parameters
    ----------
    precision: int, optional
        The number of binary projections that are preformed to reconstruct an unsigned floating point projection.
        if the input contains both positive and negative values, the total number of projections is 2*precision

    Returns
    -------
    X_bits: np.array(dtype = np.unit8)

    """

    def __init__(self, precision=6, **kwargs):
        assert(0 < precision <= 8)
        if "n_bits" in kwargs.keys() or "starting_bit" in kwargs.keys():
            raise RuntimeError("Encoder interface has changed from n_bit to precision")
        self.precision = precision
        self.magnitude = None, None

    @preserve_type
    def transform(self, X):

        def get_int_magnitude(X_):
            # separate case for integers to increase precision.
            # ensures X_quantized is just a shift.
            magnitude = X_.max()
            if magnitude < 0:
                return 0
            shift = self.precision-np.ceil(np.log2(X_.max()+1))
            return (2**self.precision-1) * 0.5**shift

        if np.issubdtype(X.dtype, np.signedinteger):
            magnitude_p = get_int_magnitude(+X)
            magnitude_n = get_int_magnitude(-X)
        elif np.issubdtype(X.dtype, np.integer):
            magnitude_p = get_int_magnitude(+X)
            magnitude_n = 0
        else:
            magnitude_p = np.max(+X.max(), 0)
            magnitude_n = np.max(-X.min(), 0)

        X = X.astype(np.float)

        dequantization_scale = (1 - 0.5 ** self.precision) * 2
        self.magnitude = magnitude_p/dequantization_scale, magnitude_n/dequantization_scale

        # Takes inputs in the range 0-1 ands splits into n (=precision) bitplanes
        def get_bits_unit_positive(X_, precision):
            X_quantized = (X_ * (2**precision-1) + 0.5).astype(np.uint8)
            X_bits = np.unpackbits(X_quantized[:, None], axis=1, bitorder='big')[:, -precision:]
            return X_bits.reshape(-1, *X_bits.shape[2:])

        magnitude_p_safe = magnitude_p if magnitude_p != 0 else 1
        magnitude_n_safe = magnitude_n if magnitude_n != 0 else 1

        if magnitude_n <= 0:
            return get_bits_unit_positive(np.clip(+X/magnitude_p_safe, 0, 1), self.precision)
        if magnitude_p <= 0:
            return get_bits_unit_positive(np.clip(-X/magnitude_n_safe, 0, 1), self.precision)

        Xp_bits = get_bits_unit_positive(np.clip(+X/magnitude_p_safe, 0, 1), self.precision)
        Xn_bits = get_bits_unit_positive(np.clip(-X/magnitude_n_safe, 0, 1), self.precision)

        return np.concatenate((Xp_bits, Xn_bits), 0)

    def get_params(self):
        """
        internal information necessary to undo the transformation,
        must be passed to the SeparatedBitPlanDecoder init.
        """
        return {'precision': self.precision, 'magnitude_p': self.magnitude[0], 'magnitude_n': self.magnitude[1]}


class SeparatedBitPlanDecoder(BaseTransformer):
    def __init__(self, precision, magnitude_p=1, magnitude_n=0, decoding_decay=0.5):
        """Init takes the output of the get_params() method of the SeparatedBitPlanEncoder"""
        self.precision = precision
        self.magnitude_p = magnitude_p
        self.magnitude_n = magnitude_n
        self.decoding_decay = decoding_decay

    @preserve_type
    def transform(self, X):
        n_out, n_features = X.shape
        sides = 2 if (self.magnitude_n > 0 and self.magnitude_p > 0) else 1
        n_dim_0 = int(n_out / (self.precision*sides))

        X = np.reshape(X, (sides, n_dim_0, self.precision, n_features))

        # recombines the bitplanes with the correct weights.
        def decode_unit_positive(X_):
            # compute factors for each bit to weight their significance
            decay_factors = (self.decoding_decay ** np.arange(self.precision)).astype('float32')
            if self.precision < 16:
                d = {'X' + str(i): X_[:, i] for i in range(self.precision)}
                d.update({'decay_factors' + str(i): decay_factors[i] for i in range(self.precision)})
                eval_str = ' + '.join(['X' + str(i) + '*' + 'decay_factors' + str(i) for i in range(self.precision)])
                X_dec = ne.evaluate(eval_str, d)
            else:
                # fallback to slower version if n_bits > 15 because of
                # https://gitlab.lighton.ai/main/lightonml/issues/58
                X_dec = np.einsum('ijk,j->ik', X_, decay_factors).astype('float32')
            return X_dec

        X_transformed = X.astype(np.float)
        Xp_transformed_raw = X_transformed[0]
        Xn_transformed_raw = X_transformed[1 if sides == 2 else 0]
        
        if self.magnitude_n <= 0:
            return decode_unit_positive(Xp_transformed_raw)*self.magnitude_p
        if self.magnitude_p <= 0:
            return -decode_unit_positive(Xp_transformed_raw)*self.magnitude_n
        
        Xp_transformed = decode_unit_positive(Xp_transformed_raw)*self.magnitude_p
        Xn_transformed = decode_unit_positive(Xn_transformed_raw)*self.magnitude_n

        return Xp_transformed - Xn_transformed
