
Bitplan encoding and decoding
=============================

`SeparatedBitPlanEncoder <lightonml.encoding.base.SeparatedBitPlanEncoder>`
and `MixingBitPlanDecoder <lightonml.encoding.base.MixingBitPlanDecoder>` implement a
simple but powerful encoding scheme (especially useful for RGB images).

Sample data
-----------

.. code:: ipython3

    import numpy as np

.. code:: ipython3

    sample = np.array([[204, 180, 73], [133, 11, 39]], dtype='uint8')
    sample



.. parsed-literal::

    array([[204, 180,  73],
           [133,  11,  39]], dtype=uint8)



Encoding
--------

.. figure:: ../_static/img/encoding.png
   :alt: decomposition of an RGB image in bitplanes

   alt text

Let's take an RGB colored image: it is composed of 3 channels (red,
green and blue) of the same width and height of ``uint8``. The
`SeparatedBitPlanEncoder <lightonml.encoding.base.SeparatedBitPlanEncoder>`
flattens and concatenates each channel and builds the binary representation of them.

We start by explaining what happens with the default setting
``n_bits=8`` and ``starting_bit=0`` and an array of ``uint8`` in input.

.. code:: ipython3

    import lightonml.encoding.base as base

.. code:: ipython3

    encoder = base.SeparatedBitPlanEncoder()
    encoder




.. parsed-literal::

    SeparatedBitPlanEncoder(n_bits=8, starting_bit=0)



.. code:: ipython3

    encoded_sample = encoder.transform(sample)
    print('The encoded sample has shape {}.'.format(encoded_sample.shape))
    print('The shape is (n_samples*n_bits, n_features), in this case (2*8, 3)')
    print(encoded_sample)


.. parsed-literal::

    The encoded sample has shape (16, 3).
    The shape is (n_samples*n_bits, n_features), in this case (2*8, 3)
    [[0 0 1]
     [0 0 0]
     [1 1 0]
     [1 0 1]
     [0 1 0]
     [0 1 0]
     [1 0 1]
     [1 1 0]
     [1 1 1]
     [0 1 1]
     [1 0 1]
     [0 1 0]
     [0 0 0]
     [0 0 1]
     [0 0 0]
     [1 0 0]]


Let's see what happens inside the `transform` method:

-  we add an auxiliary axis to go 3D
-  we unpack the bit representation on the auxiliary axis

.. code:: ipython3

    # record the original dimensions
    n_samples, n_features = sample.shape
    print('Original shape: ({}, {})'.format(n_samples, n_features))
    
    # add an auxiliary axis: [n_samples, n_features] -> [n_samples, n_features, 1]
    sample_uint8 = np.expand_dims(sample, axis=2).view(np.uint8)
    print('Expanded shape: {}'.format(sample_uint8.shape))
    
    # Unpacks the bits along the auxiliary axis: [n_samples, n_features, 1] -> [n_samples, n_features, 8]
    sample_uint8_unpacked = np.unpackbits(sample_uint8, axis=2)
    print('Unpacked shape: {}'.format(sample_uint8_unpacked.shape))
    
    print('Unpacked sample')
    print(sample_uint8_unpacked)


.. parsed-literal::

    Original shape: (2, 3)
    Expanded shape: (2, 3, 1)
    Unpacked shape: (2, 3, 8)
    Unpacked sample
    [[[1 1 0 0 1 1 0 0]
      [1 0 1 1 0 1 0 0]
      [0 1 0 0 1 0 0 1]]
    
     [[1 0 0 0 0 1 0 1]
      [0 0 0 0 1 0 1 1]
      [0 0 1 0 0 1 1 1]]]


In ``uint8`` we can represent the interval :math:`[0, 255]`. Let's take
the first row of ``sample``:

+---------------------+---------------------+--------------------+--------------------+--------------------+-------------------+-------------------+-------------------+-------------------+
| powers of 2         | :math:`2^7 (128)`   | :math:`2^6 (64)`   | :math:`2^5 (32)`   | :math:`2^4 (16)`   | :math:`2^3 (8)`   | :math:`2^2 (4)`   | :math:`2^1 (2)`   | :math:`2^0 (1)`   |
+=====================+=====================+====================+====================+====================+===================+===================+===================+===================+
| binary rep of 204   | 1                   | 1                  | 0                  | 0                  | 1                 | 1                 | 0                 | 0                 |
+---------------------+---------------------+--------------------+--------------------+--------------------+-------------------+-------------------+-------------------+-------------------+
| binary rep of 180   | 1                   | 0                  | 1                  | 1                  | 0                 | 1                 | 0                 | 0                 |
+---------------------+---------------------+--------------------+--------------------+--------------------+-------------------+-------------------+-------------------+-------------------+
| binary rep of 73    | 0                   | 1                  | 0                  | 0                  | 1                 | 0                 | 0                 | 1                 |
+---------------------+---------------------+--------------------+--------------------+--------------------+-------------------+-------------------+-------------------+-------------------+

This is the unpacked bit representation for each element, with bits
going from the most significant (MSB) to the least significant (LSB).

-  we reverse the order of the bit to go from the least significant to
   the most significant bit

.. code:: ipython3

    # Reverse the order of bits: MSB to LSB becomes LSB to MSB
    # LSB = Least Significant Bit
    # MSB = Most Significant Bit
    sample_uint8_reversed = np.flip(sample_uint8_unpacked, axis=2)
    
    print('Reversed sample')
    print(sample_uint8_reversed)
    print('You can see that we just reversed the order of the representation.')


.. parsed-literal::

    Reversed sample
    [[[0 0 1 1 0 0 1 1]
      [0 0 1 0 1 1 0 1]
      [1 0 0 1 0 0 1 0]]
    
     [[1 0 1 0 0 0 0 1]
      [1 1 0 1 0 0 0 0]
      [1 1 1 0 0 1 0 0]]]
    You can see that we just reversed the order of the representation.


-  we switch the auxiliary axis with the features axis

.. code:: ipython3

    # switch axis 2 with axis 1
    encoded_sample = np.transpose(sample_uint8_reversed, [0, 2, 1])
    
    print('Encoded sample')
    print(encoded_sample)
    print('We have switched axis 1 and 2, the representation is now on columns.')


.. parsed-literal::

    Encoded sample
    [[[0 0 1]
      [0 0 0]
      [1 1 0]
      [1 0 1]
      [0 1 0]
      [0 1 0]
      [1 0 1]
      [1 1 0]]
    
     [[1 1 1]
      [0 1 1]
      [1 0 1]
      [0 1 0]
      [0 0 0]
      [0 0 1]
      [0 0 0]
      [1 0 0]]]
    We have switched axis 1 and 2, the representation is now on columns.


-  we select the bit representation or a part of it by slicing

.. code:: ipython3

    # slicing does nothing if self.starting_bit=0 and n_bits=bitwidth of input - like in this case
    encoded_sample = encoded_sample[:, encoder.starting_bit:encoder.n_bits + encoder.starting_bit, :]
    print(encoded_sample)


.. parsed-literal::

    [[[0 0 1]
      [0 0 0]
      [1 1 0]
      [1 0 1]
      [0 1 0]
      [0 1 0]
      [1 0 1]
      [1 1 0]]
    
     [[1 1 1]
      [0 1 1]
      [1 0 1]
      [0 1 0]
      [0 0 0]
      [0 0 1]
      [0 0 0]
      [1 0 0]]]


-  we reshape the encoded sample to [n\_samples \* n\_bits, n\_features]

In the end we get a representation were the columns are concatenated
``n_bits`` representation of each feature of the samples.

.. code:: ipython3

    # the encoded sample is then reshaped to [n_samples * n_bits, n_features]
    reshaped_encoded_sample = encoded_sample.reshape((n_samples * encoder.n_bits, n_features))
    print('Reshaped encoded shape: {}'.format(reshaped_encoded_sample.shape))
    print('Encoded sample:')
    print(reshaped_encoded_sample)
    print('Each column is the concatenation of separate columns of the previous cell')


.. parsed-literal::

    Reshaped encoded shape: (16, 3)
    Encoded sample:
    [[0 0 1]
     [0 0 0]
     [1 1 0]
     [1 0 1]
     [0 1 0]
     [0 1 0]
     [1 0 1]
     [1 1 0]
     [1 1 1]
     [0 1 1]
     [1 0 1]
     [0 1 0]
     [0 0 0]
     [0 0 1]
     [0 0 0]
     [1 0 0]]
    Each column is the concatenation of separate columns of the previous cell


Decoding
--------

.. figure:: ../_static/img/decoding.png
   :alt: decoding

   alt text

.. code:: ipython3

    decoder = base.MixingBitPlanDecoder(decoding_decay=2)
    decoder




.. parsed-literal::

    MixingBitPlanDecoder(decoding_decay=2, n_bits=8)



Note that here we set ``decoding_decay``\ :math:`=2`, but when using the
OPU, where the random features are in :math:`[0, 255]`, you need to use
:math:`0.5`.

.. code:: ipython3

    decoded_sample = decoder.transform(reshaped_encoded_sample)
    
    print('The decoded sample returns to the original shape {} [n_samples, n_features].'.format(decoded_sample.shape))
    print(decoded_sample)


.. parsed-literal::

    The decoded sample returns to the original shape (2, 3) [n_samples, n_features].
    [[ 204.  180.   73.]
     [ 133.   11.   39.]]


This is what happens in the ``transform`` method:

-  we compute what was the original shape of the data

.. note::
    ``n_bits`` must be set to the same value used for the encoder, otherwise an error is raised.

.. code:: ipython3

    # compute the original shape of the data
    n_out, n_features = reshaped_encoded_sample.shape
    n_dim_0 = n_out // decoder.n_bits

-  we reshape the array to 3D [n\_samples, n\_bits, n\_features]

.. code:: ipython3

    # the data are reshaped in 3D [n_samples, n_bits, n_features]
    reshaped_encoded_sample = np.reshape(reshaped_encoded_sample, (n_dim_0, decoder.n_bits, n_features))

-  we build an array with decaying factors using:

.. math:: \mbox{DecayFactor}(i) = \left (2\right )^{-i}

-  we multiply the encoded sample with the decay factors along the bit
   dimension

.. code:: ipython3

    # a decay_factors array is built, that weights the importance of every bit in the 
    # product on the second line
    decay_factors = np.reshape(decoder.decoding_decay ** np.arange(decoder.n_bits), (1, decoder.n_bits, 1))
    decayed_sample = reshaped_encoded_sample * decay_factors

-  we sum over the bit axis

.. code:: ipython3

    decoded_sample = np.sum(decayed_sample, axis=1)

.. code:: ipython3

    decoded_sample




.. parsed-literal::

    array([[204, 180,  73],
           [133,  11,  39]])



Optional arguments
------------------

The parameters ``n_bits`` and ``starting_bits`` defaults can be changed.
This is useful if you notice that certain bitplanes are just noise and
you want to throw them away.
