
"""OPU object is mainly used with fit1d, fit2d and transform/linear_transform methods.
They are quite versatile in terms of input dimensions, here's a
summary.
"""

from lightonml import OPU
import numpy as np


# noinspection PyUnboundLocalVariable
def opu_tutorial(opu: OPU):
    # Rule is that OPU input is binary, so input vector elements should be 0 or 1
    # To get your input as binary, see the encoders from lightonml package.

    # The tutorial makes use of NumPy arrays, but you can use PyTorch tensors as input.
    # The output then is a tensor as well.

    # The dimension of output is governed by the n_components attribute:
    print("Number of components:", opu.n_components)

    # Simplest input is a batch of N 1d vectors, of arbitrary size
    inp_1d = random_bin((2000, 1000))
    out1 = opu.fit_transform1d(inp_1d)
    print("1D transform out shape", out1.shape)  # (2000, opu.n_components)

    # In this example we fit the OPU with the input and then transform or linear_transform,
    # but if we have several batches to be transformed, we must call fit first
    # on one the batches, and then transform:
    inp_1d_b = random_bin((2000, 1000))
    opu.fit1d(inp_1d)
    out1 = opu.transform(inp_1d)
    out1_b = opu.transform(inp_1d_b)
    print("transform of two separate batches", out1.shape, out1_b.shape)

    # For a linear transform, use OPU.linear_transform with the same parameters:
    out1_l = opu.linear_transform(inp_1d)
    print("linear_transform of a batch", out1.shape, out1_l.shape)

    # You can also run fit with the number of features, instead:
    opu.fit1d(n_features=inp_1d.shape[1])
    out1_alt = opu.transform(inp_1d)
    print("transform 1d", out1_alt.shape)

    # But you can transform a single vector if you wish
    single = opu.fit_transform1d(inp_1d[0])
    print("single transform out shape", single.shape)  # n_components

    # However if you have many single vectors, you will need either to use the
    # online mode (see below), or have them transformed in a single batch,
    # you will gain a lot of performance

    # if you have 2d vectors, you'll benefit from the fact that OPU input is
    # physically in 2d, so don't reshape them! Instead, use fit_transform2d:
    inp_2d = random_bin((2000, 13, 10))  # 200 vectors of shape 13x10
    out2 = opu.fit_transform2d(inp_2d)
    print("2D transform out shape", out2.shape)  # (2000, opu.n_components)

    # If you're batch is 2D-shaped, transform will recognize it and return
    # an output of the same shape:
    inp_3d = random_bin((300, 100, 3000))
    # call 1D transform on a 3D vector
    out_3d = opu.fit_transform1d(inp_3d)  # batch of 300x100 1D vectors
    print("3D transform out shape", out_3d.shape)  # (300, 100, opu.n_components)

    # You can also have bit-packed input, which will optimize memory-space,
    # as well as being a bit more efficient
    inp_1d_p = np.packbits(inp_1d, axis=1)
    print("1D packed transform in shape", inp_1d_p.shape)  # (2000, 1000/8)
    out1_p = opu.fit_transform1d(inp_1d_p, packed=True)
    print("1D packed transform out shape", out1_p.shape)  # (2000, opu.n_components)

    # If your input is 2d AND bit-packed, you must tell what shape it is
    inp_2d_p = np.packbits(inp_2d.reshape(2000, -1), axis=1)
    print("1D packed transform in shape", inp_2d_p.shape)  # (2000, 1300/8)
    out2_p = opu.fit_transform2d(inp_2d_p, packed=True, n_2d_features=(13, 10))
    print("2D packed transform out shape", out2_p.shape)  # (2000, opu.n_components)

    # Input is formatted to match OPU's input device, but you can pass directly
    # formatted input of the correct input size (should it be 1d or 2d)
    input_shape = tuple(opu.device.input_shape)
    single_raw_2d = random_bin(input_shape)
    single_raw_1d = np.reshape(single_raw_2d, -1)
    out_raw_1d = opu.fit_transform1d(single_raw_1d)
    print("1D raw out shape", out_raw_1d.shape)  # (2000, opu.n_components)
    out_raw_2d = opu.fit_transform2d(single_raw_2d)
    print("1D raw out shape", out_raw_2d.shape)  # (2000, opu.n_components)
    # Of course, the equivalent for packed input of input device's size
    # will work the same way

    # It can also be a batch of them
    many_raw_2d = random_bin((100, ) + input_shape)
    out_many_raw_2d = opu.fit_transform2d(many_raw_2d)
    print("Many raw out shape", out_many_raw_2d.shape)   # (100, opu.n_components)

    # The online mode allows you to run accelerate the run of single vectors:
    n_features1d = 1200
    opu.fit1d(n_features=n_features1d, online=True)
    for _ in range(10):
        online_out = opu.transform(random_bin(n_features1d))
    print("Online out shape", online_out.shape)

    n_features2d = (50, 50)
    opu.fit2d(n_features=n_features2d, online=True)
    for _ in range(10):
        online_out = opu.transform(random_bin(n_features2d))
    print("Online out shape", online_out.shape)


def random_bin(shape):
    """Generates a random vector of 0s and 1s"""
    return np.random.randint(0, 2, size=shape, dtype=bool)


if __name__ == '__main__':
    opu_ = OPU()
    opu_tutorial(opu_)
