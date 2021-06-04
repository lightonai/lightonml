import argparse
import time
from contextlib import contextmanager
import numpy as np
from lightonml import OPU, __version__ as lgopu_version


@contextmanager
def benchmark(n_times):
    begin = time.time()
    yield
    elapsed = time.time() - begin
    print("{:d} transforms in {:.2f} s ({:.2f} Hz)".format(n_times,
                                                           elapsed, n_times / elapsed))


def main():
    print("LightOn OPU version ", lgopu_version)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nbimages", type=int,
                        help="number of images", default=3000)
    args = parser.parse_args()
    n_images = args.nbimages

    with OPU() as opu:
        print("transforms without formatting")
        ins = np.ones((n_images, opu.max_n_features), dtype=np.uint8)
        ins_packed = np.packbits(ins, axis=1)
        with benchmark(n_images):
            opu.fit_transform1d(ins_packed, packed=True)

        n_features = 1000
        print("1D linear transforms with formatting")
        ins = np.ones((n_images, n_features), dtype=np.uint8)
        with benchmark(n_images):
            opu.fit1d(ins)
            opu.linear_transform(ins)

        print("1D transforms with formatting")
        ins = np.ones((n_images, n_features), dtype=np.uint8)
        with benchmark(n_images):
            opu.fit_transform1d(ins)

        print("Online transform")
        n_online = 1000
        with benchmark(n_online):
            opu.fit1d(n_features=n_features, online=True)
            for _ in range(n_online):
                opu.transform(ins[0])


if __name__ == "__main__":
    main()
