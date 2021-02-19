# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import argparse
import time
import numpy as np

import lightonml
from lightonml import OPU, __version__ as lgopu_version


def transform(n_images, n_features, n_components=0, disable_pbar=False):
    opu = OPU(disable_pbar=disable_pbar, open_at_init=False)
    if n_components != 0:
       opu.n_components = n_components
    ins = np.ones((n_images, n_features), dtype=np.uint8)

    with opu:
        print(opu.version())
        begin = time.time()
        opu.fit_transform1d(ins)
        elapsed = time.time() - begin
        print("{:d} transforms in {:.2f} s ({:.2f} Hz)".format(n_images,
                                                               elapsed, n_images / elapsed))


def main():
    print("LightOn OPU version", lgopu_version)
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--n-samples", type=int,
                        help="number of images", default=3000)
    parser.add_argument("-nc", "--n-components", type=int,
                        help="number of components", default=0)
    parser.add_argument("-nf", "--n-features", type=int,
                        help="number of features", default=100)
    parser.add_argument("-v", "--verbose", type=int, default=1)
    parser.add_argument("-dp", "--disable-pbar", action="store_true",
                        help="disable progress bar")
    args = parser.parse_args()
    lightonml.set_verbose_level(args.verbose)
    # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # If there are segfaults, it's a good idea to always use stderr as it
    # always prints to the screen, so you should get as much output as
    # possible.
    # sys.stdout = sys.stderr

    # uncomment the 2 following lines trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main(args.nbimages)')
    transform(args.n_samples, args.n_features, args.n_components, args.disable_pbar)


if __name__ == "__main__":
    main()
