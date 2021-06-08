# <img src="https://cloud.lighton.ai/wp-content/uploads/2020/01/LightOnCloud.png" width=80/> LightOnML library

[![Twitter Follow](https://img.shields.io/twitter/follow/LightOnIO.svg?style=social)](https://twitter.com/LightOnIO) [![PyPi Version](https://img.shields.io/pypi/v/lightonml.svg)](https://pypi.python.org/pypi/lightonml/) [![Python Versions](https://img.shields.io/pypi/pyversions/lightonml.svg)](https://pypi.python.org/pypi/lightonml/)

LightOnML is a high level machine learning-oriented API that allows to perform random projections on
LightOn’s optical processing units (OPUs). LightOn’s OPUs are available through [LightOn’s Cloud service](https://cloud.lighton.ai).

## Features

 * Run large-scale non-linear and linear random projections using LightOn’s Aurora OPUs
 * Simulate these projections on any machine without access to an OPU
 * Encode input data in a binary form using various encoders, for OPU input

## Installation

`lightonml` doesn't require access to an OPU for some functionalities, but for performing
 computations on an OPU you'll need one. Otherwise, a simulated OPU can be used.

To install, use `pip`:

    pip install lightonml

Optional dependencies are :
* `torch`, required for the encoder classes, and the PyTorch `OPUMap`.
* `scikit-learn`, required for using the corresponding `OPUMap` to work.

## Documentation, examples and help

Main documentation can be found at the [API docs website](https://docs.lighton.ai).

Check the examples directory in the repo, if you don't have access to an OPU you can run the code locally with a simulated OPU

For getting help on the LightOn Cloud service check the [Community website](https://community.lighton.ai/)

For help on the library itself, you can use issues on this repository.

## <img src="https://cloud.lighton.ai/wp-content/uploads/2020/01/cropped-lightOnCloud-1-2.png" width=120/> Access to Optical Processing Units

To request access to LightOn Cloud and try our photonic co-processor, please visit: https://cloud.lighton.ai/

For researchers, we also have a LightOn Cloud for Research program, please visit https://cloud.lighton.ai/lighton-research/ for more information.
