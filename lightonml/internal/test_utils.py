"""Some tooling function used in lightonml test scripts"""


import argparse
import sys

_test_quick = False


def set_is_test_quick(value):
    global _test_quick
    _test_quick = value


def get_is_test_quick():
    global _test_quick
    return _test_quick


def parse_args():
    """Special parse_args to set verbose level and quick tests
    :return args with "quick" and "verbose" attribute, and argv to be left
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-oq', help="Don't delete OPU at each test",
                        action="store_true", dest="quick")
    parser.add_argument('-ov', help='OPU verbose level',
                        type=int, default=0, dest="verbose")
    # Use parse_known_args to prevent error on later
    args, left_args = parser.parse_known_args()
    return args, sys.argv[:1] + left_args


def restore_exposure(func):
    """Decorator for quick test mode, restores exposure after test"""
    def func_wrapper(self):
        try:
            func(self)
        finally:
            if self.quick_test:
                self.device.exposure_us = self.exposure_us
    return func_wrapper


def restore_frametime(func):
    """Decorator for quick test mode, restores frametime after test"""
    def func_wrapper(self):
        try:
            func(self)
        finally:
            if self.quick_test:
                self.device.frametime_us = self.frametime_us
    return func_wrapper


def skip_if_quick(_):
    """Decorator for skipping a test case in quick mode"""
    def func_wrapper(self):
        if self.quick_test:
            self.skipTest("can test only without test_utils.is_test_quick")
    return func_wrapper
