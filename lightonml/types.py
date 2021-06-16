from enum import Enum


class OutputRescaling(Enum):
    """Strategy used for rescaling the output"""
    variance = 1
    """Rescale with the standard deviation computed on a Gaussian input"""
    norm = 2
    """Ensure approximate conservation of the norm (RIP)"""
    none = 3
    """No rescaling"""