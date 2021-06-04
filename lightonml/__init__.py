import pkg_resources
import warnings

try:
    package = pkg_resources.get_distribution('lightonml')
    __version__ = package.version
except pkg_resources.ResolutionError:
    __version__ = "unversioned"

# Don't expose pkg_resources in module's namespace
del pkg_resources

try:
    # Cleanup possibly existing global opu loggers
    # noinspection PyPackageRequirements
    import spdlog
    spdlog.drop("opu")
except (ImportError, RuntimeError):
    pass
finally:
    # Don't expose spdlog in module's namespace
    if "spdlog" in vars():
        del spdlog

_logger = None
# Set to 0, 1, 2 or 3
_verbose_level = 0
_has_warned_spd = False
_ml_data_dir = ""


def get_ml_data_dir():
    """Set the location the directory used in lightonml.datasets"""
    global _ml_data_dir
    return _ml_data_dir


def set_ml_data_dir(value):
    """Set the location the directory used in lightonml.datasets

    Overrides location defined in /etc/lighton/host.json or ~/.lighton.json
    Overridden by environment variable LIGHTONML_DATA_DIR
    """
    global _ml_data_dir
    _ml_data_dir = value


# noinspection PyUnresolvedReferences
def set_verbose_level(verbose_level):
    """Set the log_level for the lightonml module.
    Once change, one has to re-execute the get_trace_fn and alike
    Levels are 0: nothing, 1: print info, 2: debug info, 3: trace info
    """
    global _verbose_level, _logger
    _verbose_level = verbose_level
    __init_logger()
    if _logger:
        import spdlog
        if verbose_level <= 0:
            _logger.set_level(spdlog.LogLevel.OFF)
        elif verbose_level == 1:
            _logger.set_level(spdlog.LogLevel.INFO)
        elif verbose_level == 2:
            _logger.set_level(spdlog.LogLevel.DEBUG)
        elif verbose_level >= 3:
            _logger.set_level(spdlog.LogLevel.TRACE)


def get_verbose_level():
    global _verbose_level
    return _verbose_level


def get_debug_fn():
    """Returns debug logging function, or blank if logging level isn't debug"""
    return __get_logging_fn(2, "debug")


def get_trace_fn():
    """Returns trace loggeing function, or blank if logging level isn't trace"""
    return __get_logging_fn(3, "trace")


def get_print_fn():
    from lightonml.internal import utils
    global _verbose_level
    if _verbose_level >= 1:
        return print
    else:
        return utils.blank_fn


def __get_logging_fn(threshold_level, spd_level):
    from lightonml.internal import utils
    global _verbose_level, _logger, _has_warned_spd
    if _verbose_level >= threshold_level:
        __init_logger()
        if _logger:
            # Return the logging function (_logger.debug or _logger.trace)
            return getattr(_logger, spd_level)
        else:
            if not _has_warned_spd:
                warnings.warn("To have logging, install the spdlog package")
                _has_warned_spd = True
            return utils.blank_fn
    else:
        # below threshold, logger will be a blank function
        return utils.blank_fn


def __init_logger():
    """Init the logger, if spdlog is installed"""
    global _verbose_level, _logger, _has_warned_spd
    try:
        # Module is imported only once
        import spdlog
        # If spdlog package is available, instantiate a trace logger
        if not _logger:
            try:
                _logger = spdlog.get("opu")
            except RuntimeError:
                _logger = spdlog.ConsoleLogger("opu")
    except ImportError:
        _logger = None


# Finally make OPU available at the package level
# noinspection PyPep8
from lightonml.opu import OPU
