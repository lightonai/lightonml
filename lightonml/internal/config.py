# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import json
from pathlib import Path
import urllib.request
from typing import Any

from lightonml.internal import utils

"""
This is the Python equivalent of opu_config.sh shell script
"""
host_path = Path("/etc/lighton/host.json")


# noinspection PyUnresolvedReferences
def get_config_from(config_location: str, trace_fn):
    trace_fn("loading config from " + config_location)
    # figure out protocol (file or http)
    if config_location.startswith("http"):
        # load from http URL (on opu-station, usually)
        content = urllib.request.urlopen(config_location).read().decode()
        return json.loads(content)
    elif config_location.startswith("/"):
        return json.loads(Path(config_location).read_text())
    else:
        raise ValueError("Unknown protocol for config_location")


def load_config(override_location: str = "", trace_fn=utils.blank_fn):
    # read the OPU config from JSON
    # First if override_location is not "", use this location
    # Then look for entry in /etc/lighton/host.json, and if non-existent
    # use /etc/lighton/opu.json (the path we had before)

    if override_location:
        return get_config_from(override_location, trace_fn)

    if not host_path.exists():
        # If no host configuration, fail back to normal location, opu.json
        config_location = "/etc/lighton/opu.json"
    else:
        host = json.loads(host_path.read_text())
        config_location = host["opu_config"]
    return get_config_from(config_location, trace_fn)


def host_has_opu_config():
    return host_path.exists() or Path("/etc/lighton/opu.json").exists()


def get_host_option(key: str, default: Any = None):
    """Reads lighton's host.json option"""
    if host_path.exists():
        host = json.loads(host_path.read_text())
        return host.get(key, default)
    else:
        return default


def opu_version(config_d: dict) -> str:
    """Given an OPU config dict, returns array with OPU name, version, and core information"""
    opu_name = config_d.get('name', "NA")
    opu_version_ = config_d.get('version', "NA")
    opu_location = config_d.get('location', "NA")
    version = f"OPU {opu_name}-{opu_version_}-{opu_location}; "
    opu_type = config_d.get('core_type', "NA")
    opu_core_version = config_d.get('core_version', "NA")
    version += f"core type {opu_type}, core version {opu_core_version}"
    return version
