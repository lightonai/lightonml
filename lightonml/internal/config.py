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

    host_path = Path("/etc/lighton/host.json")
    if not host_path.exists():
        # If no host configuration, fail back to normal location, opu.json
        config_location = "/etc/lighton/opu.json"
    else:
        host = json.loads(host_path.read_text())
        config_location = host["opu_config"]
    return get_config_from(config_location, trace_fn)


def get_host_option(key:str, default:Any = None):
    """Reads lighton's host.json option"""
    host_path = Path("/etc/lighton/host.json")
    if host_path.exists():
        host = json.loads(host_path.read_text())
        return host.get(key, default)
    else:
        return default
