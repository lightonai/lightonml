# Copyright (c) 2020 LightOn, All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import sys
# import tqdm (progress bar) only if available
try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from tqdm import tqdm
except ImportError:
    pass
from lightonml import get_verbose_level


class Progress:
    def __init__(self, total, description, disable):
        self.total = total
        self.disable_pbar = get_verbose_level() < 1 or disable
        self.pbar = None
        self.description = description

    def __enter__(self):
        if 'tqdm' in sys.modules and self.total > 1 and not self.disable_pbar:
            self.pbar = tqdm(total=self.total, desc=self.description)
        else:
            self.pbar = None
        return self

    def update(self, n_inc):
        if self.pbar is not None:
            self.pbar.update(n_inc)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar is not None:
            self.pbar.close()
