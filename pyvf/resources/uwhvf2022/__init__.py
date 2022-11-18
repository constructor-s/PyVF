"""
This folder should contain:

Name: VF_Data.csv
Size: 33735723 bytes (32 MiB)
CRC32: 6666F3EC
CRC64: CA5F41ECA79C2690
SHA256: 5b6999ca0ca229a93b1aacc44bbac7f68eb28962ace0aa9ef31e7fca28fb6b0e
SHA1: 30b8017f25d81001d583925f990c56ffdee8f5c0
BLAKE2sp: 7a6ccf55666032a0d2eb180b57ec5df93211a0b8d93e417e33e7e505123b1b3e

This file is available at:
https://github.com/uw-biomedical-ml/uwhvf/raw/0c07384b1345aca702f503a959d8815ff0bfa17a/CSV/VF_Data.csv

By using this data, you accept the terms outlined by the original authors of:
UWHVF: A real-world, open source dataset of Humphrey Visual Fields (HVF) 
from the University of Washington
https://github.com/uw-biomedical-ml/uwhvf/blob/master/LICENSE

Copyright 2022 Bill Runjie Shi
At the Vision and Eye Movements Lab, University of Toronto.
Visit us at: http://www.eizenman.ca/

This file is part of PyVF.

PyVF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyVF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyVF. If not, see <https://www.gnu.org/licenses/>.
"""

import pandas
import numpy as np
import importlib.resources
import os
import logging

_logger = logging.getLogger(__name__)

VF_DATA_URL = "https://github.com/uw-biomedical-ml/uwhvf/raw/0c07384b1345aca702f503a959d8815ff0bfa17a/CSV/VF_Data.csv"  # "http://localhost:8000/VF_Data.csv"
VF_DATA_FILENAME = "VF_Data.csv"

if not importlib.resources.is_resource(__name__, VF_DATA_FILENAME):
    _logger.error("%s does not exist", VF_DATA_FILENAME)
    _logger.warning("Downloading %s from %s", VF_DATA_FILENAME, VF_DATA_URL)
    _logger.warning("By downloading, you agree to the terms of outlined by UWHVF Repository")
    _logger.warning("Visit https://github.com/uw-biomedical-ml/uwhvf/blob/master/LICENSE for more details")

    # Hacky implementation
    dirname, basename = os.path.split(__spec__.origin)

    from urllib.request import urlretrieve
    urlretrieve(VF_DATA_URL, os.path.join(dirname, VF_DATA_FILENAME))

    _logger.warning("Downloaded %s", VF_DATA_FILENAME)

with importlib.resources.open_binary(__name__, VF_DATA_FILENAME) as f:
    VF_DATA = pandas.read_csv(f)

    VF_THRESHOLD = VF_DATA[[f"Sens_{i}" for i in range(1, 55)]]
    VF_THRESHOLD = VF_THRESHOLD.values.astype(np.float32)
    VF_THRESHOLD.flags.writeable = False

    VF_TD = VF_DATA[[f"TD_{i}" for i in range(1, 55)]]
    VF_TD = VF_TD.values.astype(np.float32)
    VF_TD.flags.writeable = False

    VF_PD = VF_DATA[[f"PD_{i}" for i in range(1, 55)]]
    VF_PD = VF_PD.values.astype(np.float32)
    VF_PD.flags.writeable = False

    VF_THRESHOLD_SITES = VF_DATA["PatID"] * 2 - 1 + (VF_DATA["Eye"] == "Left")
    VF_THRESHOLD_SITES = VF_THRESHOLD_SITES.values.astype(np.int32)
    VF_THRESHOLD_SITES.flags.writeable = False

    VF_THRESHOLD_INFO = VF_DATA.loc[:, :"GH"]
    VF_THRESHOLD_INFO["STUDY_SITE_ID"] = VF_THRESHOLD_SITES

