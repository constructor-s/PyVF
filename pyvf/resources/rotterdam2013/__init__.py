"""
This folder should contain:

Name: LongGlaucVF_20150216.zip
Size: 1117003 bytes (1090 KiB)
CRC32: F5591683
CRC64: DB72172E5AAB50F2
SHA256: 1B0AD98E1260E33553210F5F81CE45552DBB7947CC2C5BA4CA42B7926E93D51D
SHA1: 0C09C6D86CC1C649CCED8222A93276E9E270C83D
BLAKE2sp: 2C59A231D162FF5416A41FC045A8634F60A024A2FF8645C4A7127E378DDA7770

To download this file, please visit: http://www.rodrep.com/longitudinal-glaucomatous-vf-data---description.html
By using this data, you accept the terms outlined by Rotterdam Ophthalmic Data Repository
at http://data.rodrep.com/license.html

Copyright 2020 Bill Runjie Shi
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
import zipfile
import os
import logging

_logger = logging.getLogger(__name__)

LongGlaucVF_20150216_URL = "http://data.rodrep.com/LongGlaucVF_20150216.zip"  # "http://localhost:8000/LongGlaucVF_20150216.zip"
LongGlaucVF_20150216_FILENAME = "LongGlaucVF_20150216.zip"

if not importlib.resources.is_resource(__name__, LongGlaucVF_20150216_FILENAME):
    _logger.error("%s does not exist", LongGlaucVF_20150216_FILENAME)
    _logger.warning("Downloading %s from %s", LongGlaucVF_20150216_FILENAME, LongGlaucVF_20150216_URL)
    _logger.warning("By downloading, you agree to the terms of outlined by Rotterdam Ophthalmic Data Repository")
    _logger.warning("Visit http://data.rodrep.com/license.html for more details")

    # Hacky implementation
    dirname, basename = os.path.split(__spec__.origin)

    from urllib.request import urlretrieve
    urlretrieve(LongGlaucVF_20150216_URL, os.path.join(dirname, LongGlaucVF_20150216_FILENAME))

    _logger.warning("Downloaded %s", LongGlaucVF_20150216_FILENAME)

with importlib.resources.open_binary(__name__, LongGlaucVF_20150216_FILENAME) as zf:
    root = zipfile.ZipFile(zf)
    with root.open("VisualFields.csv") as f:
        VISUAL_FIELDS = pandas.read_csv(f)
    with root.open("VFPoints.csv") as f:
        VF_POINTS = pandas.read_csv(f)

    VF_POINTS_OD = VF_POINTS.merge(VISUAL_FIELDS[["STUDY_ID", "FIELD_ID", "SITE"]], how="left")
    os_mask = VF_POINTS_OD["SITE"] == "OS"
    VF_POINTS_OD["STUDY_SITE_ID"] = VF_POINTS_OD["STUDY_ID"] * 2 - 1 + os_mask  # starts from one
    VF_POINTS_OD.loc[os_mask, "X"] *= -1
    VF_POINTS_OD.loc[os_mask, "SITE"] = "OD"
    VF_POINTS_OD = VF_POINTS_OD.sort_values(by=["FIELD_ID", "Y", "X"], ascending=[True, False, True])

    M = 54
    VF_THRESHOLD = VF_POINTS_OD["THRESHOLD"].to_numpy(dtype=np.float32).reshape([-1, M])
    VF_THRESHOLD_SITES = VF_POINTS_OD["STUDY_SITE_ID"].to_numpy(dtype=np.int32).reshape([-1, M])
    VF_THRESHOLD_SITES = VF_THRESHOLD_SITES[:, 0]
    VF_THRESHOLD_INFO = VF_POINTS_OD.iloc[::M, :][["FIELD_ID", "STUDY_SITE_ID"]]
    VF_THRESHOLD_INFO = VF_THRESHOLD_INFO.merge(VISUAL_FIELDS, how="left")
    N, M = VF_THRESHOLD.shape
    VF_BLINDSPOTS = (25, 34)

    """
    24-2 VF Map, OD coordinates
             00,01,02,03,
          04,05,06,07,08, 09,
       10,11,12,13,14,15, 16, 17,
    18,19,20,21,22,23,24,(25),26,
    27,28,29,30,31,32,33,(34),35,
       36,37,38,39,40,41, 42, 43,
          44,45,46,47,48, 49,
             50,51,52,53
    """

