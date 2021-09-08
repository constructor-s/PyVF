"""
Anonymization script for PDF SFA files exported from Zeiss Forum.
Intended for scientific research use only.
(Work in progress)

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

from pdfminer.converter import PDFLayoutAnalyzer, utils
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from datetime import datetime, timedelta
from io import BytesIO
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import hashlib
import re
from copy import copy
from pdfminer.pdfinterp import PDFTextState, PDFGraphicState
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdftypes import PDFStream
import logging
_logger = logging.getLogger(__name__)


class HFAPDFParser:
    REGEX_PATTERN_242 = "\n".join((
        "30" + chr(176),
        "30" + chr(176),
        "30" + chr(176),
        r"(?P<vf>(<?\d+\n){54})(?P<vfimg>(.+\n){412})" +
        r"(?P<td>(-?\d+\n){52})(?P<pd>((-?\d+\n){52})|(MD Threshold exceeded.\nSee Total Deviation plot.\n))" +
        r"(?P<tdp>([10]\.\d+\n){52})(?P<pdp>(([10]\.\d+\n){52})|(MD Threshold exceeded.\nSee Total Deviation plot.\n))"
        "Total Deviation",
        "Pattern Deviation"
    ))  # Currently only used to parse the vf, td, pd, td probability, pd probability regions
    REGEX_COMPILED_242 = re.compile(REGEX_PATTERN_242)

    # PDF matrices - these can be obtained with "pdf_parser_example.py --dump"
    MATRICES_VF_242_OS = np.array(((1, 0, 0, 1, 242.23, 588.31),(1, 0, 0, 1, 224.6, 588.31),(1, 0, 0, 1, 206.98, 588.31),(1, 0, 0, 1, 189.35, 588.31),(1, 0, 0, 1, 259.85, 570.69),(1, 0, 0, 1, 242.23, 570.69),(1, 0, 0, 1, 224.6, 570.69),(1, 0, 0, 1, 206.98, 570.69),(1, 0, 0, 1, 189.35, 570.69),(1, 0, 0, 1, 171.73, 570.69),(1, 0, 0, 1, 277.48, 553.06),(1, 0, 0, 1, 259.85, 553.06),(1, 0, 0, 1, 242.23, 553.06),(1, 0, 0, 1, 224.6, 553.06),(1, 0, 0, 1, 206.98, 553.06),(1, 0, 0, 1, 189.35, 553.06),(1, 0, 0, 1, 171.73, 553.06),(1, 0, 0, 1, 154.1, 553.06),(1, 0, 0, 1, 295.1, 535.44),(1, 0, 0, 1, 277.48, 535.44),(1, 0, 0, 1, 259.85, 535.44),(1, 0, 0, 1, 242.23, 535.44),(1, 0, 0, 1, 224.6, 535.44),(1, 0, 0, 1, 206.98, 535.44),(1, 0, 0, 1, 189.35, 535.44),(1, 0, 0, 1, 171.73, 535.44),(1, 0, 0, 1, 154.1, 535.44),(1, 0, 0, 1, 295.1, 517.81),(1, 0, 0, 1, 277.48, 517.81),(1, 0, 0, 1, 259.85, 517.81),(1, 0, 0, 1, 242.23, 517.81),(1, 0, 0, 1, 224.6, 517.81),(1, 0, 0, 1, 206.98, 517.81),(1, 0, 0, 1, 189.35, 517.81),(1, 0, 0, 1, 173.39, 517.81),(1, 0, 0, 1, 154.1, 517.81),(1, 0, 0, 1, 277.48, 500.19),(1, 0, 0, 1, 259.85, 500.19),(1, 0, 0, 1, 242.23, 500.19),(1, 0, 0, 1, 224.6, 500.19),(1, 0, 0, 1, 206.98, 500.19),(1, 0, 0, 1, 189.35, 500.19),(1, 0, 0, 1, 171.73, 500.19),(1, 0, 0, 1, 154.1, 500.19),(1, 0, 0, 1, 259.85, 482.56),(1, 0, 0, 1, 242.23, 482.56),(1, 0, 0, 1, 224.6, 482.56),(1, 0, 0, 1, 206.98, 482.56),(1, 0, 0, 1, 189.35, 482.56),(1, 0, 0, 1, 171.73, 482.56),(1, 0, 0, 1, 242.23, 464.94),(1, 0, 0, 1, 224.6, 464.94),(1, 0, 0, 1, 206.98, 464.94),(1, 0, 0, 1, 189.35, 464.94)))
    MATRICES_VF_242_OD = np.array(((1, 0, 0, 1, 189.35, 588.31),(1, 0, 0, 1, 206.98, 588.31),(1, 0, 0, 1, 224.6, 588.31),(1, 0, 0, 1, 242.23, 588.31),(1, 0, 0, 1, 171.73, 570.69),(1, 0, 0, 1, 189.35, 570.69),(1, 0, 0, 1, 206.98, 570.69),(1, 0, 0, 1, 224.6, 570.69),(1, 0, 0, 1, 242.23, 570.69),(1, 0, 0, 1, 259.85, 570.69),(1, 0, 0, 1, 154.1, 553.06),(1, 0, 0, 1, 171.73, 553.06),(1, 0, 0, 1, 189.35, 553.06),(1, 0, 0, 1, 206.98, 553.06),(1, 0, 0, 1, 224.6, 553.06),(1, 0, 0, 1, 242.23, 553.06),(1, 0, 0, 1, 259.85, 553.06),(1, 0, 0, 1, 277.48, 553.06),(1, 0, 0, 1, 136.48, 535.44),(1, 0, 0, 1, 154.1, 535.44),(1, 0, 0, 1, 171.73, 535.44),(1, 0, 0, 1, 189.35, 535.44),(1, 0, 0, 1, 206.98, 535.44),(1, 0, 0, 1, 224.6, 535.44),(1, 0, 0, 1, 242.23, 535.44),(1, 0, 0, 1, 259.85, 535.44),(1, 0, 0, 1, 277.48, 535.44),(1, 0, 0, 1, 136.48, 517.81),(1, 0, 0, 1, 154.1, 517.81),(1, 0, 0, 1, 171.73, 517.81),(1, 0, 0, 1, 189.35, 517.81),(1, 0, 0, 1, 206.98, 517.81),(1, 0, 0, 1, 224.6, 517.81),(1, 0, 0, 1, 242.23, 517.81),(1, 0, 0, 1, 261.52, 517.81),(1, 0, 0, 1, 277.48, 517.81),(1, 0, 0, 1, 154.1, 500.19),(1, 0, 0, 1, 171.73, 500.19),(1, 0, 0, 1, 189.35, 500.19),(1, 0, 0, 1, 206.98, 500.19),(1, 0, 0, 1, 224.6, 500.19),(1, 0, 0, 1, 242.23, 500.19),(1, 0, 0, 1, 259.85, 500.19),(1, 0, 0, 1, 277.48, 500.19),(1, 0, 0, 1, 171.73, 482.56),(1, 0, 0, 1, 189.35, 482.56),(1, 0, 0, 1, 206.98, 482.56),(1, 0, 0, 1, 224.6, 482.56),(1, 0, 0, 1, 242.23, 482.56),(1, 0, 0, 1, 259.85, 482.56),(1, 0, 0, 1, 189.35, 464.94),(1, 0, 0, 1, 206.98, 464.94),(1, 0, 0, 1, 224.6, 464.94),(1, 0, 0, 1, 242.23, 464.94)))
    MATRICES_TD_242_OS = np.array(((1, 0, 0, 1, 143.11, 429.03),(1, 0, 0, 1, 131.3, 429.03),(1, 0, 0, 1, 119.49, 429.03),(1, 0, 0, 1, 106.68, 429.03),(1, 0, 0, 1, 153.93, 417.22),(1, 0, 0, 1, 142.11, 417.22),(1, 0, 0, 1, 130.3, 417.22),(1, 0, 0, 1, 118.49, 417.22),(1, 0, 0, 1, 107.68, 417.22),(1, 0, 0, 1, 94.86, 417.22),(1, 0, 0, 1, 165.74, 405.41),(1, 0, 0, 1, 153.93, 405.41),(1, 0, 0, 1, 142.11, 405.41),(1, 0, 0, 1, 130.3, 405.41),(1, 0, 0, 1, 118.49, 405.41),(1, 0, 0, 1, 106.68, 405.41),(1, 0, 0, 1, 94.86, 405.41),(1, 0, 0, 1, 84.05, 405.41),(1, 0, 0, 1, 178.55, 393.59),(1, 0, 0, 1, 165.74, 393.59),(1, 0, 0, 1, 153.93, 393.59),(1, 0, 0, 1, 142.11, 393.59),(1, 0, 0, 1, 131.3, 393.59),(1, 0, 0, 1, 118.49, 393.59),(1, 0, 0, 1, 107.68, 393.59),(1, 0, 0, 1, 83.05, 393.59),(1, 0, 0, 1, 177.55, 381.78),(1, 0, 0, 1, 165.74, 381.78),(1, 0, 0, 1, 153.93, 381.78),(1, 0, 0, 1, 142.11, 381.78),(1, 0, 0, 1, 131.3, 381.78),(1, 0, 0, 1, 119.49, 381.78),(1, 0, 0, 1, 107.68, 381.78),(1, 0, 0, 1, 84.05, 381.78),(1, 0, 0, 1, 166.74, 369.97),(1, 0, 0, 1, 153.93, 369.97),(1, 0, 0, 1, 142.11, 369.97),(1, 0, 0, 1, 130.3, 369.97),(1, 0, 0, 1, 118.49, 369.97),(1, 0, 0, 1, 106.68, 369.97),(1, 0, 0, 1, 94.86, 369.97),(1, 0, 0, 1, 84.05, 369.97),(1, 0, 0, 1, 153.93, 358.16),(1, 0, 0, 1, 142.11, 358.16),(1, 0, 0, 1, 130.3, 358.16),(1, 0, 0, 1, 118.49, 358.16),(1, 0, 0, 1, 107.68, 358.16),(1, 0, 0, 1, 95.86, 358.16),(1, 0, 0, 1, 142.11, 346.34),(1, 0, 0, 1, 130.3, 346.34),(1, 0, 0, 1, 118.49, 346.34),(1, 0, 0, 1, 107.68, 346.34)))
    MATRICES_TD_242_OD = np.array(((1, 0, 0, 1, 106.68, 429.03),(1, 0, 0, 1, 118.49, 429.03),(1, 0, 0, 1, 130.3, 429.03),(1, 0, 0, 1, 142.11, 429.03),(1, 0, 0, 1, 94.86, 417.22),(1, 0, 0, 1, 106.68, 417.22),(1, 0, 0, 1, 118.49, 417.22),(1, 0, 0, 1, 130.3, 417.22),(1, 0, 0, 1, 142.11, 417.22),(1, 0, 0, 1, 153.93, 417.22),(1, 0, 0, 1, 83.05, 405.41),(1, 0, 0, 1, 94.86, 405.41),(1, 0, 0, 1, 106.68, 405.41),(1, 0, 0, 1, 119.49, 405.41),(1, 0, 0, 1, 130.3, 405.41),(1, 0, 0, 1, 142.11, 405.41),(1, 0, 0, 1, 153.93, 405.41),(1, 0, 0, 1, 165.74, 405.41),(1, 0, 0, 1, 71.24, 393.59),(1, 0, 0, 1, 83.05, 393.59),(1, 0, 0, 1, 94.86, 393.59),(1, 0, 0, 1, 106.68, 393.59),(1, 0, 0, 1, 118.49, 393.59),(1, 0, 0, 1, 131.3, 393.59),(1, 0, 0, 1, 143.11, 393.59),(1, 0, 0, 1, 165.74, 393.59),(1, 0, 0, 1, 71.24, 381.78),(1, 0, 0, 1, 83.05, 381.78),(1, 0, 0, 1, 95.86, 381.78),(1, 0, 0, 1, 106.68, 381.78),(1, 0, 0, 1, 119.49, 381.78),(1, 0, 0, 1, 131.3, 381.78),(1, 0, 0, 1, 142.11, 381.78),(1, 0, 0, 1, 165.74, 381.78),(1, 0, 0, 1, 83.05, 369.97),(1, 0, 0, 1, 95.86, 369.97),(1, 0, 0, 1, 107.68, 369.97),(1, 0, 0, 1, 118.49, 369.97),(1, 0, 0, 1, 130.3, 369.97),(1, 0, 0, 1, 142.11, 369.97),(1, 0, 0, 1, 153.93, 369.97),(1, 0, 0, 1, 165.74, 369.97),(1, 0, 0, 1, 94.86, 358.16),(1, 0, 0, 1, 106.68, 358.16),(1, 0, 0, 1, 118.49, 358.16),(1, 0, 0, 1, 130.3, 358.16),(1, 0, 0, 1, 143.11, 358.16),(1, 0, 0, 1, 154.93, 358.16),(1, 0, 0, 1, 107.68, 346.34),(1, 0, 0, 1, 118.49, 346.34),(1, 0, 0, 1, 130.3, 346.34),(1, 0, 0, 1, 143.11, 346.34)))
    MATRICES_PD_242_OS = np.array(((1, 0, 0, 1, 325.86, 429.03),(1, 0, 0, 1, 315.05, 429.03),(1, 0, 0, 1, 303.24, 429.03),(1, 0, 0, 1, 290.43, 429.03),(1, 0, 0, 1, 337.68, 417.22),(1, 0, 0, 1, 325.86, 417.22),(1, 0, 0, 1, 314.05, 417.22),(1, 0, 0, 1, 302.24, 417.22),(1, 0, 0, 1, 291.42, 417.22),(1, 0, 0, 1, 278.61, 417.22),(1, 0, 0, 1, 349.49, 405.41),(1, 0, 0, 1, 337.68, 405.41),(1, 0, 0, 1, 325.86, 405.41),(1, 0, 0, 1, 314.05, 405.41),(1, 0, 0, 1, 302.24, 405.41),(1, 0, 0, 1, 290.43, 405.41),(1, 0, 0, 1, 278.61, 405.41),(1, 0, 0, 1, 267.8, 405.41),(1, 0, 0, 1, 362.3, 393.59),(1, 0, 0, 1, 349.49, 393.59),(1, 0, 0, 1, 337.68, 393.59),(1, 0, 0, 1, 325.86, 393.59),(1, 0, 0, 1, 315.05, 393.59),(1, 0, 0, 1, 302.24, 393.59),(1, 0, 0, 1, 291.42, 393.59),(1, 0, 0, 1, 266.8, 393.59),(1, 0, 0, 1, 361.3, 381.78),(1, 0, 0, 1, 349.49, 381.78),(1, 0, 0, 1, 337.68, 381.78),(1, 0, 0, 1, 325.86, 381.78),(1, 0, 0, 1, 315.05, 381.78),(1, 0, 0, 1, 303.24, 381.78),(1, 0, 0, 1, 291.42, 381.78),(1, 0, 0, 1, 267.8, 381.78),(1, 0, 0, 1, 350.49, 369.97),(1, 0, 0, 1, 337.68, 369.97),(1, 0, 0, 1, 325.86, 369.97),(1, 0, 0, 1, 314.05, 369.97),(1, 0, 0, 1, 302.24, 369.97),(1, 0, 0, 1, 290.43, 369.97),(1, 0, 0, 1, 278.61, 369.97),(1, 0, 0, 1, 267.8, 369.97),(1, 0, 0, 1, 337.68, 358.16),(1, 0, 0, 1, 325.86, 358.16),(1, 0, 0, 1, 314.05, 358.16),(1, 0, 0, 1, 302.24, 358.16),(1, 0, 0, 1, 291.42, 358.16),(1, 0, 0, 1, 279.61, 358.16),(1, 0, 0, 1, 325.86, 346.34),(1, 0, 0, 1, 314.05, 346.34),(1, 0, 0, 1, 302.24, 346.34),(1, 0, 0, 1, 291.42, 346.34)))
    MATRICES_PD_242_OD = np.array(((1, 0, 0, 1, 290.43, 429.03),(1, 0, 0, 1, 302.24, 429.03),(1, 0, 0, 1, 314.05, 429.03),(1, 0, 0, 1, 325.86, 429.03),(1, 0, 0, 1,278.61, 417.22),(1, 0, 0, 1, 290.43, 417.22),(1, 0, 0, 1, 302.24, 417.22),(1, 0, 0, 1, 314.05, 417.22),(1, 0, 0, 1, 325.86, 417.22),(1,0, 0, 1, 337.68, 417.22),(1, 0, 0, 1, 266.8, 405.41),(1, 0, 0, 1, 278.61, 405.41),(1, 0, 0, 1, 290.43, 405.41),(1, 0, 0, 1, 303.24,405.41),(1, 0, 0, 1, 314.05, 405.41),(1, 0, 0, 1, 325.86, 405.41),(1, 0, 0, 1, 337.68, 405.41),(1, 0, 0, 1, 349.49, 405.41),(1, 0, 0,1, 254.99, 393.59),(1, 0, 0, 1, 266.8, 393.59),(1, 0, 0, 1, 278.61, 393.59),(1, 0, 0, 1, 290.43, 393.59),(1, 0, 0, 1, 302.24,393.59),(1, 0, 0, 1, 315.05, 393.59),(1, 0, 0, 1, 326.86, 393.59),(1, 0, 0, 1, 349.49, 393.59),(1, 0, 0, 1, 254.99, 381.78),(1, 0, 0,1, 266.8, 381.78),(1, 0, 0, 1, 279.61, 381.78),(1, 0, 0, 1, 290.43, 381.78),(1, 0, 0, 1, 303.24, 381.78),(1, 0, 0, 1, 315.05,381.78),(1, 0, 0, 1, 325.86, 381.78),(1, 0, 0, 1, 349.49, 381.78),(1, 0, 0, 1, 266.8, 369.97),(1, 0, 0, 1, 279.61, 369.97),(1, 0, 0, 1,291.42, 369.97),(1, 0, 0, 1, 302.24, 369.97),(1, 0, 0, 1, 314.05, 369.97),(1, 0, 0, 1, 325.86, 369.97),(1, 0, 0, 1, 337.68, 369.97),(1,0, 0, 1, 349.49, 369.97),(1, 0, 0, 1, 278.61, 358.16),(1, 0, 0, 1, 290.43, 358.16),(1, 0, 0, 1, 302.24, 358.16),(1, 0, 0, 1, 314.05,358.16),(1, 0, 0, 1, 326.86, 358.16),(1, 0, 0, 1, 338.67, 358.16),(1, 0, 0, 1, 291.42, 346.34),(1, 0, 0, 1, 302.24, 346.34),(1, 0, 0,1, 314.05, 346.34),(1, 0, 0, 1, 326.86, 346.34)))
    MATRICES_TDP_242_OS = np.array(((7.31, 0.0, 0.0, 7.31, 140.63, 293.63),(7.31, 0.0, 0.0, 7.31, 128.81, 293.63),(7.31, 0.0, 0.0, 7.31, 117.0, 293.63),(7.31, 0.0, 0.0, 7.31, 105.19, 293.63),(7.31, 0.0, 0.0, 7.31, 152.44, 281.81),(7.31, 0.0, 0.0, 7.31, 140.63, 281.81),(7.31, 0.0, 0.0, 7.31, 128.81, 281.81),(7.31, 0.0, 0.0, 7.31, 117.0, 281.81),(7.31, 0.0, 0.0, 7.31, 105.19, 281.81),(7.31, 0.0, 0.0, 7.31, 93.38, 281.81),(7.31, 0.0, 0.0, 7.31, 164.25, 270.0),(7.31, 0.0, 0.0, 7.31, 152.44, 270.0),(7.31, 0.0, 0.0, 7.31, 140.63, 270.0),(7.31, 0.0, 0.0, 7.31, 128.81, 270.0),(7.31, 0.0, 0.0, 7.31, 117.0, 270.0),(7.31, 0.0, 0.0, 7.31, 105.19, 270.0),(7.31, 0.0, 0.0, 7.31, 93.38, 270.0),(7.31, 0.0, 0.0, 7.31, 81.56, 270.0),(7.31, 0.0, 0.0, 7.31, 176.06, 258.19),(7.31, 0.0, 0.0, 7.31, 164.25, 258.19),(7.31, 0.0, 0.0, 7.31, 152.44, 258.19),(7.31, 0.0, 0.0, 7.31, 140.63, 258.19),(7.31, 0.0, 0.0, 7.31, 128.81, 258.19),(7.31, 0.0, 0.0, 7.31, 117.0, 258.19),(7.31, 0.0, 0.0, 7.31, 105.19, 258.19),(7.31, 0.0, 0.0, 7.31, 81.56, 258.19),(7.31, 0.0, 0.0, 7.31, 176.06, 246.38),(7.31, 0.0, 0.0, 7.31, 164.25, 246.38),(7.31, 0.0, 0.0, 7.31, 152.44, 246.38),(7.31, 0.0, 0.0, 7.31, 140.63, 246.38),(7.31, 0.0, 0.0, 7.31, 128.81, 246.38),(7.31, 0.0, 0.0, 7.31, 117.0, 246.38),(7.31, 0.0, 0.0, 7.31, 105.19, 246.38),(7.31, 0.0, 0.0, 7.31, 81.56, 246.38),(7.31, 0.0, 0.0, 7.31, 164.25, 234.56),(7.31, 0.0, 0.0, 7.31, 152.44, 234.56),(7.31, 0.0, 0.0, 7.31, 140.63, 234.56),(7.31, 0.0, 0.0, 7.31, 128.81, 234.56),(7.31, 0.0, 0.0, 7.31, 117.0, 234.56),(7.31, 0.0, 0.0, 7.31, 105.19, 234.56),(7.31, 0.0, 0.0, 7.31, 93.38, 234.56),(7.31, 0.0, 0.0, 7.31, 81.56, 234.56),(7.31, 0.0, 0.0, 7.31, 152.44, 222.75),(7.31, 0.0, 0.0, 7.31, 140.63, 222.75),(7.31, 0.0, 0.0, 7.31, 128.81, 222.75),(7.31, 0.0, 0.0, 7.31, 117.0, 222.75),(7.31, 0.0, 0.0, 7.31, 105.19, 222.75),(7.31, 0.0, 0.0, 7.31, 93.38, 222.75),(7.31, 0.0, 0.0, 7.31, 140.63, 210.94),(7.31, 0.0, 0.0, 7.31, 128.81, 210.94),(7.31, 0.0, 0.0, 7.31, 117.0, 210.94),(7.31, 0.0, 0.0, 7.31, 105.19, 210.94)))
    MATRICES_TDP_242_OD = np.array(((7.31, 0.0, 0.0, 7.31, 105.19, 293.63),(7.31, 0.0, 0.0, 7.31, 117.0, 293.63),(7.31, 0.0, 0.0, 7.31, 128.81, 293.63),(7.31, 0.0, 0.0, 7.31, 140.63, 293.63),(7.31, 0.0, 0.0, 7.31, 93.38, 281.81),(7.31, 0.0, 0.0, 7.31, 105.19, 281.81),(7.31, 0.0, 0.0, 7.31, 117.0, 281.81),(7.31, 0.0, 0.0, 7.31, 128.81, 281.81),(7.31, 0.0, 0.0, 7.31, 140.63, 281.81),(7.31, 0.0, 0.0, 7.31, 152.44, 281.81),(7.31, 0.0, 0.0, 7.31, 81.56, 270.0),(7.31, 0.0, 0.0, 7.31, 93.38, 270.0),(7.31, 0.0, 0.0, 7.31, 105.19, 270.0),(7.31, 0.0, 0.0, 7.31, 117.0, 270.0),(7.31, 0.0, 0.0, 7.31, 128.81, 270.0),(7.31, 0.0, 0.0, 7.31, 140.63, 270.0),(7.31, 0.0, 0.0, 7.31, 152.44, 270.0),(7.31, 0.0, 0.0, 7.31, 164.25, 270.0),(7.31, 0.0, 0.0, 7.31, 69.75, 258.19),(7.31, 0.0, 0.0, 7.31, 81.56, 258.19),(7.31, 0.0, 0.0, 7.31, 93.38, 258.19),(7.31, 0.0, 0.0, 7.31, 105.19, 258.19),(7.31, 0.0, 0.0, 7.31, 117.0, 258.19),(7.31, 0.0, 0.0, 7.31, 128.81, 258.19),(7.31, 0.0, 0.0, 7.31, 140.63, 258.19),(7.31, 0.0, 0.0, 7.31, 164.25, 258.19),(7.31, 0.0, 0.0, 7.31, 69.75, 246.38),(7.31, 0.0, 0.0, 7.31, 81.56, 246.38),(7.31, 0.0, 0.0, 7.31, 93.38, 246.38),(7.31, 0.0, 0.0, 7.31, 105.19, 246.38),(7.31, 0.0, 0.0, 7.31, 117.0, 246.38),(7.31, 0.0, 0.0, 7.31, 128.81, 246.38),(7.31, 0.0, 0.0, 7.31, 140.63, 246.38),(7.31, 0.0, 0.0, 7.31, 164.25, 246.38),(7.31, 0.0, 0.0, 7.31, 81.56, 234.56),(7.31, 0.0, 0.0, 7.31, 93.38, 234.56),(7.31, 0.0, 0.0, 7.31, 105.19, 234.56),(7.31, 0.0, 0.0, 7.31, 117.0, 234.56),(7.31, 0.0, 0.0, 7.31, 128.81, 234.56),(7.31, 0.0, 0.0, 7.31, 140.63, 234.56),(7.31, 0.0, 0.0, 7.31, 152.44, 234.56),(7.31, 0.0, 0.0, 7.31, 164.25, 234.56),(7.31, 0.0, 0.0, 7.31, 93.38, 222.75),(7.31, 0.0, 0.0, 7.31, 105.19, 222.75),(7.31, 0.0, 0.0, 7.31, 117.0, 222.75),(7.31, 0.0, 0.0, 7.31, 128.81, 222.75),(7.31, 0.0, 0.0, 7.31, 140.63, 222.75),(7.31, 0.0, 0.0, 7.31, 152.44, 222.75),(7.31, 0.0, 0.0, 7.31, 105.19, 210.94),(7.31, 0.0, 0.0, 7.31, 117.0, 210.94),(7.31, 0.0, 0.0, 7.31, 128.81, 210.94),(7.31, 0.0, 0.0, 7.31, 140.63, 210.94)))
    MATRICES_PDP_242_OS = np.array(((7.31, 0.0, 0.0, 7.31, 324.38, 293.63),(7.31, 0.0, 0.0, 7.31, 312.56, 293.63),(7.31, 0.0, 0.0, 7.31, 300.75, 293.63),(7.31, 0.0, 0.0, 7.31, 288.94, 293.63),(7.31, 0.0, 0.0, 7.31, 336.19, 281.81),(7.31, 0.0, 0.0, 7.31, 324.38, 281.81),(7.31, 0.0, 0.0, 7.31, 312.56, 281.81),(7.31, 0.0, 0.0, 7.31, 300.75, 281.81),(7.31, 0.0, 0.0, 7.31, 288.94, 281.81),(7.31, 0.0, 0.0, 7.31, 277.13, 281.81),(7.31, 0.0, 0.0, 7.31, 348.0, 270.0),(7.31, 0.0, 0.0, 7.31, 336.19, 270.0),(7.31, 0.0, 0.0, 7.31, 324.38, 270.0),(7.31, 0.0, 0.0, 7.31, 312.56, 270.0),(7.31, 0.0, 0.0, 7.31, 300.75, 270.0),(7.31, 0.0, 0.0, 7.31, 288.94, 270.0),(7.31, 0.0, 0.0, 7.31, 277.13, 270.0),(7.31, 0.0, 0.0, 7.31, 265.31, 270.0),(7.31, 0.0, 0.0, 7.31, 359.81, 258.19),(7.31, 0.0, 0.0, 7.31, 348.0, 258.19),(7.31, 0.0, 0.0, 7.31, 336.19, 258.19),(7.31, 0.0, 0.0, 7.31, 324.38, 258.19),(7.31, 0.0, 0.0, 7.31, 312.56, 258.19),(7.31, 0.0, 0.0, 7.31, 300.75, 258.19),(7.31, 0.0, 0.0, 7.31, 288.94, 258.19),(7.31, 0.0, 0.0, 7.31, 265.31, 258.19),(7.31, 0.0, 0.0, 7.31, 359.81, 246.38),(7.31, 0.0, 0.0, 7.31, 348.0, 246.38),(7.31, 0.0, 0.0, 7.31, 336.19, 246.38),(7.31, 0.0, 0.0, 7.31, 324.38, 246.38),(7.31, 0.0, 0.0, 7.31, 312.56, 246.38),(7.31, 0.0, 0.0, 7.31, 300.75, 246.38),(7.31, 0.0, 0.0, 7.31, 288.94, 246.38),(7.31, 0.0, 0.0, 7.31, 265.31, 246.38),(7.31, 0.0, 0.0, 7.31, 348.0, 234.56),(7.31, 0.0, 0.0, 7.31, 336.19, 234.56),(7.31, 0.0, 0.0, 7.31, 324.38, 234.56),(7.31, 0.0, 0.0, 7.31, 312.56, 234.56),(7.31, 0.0, 0.0, 7.31, 300.75, 234.56),(7.31, 0.0, 0.0, 7.31, 288.94, 234.56),(7.31, 0.0, 0.0, 7.31, 277.13, 234.56),(7.31, 0.0, 0.0, 7.31, 265.31, 234.56),(7.31, 0.0, 0.0, 7.31, 336.19, 222.75),(7.31, 0.0, 0.0, 7.31, 324.38, 222.75),(7.31, 0.0, 0.0, 7.31, 312.56, 222.75),(7.31, 0.0, 0.0, 7.31, 300.75, 222.75),(7.31, 0.0, 0.0, 7.31, 288.94, 222.75),(7.31, 0.0, 0.0, 7.31, 277.13, 222.75),(7.31, 0.0, 0.0, 7.31, 324.38, 210.94),(7.31, 0.0, 0.0, 7.31, 312.56, 210.94),(7.31, 0.0, 0.0, 7.31, 300.75, 210.94),(7.31, 0.0, 0.0, 7.31, 288.94, 210.94)))
    MATRICES_PDP_242_OD = np.array(((7.31, 0.0, 0.0, 7.31, 288.94, 293.63),(7.31, 0.0, 0.0, 7.31, 300.75, 293.63),(7.31, 0.0, 0.0, 7.31, 312.56, 293.63),(7.31, 0.0, 0.0, 7.31, 324.38, 293.63),(7.31, 0.0, 0.0, 7.31, 277.13, 281.81),(7.31, 0.0, 0.0, 7.31, 288.94, 281.81),(7.31, 0.0, 0.0, 7.31, 300.75, 281.81),(7.31, 0.0, 0.0, 7.31, 312.56, 281.81),(7.31, 0.0, 0.0, 7.31, 324.38, 281.81),(7.31, 0.0, 0.0, 7.31, 336.19, 281.81),(7.31, 0.0, 0.0, 7.31, 265.31, 270.0),(7.31, 0.0, 0.0, 7.31, 277.13, 270.0),(7.31, 0.0, 0.0, 7.31, 288.94, 270.0),(7.31, 0.0, 0.0, 7.31, 300.75, 270.0),(7.31, 0.0, 0.0, 7.31, 312.56, 270.0),(7.31, 0.0, 0.0, 7.31, 324.38, 270.0),(7.31, 0.0, 0.0, 7.31, 336.19, 270.0),(7.31, 0.0, 0.0, 7.31, 348.0, 270.0),(7.31, 0.0, 0.0, 7.31, 253.5, 258.19),(7.31, 0.0, 0.0, 7.31, 265.31, 258.19),(7.31, 0.0, 0.0, 7.31, 277.13, 258.19),(7.31, 0.0, 0.0, 7.31, 288.94, 258.19),(7.31, 0.0, 0.0, 7.31, 300.75, 258.19),(7.31, 0.0, 0.0, 7.31, 312.56, 258.19),(7.31, 0.0, 0.0, 7.31, 324.38, 258.19),(7.31, 0.0, 0.0, 7.31, 348.0, 258.19),(7.31, 0.0, 0.0, 7.31, 253.5, 246.38),(7.31, 0.0, 0.0, 7.31, 265.31, 246.38),(7.31, 0.0, 0.0, 7.31, 277.13, 246.38),(7.31, 0.0, 0.0, 7.31, 288.94, 246.38),(7.31, 0.0, 0.0, 7.31, 300.75, 246.38),(7.31, 0.0, 0.0, 7.31, 312.56, 246.38),(7.31, 0.0, 0.0, 7.31, 324.38, 246.38),(7.31, 0.0, 0.0, 7.31, 348.0, 246.38),(7.31, 0.0, 0.0, 7.31, 265.31, 234.56),(7.31, 0.0, 0.0, 7.31, 277.13, 234.56),(7.31, 0.0, 0.0, 7.31, 288.94, 234.56),(7.31, 0.0, 0.0, 7.31, 300.75, 234.56),(7.31, 0.0, 0.0, 7.31, 312.56, 234.56),(7.31, 0.0, 0.0, 7.31, 324.38, 234.56),(7.31, 0.0, 0.0, 7.31, 336.19, 234.56),(7.31, 0.0, 0.0, 7.31, 348.0, 234.56),(7.31, 0.0, 0.0, 7.31, 277.13, 222.75),(7.31, 0.0, 0.0, 7.31, 288.94, 222.75),(7.31, 0.0, 0.0, 7.31, 300.75, 222.75),(7.31, 0.0, 0.0, 7.31, 312.56, 222.75),(7.31, 0.0, 0.0, 7.31, 324.38, 222.75),(7.31, 0.0, 0.0, 7.31, 336.19, 222.75),(7.31, 0.0, 0.0, 7.31, 288.94, 210.94),(7.31, 0.0, 0.0, 7.31, 300.75, 210.94),(7.31, 0.0, 0.0, 7.31, 312.56, 210.94),(7.31, 0.0, 0.0, 7.31, 324.38, 210.94)))

    def __init__(self, fp):
        fp.seek(0)
        self.raw_pdf = fp.read()  # Save a copy in memory for anonymization

        parser = PDFParser(BytesIO(self.raw_pdf))
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = HFASFADevice(rsrcmgr)  # TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

        self.byte_sequences = device.byte_sequences
        self.text_sequences = device.text_sequences
        self._matrices = np.array([x.matrix for x in device.render_items])
        self._device = device

    @property
    def regex_match(self):
        return HFAPDFParser.REGEX_COMPILED_242.search("\n".join(map(str, self._device.render_items)))

    def anonymize(self, anonymization_fun=lambda x: b""):
        import subprocess
        uncompress_process = subprocess.run(["pdftk", "-", "output", "-", "uncompress"],
                                            input=self.raw_pdf, capture_output=True)
        uncompressed_pdf = uncompress_process.stdout

        # Sometimes there are extra b"\\" (i.e. chr(92)) in uncompressed_pdf but not in snippet
        # For example, the original PDF is b'\x00*\x005\x00$\x00\\)\x00...'
        # but the snippet that we have is b"\x00*\x005\x00$\x00)\x00)..."
        # I am not sure why this is happening, as PDF is very complicated
        # Below is a not so efficient hack, but hopefully works
        # Find all PDF bytes that represent literal strings
        import re
        snippet_dict = {m.group(0).replace(b"\\", b""): m.group(0) for m in re.finditer(rb"\(.*\)Tj", uncompressed_pdf)}

        for snippet in (self.byte_sequences[self.text_sequences.index("Patient:") + 1],
                        self.byte_sequences[self.text_sequences.index("Patient ID:") + 1],
                        self.byte_sequences[self.text_sequences.index("Date of Birth:") + 1],
                        ):
            value = anonymization_fun(snippet)
            if isinstance(value, str):
                value = value.encode("UTF-8")

            if b"("+snippet+b")Tj" in snippet_dict:
                uncompressed_pdf = uncompressed_pdf.replace(snippet_dict[b"("+snippet+b")Tj"], b"("+value+b")Tj")
            elif b"("+snippet+b")Tj" in uncompressed_pdf:
                uncompressed_pdf = uncompressed_pdf.replace(b"("+snippet+b")Tj", b"("+value+b")Tj")
            else:
                raise ValueError("PDF anonymization failed: Cannot locate byte sequence to remove in PDF: " + repr(snippet))
            # Since this PDF is passing through pdftk compression again,
            # we actually don't have to maintain the same byte length

        anonymized_process = subprocess.run(["pdftk", "-", "output", "-", "compress"],
                                            input=uncompressed_pdf, capture_output=True)
        return HFAPDFParser(BytesIO(anonymized_process.stdout))

    def get_value(self, key, offset=1):
        if key not in self.text_sequences:
            _logger.debug("Cannot find %s in text_sequences. (PDF may have been modified/anonymized)", repr(key))
            return ""

        # key exists in self.text_sequences
        key_index = self.text_sequences.index(key)
        # value = self.text_sequences[key_index + offset]
        real_offset = offset
        i = 0
        while i < real_offset:
            i += 1
            if self.text_sequences[key_index + i] == ' ':
                real_offset += 1
        return self.text_sequences[key_index + real_offset]

    def get_value_list(self, key, offset_start=1, length=1):
        key_index = self.text_sequences.index(key)
        value = self.text_sequences[key_index + offset_start:key_index + offset_start + length]
        return value

    def get_value_try_multiple_methods(self, kwargs_list, validate_fun):
        """
        For slightly different PDF formats, some key locators may not work.
        This is a helper method wrapping the get_value method
        to conveniently retry different ways if invalid values were obtained
        """
        assert len(kwargs_list) > 0, "Must provide at least one kwargs in kwargs_list"
        success = False

        for kwargs in kwargs_list:
            try:
                value = self.get_value(**kwargs)
            except ValueError as e:
                _logger.debug("Get value failed with %s, continue: %s", kwargs, e)
                continue

            success = validate_fun(value)
            if success:  # Returns True if it is a valid value
                break
            else:
                continue

        assert success, f"Failed on all get value options: {str(kwargs_list)}. Last value was {repr(value)}."
        # noinspection PyUnboundLocalVariable
        return value

    def get_value_from_matrix(self, matrix, tolerance=4.0):
        """
        Find the RenderItem that is closest to the provided matrix
        and return its value as string.
        The match only looks at the x, y coordinates
        in the last two elements of the matrices and uses Euclidian distance

        Parameters
        ----------
        matrix : Tuple[float]
            tuple of 6 elements representing a PDF matrix, approximate location of the RenderItem

        tolerance : float
            maximum accepted distance of the found object to the provided matrix.
            If the distance metric is higher than the tolerance, then empty string will be returned

        Returns
        -------
        str
            Value of the PDF RenderItem that is closest to the provided matrix
        """
        diff = self._matrices[:, 4:6] - matrix[4:6]
        distances = np.sqrt((diff**2).sum(axis=1))
        argmindist = np.argmin(distances)
        mindist = min(distances)
        matched = self._device.render_items[argmindist]
        if mindist < tolerance:
            return str(matched)
        else:
            _logger.error("Could not find eligible item that is less than tolerance = %g. The closest match with distance %g is: %s.",
                          tolerance, mindist, repr(matched))
            return ""

    @property
    def name(self):
        return self.get_value("Patient:")

    @property
    def dob(self):
        return self.get_value("Date of Birth:")

    @property
    def gender(self):
        return self.get_value("Gender:")

    @property
    def id(self):
        return self.get_value("Patient ID:")

    @property
    def laterality(self):
        value = self.get_value_try_multiple_methods((
            {"key": "Patient ID:", "offset": 2},
            {"key": "Version", "offset": -4},
            {"key": "Single Field Analysis", "offset": -1}
        ), validate_fun=lambda value: value == "OS" or value == "OD")
        return value

    @property
    def report_type(self):
        # assert value == "Single Field Analysis", f"Report type {value} currently not supported"
        value = self.get_value_try_multiple_methods((
            {"key": "Patient ID:", "offset": 3},
            {"key": "Version", "offset": -3},
            {"key": "Central 24-2 Threshold Test", "offset": -1}
        ), validate_fun=lambda value: value == "Single Field Analysis")
        return value

    @property
    def pattern(self):
        value = self.get_value_try_multiple_methods((
            {"key": "Patient ID:", "offset": 4},
            {"key": "Version", "offset": -2},
            {"key": "Single Field Analysis", "offset": 1}
        ), validate_fun=lambda value: "24-2" in value or "10-2" in value or "30-2" in value)
        return value

    @property
    def fixation_monitor(self):
        return self.get_value("Fixation Monitor:", offset=7)

    @property
    def fixation_target(self):
        return self.get_value("Fixation Target:", offset=7)

    @property
    def fixation_loss_error(self):
        return int(self.fixation_loss.split()[0].split("/")[0])

    @property
    def fixation_loss_total(self):
        return int(self.fixation_loss.split()[0].split("/")[1])

    @property
    def fixation_loss(self):
        return self.get_value("Fixation Losses:", offset=7)

    @property
    def false_positive(self):
        value = self.get_value("False POS Errors:", offset=7)
        if value == 'N/A':
            return "nan"
        return value

    @property
    def false_negative(self):
        value = self.get_value("False NEG Errors:", offset=7)
        if value == 'N/A':
            return "nan"
        return value

    @property
    def test_duration(self):
        value = self.get_value("Test Duration:", offset=7)
        t = datetime.strptime(value, "%M:%S")
        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        return delta.total_seconds()

    @property
    def fovea(self):
        return self.get_value("Fovea:", offset=7)

    @property
    def stimulus(self):
        return self.get_value("Stimulus:", offset=6)

    @property
    def background(self):
        return self.get_value("Background:", offset=6)

    @property
    def strategy(self):
        return self.get_value("Strategy:", offset=6)

    @property
    def pupil_diameter(self):
        return self.get_value("Pupil Diameter:", offset=6)

    @property
    def visual_acuity(self):
        if self.get_value("Visual Acuity:", offset=7) == "Date:":
            return None  # Visual acuity was skipped
        else:
            return self.get_value("Visual Acuity:", offset=6)

    @property
    def rx(self):
        return self.get_value("Date:", offset=-1)

    @property
    def date(self):
        # value = self.get_value("Date:", offset=3)
        value = self.get_value_from_matrix((1.0, 0.0, 0.0, 1.0, 504.5, 688.25))
        value = value.replace("\n", "")
        dt = datetime.strptime(value, "%b %d, %Y")
        return dt.date()

    @property
    def time(self):
        # value = self.get_value("Time:", offset=3)
        value = self.get_value_from_matrix((1.0, 0.0, 0.0, 1.0, 504.5, 677.75))
        value = value.replace("\n", "")
        dt = datetime.strptime(value, "%I:%M %p")
        return dt.time()

    @property
    def age(self):
        # value = self.get_value("Age:", offset=3)
        value = self.get_value_from_matrix((1.0, 0.0, 0.0, 1.0, 504.5, 667.25))
        value = float(value)
        assert 0 <= value <= 130
        return value

    @property
    def n_vf_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 54
        else:
            raise NotImplementedError(f"n_vf_loc is not yet implemented for {self.pattern}")

    @property
    def n_td_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 52
        else:
            raise NotImplementedError(f"n_td_loc is not yet implemented for {self.pattern}")

    @property
    def vf(self):
        """
        Update: Now uses the pre-generated locator matrices to parse 24-2 visual field

        Returns
        -------
        A list of visual field thresholds as float. "<0" are converted to -1.0
        """
        if self.pattern == "Central 24-2 Threshold Test":
            # Use the locator matrices to parse
            if self.laterality == "OD":
                matrices = HFAPDFParser.MATRICES_VF_242_OD
            elif self.laterality == "OS":
                matrices = HFAPDFParser.MATRICES_VF_242_OS
            else:
                assert False, f"Invalid self.laterality == {self.laterality} encountered in vf parsing"
            value_list = [self.get_value_from_matrix(m) for m in matrices]
        else:
            # Old method of using keywords
            if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
                value_list = self.get_value_list("Total Deviation", offset_start=-4-self.n_td_loc-self.n_vf_loc, length=self.n_vf_loc)
            else:
                value_list = self.get_value_list("Total Deviation", offset_start=-self.n_td_loc*2-self.n_vf_loc, length=self.n_vf_loc)
        values = [float(i) if i != "<0" else -1.0 for i in value_list]
        assert all(map(lambda x: x >= -1, values))
        return values

    @property
    def td(self):
        if self.pattern == "Central 24-2 Threshold Test":
            # Use the locator matrices to parse
            if self.laterality == "OD":
                matrices = HFAPDFParser.MATRICES_TD_242_OD
            elif self.laterality == "OS":
                matrices = HFAPDFParser.MATRICES_TD_242_OS
            else:
                assert False, f"Invalid self.laterality == {self.laterality} encountered in vf parsing"
            value_list = [self.get_value_from_matrix(m) for m in matrices]
        else:
            # Old method of using keywords
            if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
                value_list = self.get_value_list("Total Deviation", offset_start=-4-self.n_td_loc, length=self.n_td_loc)
            else:
                value_list = self.get_value_list("Total Deviation", offset_start=-self.n_td_loc*2, length=self.n_td_loc)
        try:
            return [float(i) for i in value_list]
        except ValueError as e:
            _logger.warning(f"Please double check results: Flattening list because could not parse: %s", value_list)
            # TODO: If this happens on OS, does it need to be reversed?
            return [float(j) for i in value_list for j in i.split()]

    @property
    def pd(self):
        if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
            return [float("nan")] * self.n_td_loc
        else:
            if self.pattern == "Central 24-2 Threshold Test":
                # Use the locator matrices to parse
                if self.laterality == "OD":
                    matrices = HFAPDFParser.MATRICES_PD_242_OD
                elif self.laterality == "OS":
                    matrices = HFAPDFParser.MATRICES_PD_242_OS
                else:
                    assert False, f"Invalid self.laterality == {self.laterality} encountered in vf parsing"
                value_list = [self.get_value_from_matrix(m) for m in matrices]
            else:
                # Old method of using keywords
                value_list = self.get_value_list("Pattern Deviation", offset_start=-1-self.n_td_loc, length=self.n_td_loc)
            try:
                return [float(i) for i in value_list]
            except ValueError as e:
                _logger.warning(f"Please double check results: Flattening list because could not parse: %s", value_list)
                # TODO: If this happens on OS, does it need to be reversed?
                return [float(j) for i in value_list for j in i.split()]

    @property
    def tdp(self):
        """

        Returns
        -------
        List[float]
            Probability threshold in the total deviation map
        """
        # if self.regex_match is not None:
        #     return list(map(float, self.regex_match.group('tdp').strip().split("\n")))
        if self.pattern == "Central 24-2 Threshold Test":
            # Use the locator matrices to parse
            if self.laterality == "OD":
                matrices = HFAPDFParser.MATRICES_TDP_242_OD
            elif self.laterality == "OS":
                matrices = HFAPDFParser.MATRICES_TDP_242_OS
            else:
                assert False, f"Invalid self.laterality == {self.laterality} encountered in vf parsing"
            value_list = [self.get_value_from_matrix(m) for m in matrices]
            return list(map(float, value_list))
        else:
            _logger.warning("Parsing of TD probability map currently not supported for this file.")
            return [float("nan") for _ in range(self.n_td_loc)]

    @property
    def pdp(self):
        """

        Returns
        -------
        List[float]
            Probability threshold in the pattern deviation map
        """
        if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
            return [float("nan")] * self.n_td_loc
        else:
            # if self.regex_match is not None:
            #     match_lines = self.regex_match.group('pdp').strip().split("\n")
            #     if match_lines[0].strip() == "MD Threshold exceeded.":
            #         return [float("nan") for _ in range(self.n_td_loc)]
            #     return list(map(float, self.regex_match.group('pdp').strip().split("\n")))
            if self.pattern == "Central 24-2 Threshold Test":
                # Use the locator matrices to parse
                if self.laterality == "OD":
                    matrices = HFAPDFParser.MATRICES_PDP_242_OD
                elif self.laterality == "OS":
                    matrices = HFAPDFParser.MATRICES_PDP_242_OS
                else:
                    assert False, f"Invalid self.laterality == {self.laterality} encountered in vf parsing"
                value_list = [self.get_value_from_matrix(m) for m in matrices]
                return list(map(float, value_list))
            else:
                _logger.warning("Parsing of PD probability map currently not supported for this file.")
                return [float("nan") for _ in range(self.n_td_loc)]

    @property
    def ght(self):
        # Sometimes GHT is split across two lines
        pt1 = self.get_value("GHT:", offset=1)
        pt2 = self.get_value("GHT:", offset=2)
        if pt2 == "VFI:":
            return pt1
        else:
            return " ".join((pt1, pt2))

    @property
    def vfi(self):
        return self.get_value("VFI:", offset=1)

    @property
    def md(self):
        for key in ("MD:", "MD24-2:", "MD10-2:", "MD30-2:"):
            value = self.get_value(key, offset=1)
            if value:
                # If we can find a non-empty string
                return value
            else:
                # If not, keep searching
                _logger.debug("Cannot find %s, trying other keys.", key)
                continue

    @property
    def psd(self):
        for key in ("PSD:", "PSD24-2:", "PSD10-2:", "PSD30-2:"):
            value = self.get_value(key, offset=1)
            if value:
                # If we can find a non-empty string
                return value
            else:
                # If not, keep searching
                _logger.debug("Cannot find %s, trying other keys.", key)
                continue


class HFASFADevice(PDFLayoutAnalyzer):
    def __init__(self, rsrcmgr, pageno=1, laparams=None):
        super(HFASFADevice, self).__init__(rsrcmgr, pageno=pageno, laparams=laparams)
        self.byte_sequences = []
        self.text_sequences = []
        self.render_items = []
        self._figure_matrix = utils.MATRIX_IDENTITY

    def render_string(self, textstate, seq, ncs, graphicstate):
        super(HFASFADevice, self).render_string(textstate, seq, ncs, graphicstate)
        matrix = utils.mult_matrix(textstate.matrix, self.ctm)
        self.render_items.append(StringRenderItem(copy(textstate), seq, ncs, graphicstate, matrix))  # Must use copy, otherwise font objet is different later on
        font = textstate.font
        for obj in seq:
            if not isinstance(obj, bytes):
                # For PDF anonymized by VEM's Java software
                # some obj may no long be bytes but become invalid ints...
                # skip them
                _logger.debug("obj = %s is not of bytes type, skipping and appending an empty line", repr(obj))
                obj = b""
            self.byte_sequences.append(obj)
            self.text_sequences.append("".join([font.to_unichr(c) for c in font.decode(obj)]))

    def begin_figure(self, name, bbox, matrix):
        super(HFASFADevice, self).begin_figure(name, bbox, matrix)
        self._figure_matrix = matrix # Usually matrix here is identity

    def render_image(self, name, stream):
        super(HFASFADevice, self).render_image(name, stream)
        matrix = utils.mult_matrix(self._figure_matrix, self.ctm)
        self.render_items.append(ImageRenderItem(name, stream, matrix))


@dataclass(frozen=True, repr=False)
class StringRenderItem:
    textstate: PDFTextState
    seq: List[bytes]
    ncs: PDFColorSpace
    graphicstate: PDFGraphicState
    matrix: Tuple[float]

    @property
    def decoded_seq(self):
        """
        Decode the bytes in seq using the font object.
        Sequences that cannot be parsed as replaced with empty b''.

        Returns
        -------
        List[str]
            List of decoded string
        """
        ret = []
        font = self.textstate.font
        for obj in self.seq:
            if not isinstance(obj, bytes):
                _logger.debug("obj = %s is not of bytes type, skipping and appending an empty line", repr(obj))
                obj = b""
            text = "".join([font.to_unichr(c) for c in font.decode(obj)])
            ret.append(text)
        return tuple(ret)

    def __str__(self):
        return "\n".join(self.decoded_seq)

    def __repr__(self):
        # Order here for convenience of debug, not actual order in constructor
        return f"{self.__class__.__qualname__}(decoded_seq={self.decoded_seq},matrix={self.matrix},textstate={self.textstate},seq={self.seq},ncs={self.ncs},graphicstate={self.graphicstate})"


@dataclass(frozen=True, repr=False)
class ImageRenderItem:
    name: str
    stream: PDFStream
    matrix: Tuple[float]

    # Class attribute storing sha1 hash of image bytes to semantic meaning - pre-generated
    hash2str = {
        "2b3f91b0f6b384ae20fbfd6f056adae48c870ab3": 1.0,  # Decompressed
        "35be72552461ac1bbbf5cd5c6bfaaa4520af6da8": 0.05,  # Decompressed
        "1bfae8881f1ffbdf84ae1eecb1ddcc54e7fa1937": 0.02,  # Decompressed
        "534d50117dd5fb4ae70b05aeec3ec022b703ad3b": 0.01,  # Decompressed
        "6d396f13345b0b4d808c502db975e9d6b2987e88": 0.005,  # Decompressed
    }

    @property
    def decoded_value(self):
        """
        Get the numerical value that represents the semantic meaning of this image,
        if it can be interpreted

        Returns
        -------
        float
            Decoded value of the image
        """
        return ImageRenderItem.hash2str.get(self._get_decoded_image_hash(), float("nan"))

    @property
    def decoded_image(self):
        """
        Get the image representation of this object

        Returns
        -------
        np.ndarray
            Image generated from the stream bytes as a numpy array
        """
        if self.stream.data is None:
            self.stream.decode()
        buffer = self.stream.data
        if len(buffer) == self.stream.get("Length") + 1 and buffer[-1:] == b"\n":
            buffer = buffer[:-1]  # Usually there is an extra byte of b"\n" at the end for uncompressed stream...
        # assert len(buffer) == self.stream.get("Length"), f"Mismatch length of buffer ({len(buffer)}) and length in attribute ({self.stream.get('Length')})"
        w = self.stream.get("Width")
        h = self.stream.get("Height")
        c = self.stream.get("DecodeParms", {}).get("Colors", 1)
        if c == 1:
            # Sometimes in modified files, the image is uncompressed and does not specify how many colors/channels
            # Maybe could use self.stream.get("ColorSpace") == DeviceRGB (Need to find reference of DeviceRGB literal)
            c = len(buffer) // w // h
        bits_per_component = self.stream.get("BitsPerComponent")
        if bits_per_component == 8:
            dtype = np.uint8
        else:
            raise NotImplementedError(f"bits_per_component = {bits_per_component} is not yet implemented")

        # for filter_name, params in self.stream.get_filters():  # See pdfminer\image.py
        #     if filter_name == pdfminer.psparser.LIT("FlateDecode"):
        #         buffer = zlib.decompress(buffer)
        #     else:
        #         raise NotImplementedError(f"filter_name = {filter_name} is unsupported.")
        # # Lacks apply_png_predictor

        # if len(buffer) != w * h * bits_per_component / 8:
        #     _logger.error(f"Mismatch image shape ({w}, {h}) and buffer length ({len(buffer)}). Decoding failed.")
        #     return np.zeros((h, w), dtype=dtype)
        im = np.frombuffer(buffer, dtype=dtype).reshape(h, w, c)
        return im

    def _get_decoded_image_hash(self):
        return hashlib.sha1(self.decoded_image[..., 0].tobytes()).hexdigest()

    def __str__(self):
        return str(self.decoded_value)

    def __repr__(self):
        # Order here for convenience of debug, not actual order in constructor
        return f"{self.__class__.__qualname__}(decoded_value={self.decoded_value},matrix={self.matrix},_get_decoded_image_hash()={self._get_decoded_image_hash()},name={self.name},stream={self.stream})"
