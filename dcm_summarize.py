"""
Batch save visual field result indices into a CSV format

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
import glob
import itertools
import warnings
from argparse import ArgumentParser
from collections import namedtuple
import logging
import pandas as pd
import pyvf.parse.dcm

_logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_logger.addHandler(sh)
_logger.setLevel(logging.DEBUG)

argparser = ArgumentParser("Summarize visual field results to CSV")
argparser.add_argument("-i", "--input", type=str, required=True, nargs="+", help="One or more glob patterns of .dcm files")
argparser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file")
args = argparser.parse_args()

VFSummary = namedtuple("VFSummary",
                       ["TEST_ID", "STUDY_ID", "PATTERN", "ROUTINE", "AGE_HFA", "SIDE", "MD_HFA", "PSD_HFA", "VFI_HFA",
                        "GHT_HFA", "FPR_HFA", "FNR_HFA", "FLR_HFA", "DATE_HFA", "TIME_HFA", "DURATION_HFA",
                        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14",
                        "L15", "L16", "L17", "L18", "L19", "L20", "L21", "L22", "L23", "L24", "L25", "L26",
                        "L27", "L28", "L29", "L30", "L31", "L32", "L33", "L34", "L35", "L36", "L37", "L38",
                        "L39", "L40", "L41", "L42", "L43", "L44", "L45", "L46", "L47", "L48", "L49", "L50",
                        "L51", "L52", "L53", "L54"],
               # defaults=["", "", "", "", pd.NA, "", pd.NA, pd.NA, pd.NA,
               #          "", pd.NA, pd.NA, pd.NA, pd.NaT, pd.NaT, pd.NA,
               #          pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA,
               #          pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA,
               #          pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA,
               #          pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA,
               #          pd.NA, pd.NA, pd.NA, pd.NA]
                       )

summaries = []
for test_file in itertools.chain(*map(glob.glob, args.input)):
    parser = pyvf.parse.parse(test_file)  # type: pyvf.parse.dcm.HFADCMParser
    _logger.debug(test_file)
    if parser.pattern != "Central 24-2 Threshold Test":
        _logger.warning("Skipping %s: %s", parser.pattern, test_file)
        continue
    try:
        field = VFSummary(
            TEST_ID=parser.dataset.StudyInstanceUID,
            STUDY_ID=parser.id,
            PATTERN=parser.pattern,
            ROUTINE=parser.strategy,
            AGE_HFA=(parser.datetime - parser.dob).days,
            SIDE=parser.laterality,
            MD_HFA=parser.md,
            PSD_HFA=parser.psd,
            VFI_HFA=parser.vfi,
            GHT_HFA=parser.ght,
            FPR_HFA=parser.false_positive,
            FNR_HFA=parser.false_negative,
            FLR_HFA=parser.fixation_loss,
            DATE_HFA=parser.datetime.date(),
            TIME_HFA=parser.datetime.time().replace(microsecond=0),
            DURATION_HFA=parser.pdf_parser.test_duration,
            **({f"L{i + 1}": parser.pdf_parser.vf[i] for i in range(54)})
        )
    except ValueError:
        field = VFSummary(
            TEST_ID=parser.dataset.StudyInstanceUID,
            STUDY_ID=parser.id,
            PATTERN=parser.pattern,
            ROUTINE=parser.strategy,
            AGE_HFA=(parser.datetime - parser.dob).days,
            SIDE=parser.laterality,
            MD_HFA=parser.md,
            PSD_HFA=parser.psd,
            VFI_HFA=parser.vfi,
            GHT_HFA=parser.ght,
            FPR_HFA=parser.false_positive,
            FNR_HFA=parser.false_negative,
            FLR_HFA=parser.fixation_loss,
            DATE_HFA=parser.datetime.date(),
            TIME_HFA=parser.datetime.time().replace(microsecond=0),
            DURATION_HFA=pd.NA,
            **({f"L{i + 1}": pd.NA for i in range(54)})
        )

    summaries.append(field)

if summaries:
    vf = pd.DataFrame(summaries)
    vf.to_csv(args.output, index=False)
else:
    warnings.warn("No DCM test found in input glob.")
