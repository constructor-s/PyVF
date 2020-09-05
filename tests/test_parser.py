"""
Test anonymization and parsing functionality in parse module

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

from pathlib import Path
from unittest import TestCase
import warnings
import pydicom
import pyvf.parse
import pyvf.parse.pdf
import hashlib
from collections import namedtuple


class Test_DCM_Parser(TestCase):
    def test_dcm_parser(self):
        summaries = []
        for test_file in Path("output").glob("2*.dcm"):
            # Test anonymize
            parser = pyvf.parse.parse(test_file)
            parser = parser.anonymize(anonymization_fun=lambda x:
                b"" if isinstance(x, bytes)
                else "Anonymous^Anonymous" if isinstance(x, pydicom.valuerep.PersonName)
                else hashlib.sha1(str(x).zfill(16).encode("UTF-8")).hexdigest())
            parser.save_as("output/output2.dcm")

            # Load the anonymized version back in
            parser = pyvf.parse.parse("output/output2.dcm")  # type: pyvf.parse.dcm.HFADCMParser

            VFSummary = namedtuple("VFSummary", ["TEST_ID", "STUDY_ID", "ROUTINE", "AGE_HFA", "SIDE", "MD_HFA", "PSD_HFA", "VFI_HFA", "GHT_HFA", "FPR_HFA", "FNR_HFA", "FIX_LOSS_HFA", "DATE_HFA", "TIME_HFA", "DURATION_HFA", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19", "L20", "L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30", "L31", "L32", "L33", "L34", "L35", "L36", "L37", "L38", "L39", "L40", "L41", "L42", "L43", "L44", "L45", "L46", "L47", "L48", "L49", "L50", "L51", "L52", "L53", "L54"])

            field = VFSummary(
                TEST_ID=parser.dataset.StudyInstanceUID,
                STUDY_ID=parser.id,
                ROUTINE=parser.pdf_parser.strategy,
                AGE_HFA=(parser.datetime - parser.dob).days,
                SIDE=parser.laterality,
                MD_HFA=parser.md,
                PSD_HFA=parser.psd,
                VFI_HFA=parser.vfi,
                GHT_HFA=parser.ght,
                FPR_HFA=parser.false_positive,
                FNR_HFA=parser.false_negative,
                FIX_LOSS_HFA=parser.fixation_loss,
                DATE_HFA=parser.datetime.date(),
                TIME_HFA=parser.datetime.time().replace(microsecond=0),
                DURATION_HFA=parser.pdf_parser.test_duration,
                **({f"L{i+1}": parser.pdf_parser.vf[i] for i in range(54)})
            )

            summaries.append(field)

            # self.assertEqual(parser.pdf_parser.vf[0], 27)
            # self.assertEqual(parser.pdf_parser.vf[53], 31)
            # self.assertEqual(parser.pdf_parser.td[0], -3)
            # self.assertEqual(parser.pdf_parser.td[51], 0)
            # self.assertEqual(parser.pdf_parser.pd[0], -2)
            # self.assertEqual(parser.pdf_parser.pd[51], 0)
            # self.assertEqual(parser.fixation_loss, 1.0/14.0)

            # pdf_fp = BytesIO(parser.dataset.EncapsulatedDocument)
            # pdf_parser = pyvf.parse.pdf.HFAPDFParser(pdf_fp)
            # print(pdf_parser.md)
            # print(pdf_parser.psd)
            # print(pdf_parser.ght)
            # print(pdf_parser.vfi)

        if summaries:
            import pandas as pd
            vf = pd.DataFrame(summaries)
            vf.to_csv("output/hfa_summaries.csv", index=False)
        else:
            warnings.warn(f"No DCM test case file present; no test performed.")
