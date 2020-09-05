"""
Anonymization script for DCM SFA files exported from Zeiss Forum.
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

import pyvf.parse.dcm
import pydicom
import hashlib
from argparse import ArgumentParser

parser = ArgumentParser("Anonymize Zeiss Forum HFA visual field data")
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
args = parser.parse_args()

with open(args.input, "rb") as inp:
    dcm_parser = pyvf.parse.dcm.HFADCMParser(inp)
dcm_parser = dcm_parser.anonymize(anonymization_fun=lambda x:
                b"" if isinstance(x, bytes)
                else "Anonymous^Anonymous" if isinstance(x, pydicom.valuerep.PersonName)
                else hashlib.sha1(str(x).zfill(16).encode("UTF-8")).hexdigest())
dcm_parser.save_as(args.output)
print(dcm_parser.name)
print(dcm_parser.id)
print(dcm_parser.dob)
print(dcm_parser.pdf_parser.vf)
print(dcm_parser.pdf_parser.td)
print(dcm_parser.pdf_parser.pd)
print(dcm_parser.md)
print(dcm_parser.mdsig)
print(dcm_parser.psd)
print(dcm_parser.psdsig)
print(dcm_parser.vfi)
print(dcm_parser.ght)
