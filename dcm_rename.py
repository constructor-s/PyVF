"""
Rename DCM files to their id, date, and eye
Intended for scientific research use only.

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

import pathlib
import pyvf.parse.dcm
from argparse import ArgumentParser

parser = ArgumentParser(description="Anonymize Zeiss Forum HFA visual field data")
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-n", "--dry-run", action="store_true")
args = parser.parse_args()

with open(args.input, "rb") as inp:
    dcm_parser = pyvf.parse.dcm.HFADCMParser(inp)
old_file = pathlib.Path(args.input)
new_name = (dcm_parser.id[:16] +
            dcm_parser.datetime.strftime("_%Y%m%d-%H%M%S_") +
            dcm_parser.laterality +
            old_file.suffix)
new_path = old_file.parent / new_name
if args.dry_run:
    print(args.input, "->", new_path)
else:
    old_file.rename(new_path)
