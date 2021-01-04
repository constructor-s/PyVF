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
from pathlib import Path


def anonymize_dcm(input_file, output_file):
    if isinstance(input_file, pyvf.parse.dcm.HFADCMParser):
        dcm_parser = input_file
    else:
        with open(input_file, "rb") as inp:
            dcm_parser = pyvf.parse.dcm.HFADCMParser(inp)
    dcm_parser = dcm_parser.anonymize(anonymization_fun=lambda x:
                    b"" if isinstance(x, bytes)
                    else "Anonymous^Anonymous" if isinstance(x, pydicom.valuerep.PersonName)
                    else hashlib.sha1(str(x).zfill(16).encode("UTF-8")).hexdigest())
    dcm_parser.save_as(output_file)
    # print(dcm_parser.name)
    # print(dcm_parser.id)
    # print(dcm_parser.dob)
    # print(dcm_parser.pdf_parser.vf)
    # print(dcm_parser.pdf_parser.td)
    # print(dcm_parser.pdf_parser.pd)
    # print(dcm_parser.md)
    # print(dcm_parser.mdsig)
    # print(dcm_parser.psd)
    # print(dcm_parser.psdsig)
    # print(dcm_parser.vfi)
    # print(dcm_parser.ght)
    return dcm_parser


if __name__ == '__main__':
    parser = ArgumentParser("Anonymize Zeiss Forum HFA visual field data")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    if Path(args.input).is_file():
        anonymize_dcm(args.input, args.output)
    elif Path(args.input).is_dir():
        for dcm_file in Path(args.input).rglob("*.dcm"):
            print("Input:", dcm_file)
            with open(dcm_file, "rb") as inp:
                dcm_parser = pyvf.parse.dcm.HFADCMParser(inp)
            new_name = "_".join([
                hashlib.sha1(str(dcm_parser.name).zfill(16).encode("UTF-8")).hexdigest(),
                dcm_parser.laterality,
                dcm_parser.pattern,
                dcm_parser.strategy,
                dcm_parser.datetime.strftime("%Y%m%d%H%M%S")
            ])
            new_name = new_name + ".dcm"
            output_file = Path(args.output) / dcm_file.relative_to(args.input)
            output_file = output_file.parent / new_name
            print("Output:", output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            anonymize_dcm(dcm_file, output_file)
    else:
        raise ValueError(f"{args.input} is invalid")
