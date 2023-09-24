"""
Example for parsing HFA DCM SFA

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

from pyvf.parse.dcm import HFADCMParser
import argparse
from itertools import chain
from glob import glob
import logging
_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of HFA DCM SFA parsing")
    parser.add_argument("-i", "--input", required=True, nargs="+", help="input.dcm")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig()
    _logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger('pyvf.parse.dcm').setLevel(logging.DEBUG if args.verbose else logging.INFO)

    for inp in chain(*map(glob, args.input)):
        _logger.info("Input: %s", inp)
        with open(inp, "rb") as f:
            parser = HFADCMParser(f)
        print(parser.id)
        print(parser.laterality)
        print(parser.report_type)
        print(parser.pattern)
        print(len(parser.pdf_parser.vf), parser.pdf_parser.vf)
        print(len(parser.pdf_parser.td), parser.pdf_parser.td)
        print(len(parser.pdf_parser.pd), parser.pdf_parser.pd)
        print(parser.false_positive)
        print(parser.false_negative)
        print(f"{parser.fixation_loss_error}/{parser.fixation_loss_total}")
        print(parser.pdf_parser.test_duration)
        print(parser.datetime)
        # print(parser.time)
        print(parser.dob)
        print(parser.md)
        print(parser.psd)
        print(parser.vfi)
        print(parser.ght)
