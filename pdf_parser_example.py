"""
Example for parsing HFA PDF SFA

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

from pyvf.parse.pdf import HFAPDFParser
import argparse
from itertools import chain
from glob import glob
import logging
_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Demo of HFA PDF SFA parsing")
    parser.add_argument("-i", "--input", required=True, nargs="+", help="input.pdf")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig()
    _logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger('pyvf.parse.pdf').setLevel(logging.DEBUG if args.verbose else logging.INFO)

    for inp in chain(*map(glob, args.input)):
        _logger.info("Input: %s", inp)
        with open(inp, "rb") as f:
            parser = HFAPDFParser(f)
        try:
            print(parser.id)
        except ValueError as e:
            _logger.error("Cannot parse ID: %s", e)
        print('parser.id =', parser.id)
        print('parser.dob =', parser.dob)
        print('parser.laterality =', parser.laterality)
        print('parser.report_type =', parser.report_type)
        print('parser.pattern =', parser.pattern)
        print('parser.vf =', parser.vf)
        print('parser.td =', parser.td)
        print('parser.pd =', parser.pd)
        print('parser.false_positive =', parser.false_positive)
        print('parser.false_negative =', parser.false_negative)
        print('f"{parser.fixation_loss_error}/{parser.fixation_loss_total}" =', f"{parser.fixation_loss_error}/{parser.fixation_loss_total}")
        print('parser.test_duration =', parser.test_duration)
        print('parser.date =', parser.date)
        print('parser.time =', parser.time)
        print('parser.age =', parser.age)
        print('parser.md =', parser.md)
        print('parser.psd =', parser.psd)
        print('parser.vfi =', parser.vfi)
        print('parser.ght =', parser.ght)
        if args.verbose:
            print(*list(enumerate(map(str, parser._device.render_items))), sep="\n")
