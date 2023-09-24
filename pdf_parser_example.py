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
    parser = argparse.ArgumentParser(description="Demo of HFA PDF SFA parsing")
    parser.add_argument("-i", "--input", required=True, nargs="+", help="input.pdf")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--silent", action="store_true", help="Suppress print (useful for debugging)")
    parser.add_argument("-d", "--dump", required=False, help="Dump PDF parser internal representation")
    args = parser.parse_args()

    logging.basicConfig()
    _logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger('pyvf.parse.pdf').setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.silent:
        fun = lambda *_, **__: None
    else:
        fun = print

    for inp in chain(*map(glob, args.input)):
        _logger.info("Input: %s", inp)
        with open(inp, "rb") as f:
            parser = HFAPDFParser(f)
        try:
            fun(parser.id)
            fun('parser.id =', parser.id)
            fun('parser.dob =', parser.dob)
            fun('parser.laterality =', parser.laterality)
            fun('parser.report_type =', parser.report_type)
            fun('parser.pattern =', parser.pattern)
            fun('parser.vf =', parser.vf)
            fun('parser.td =', parser.td)
            fun('parser.pd =', parser.pd)
            fun('parser.tdp =', parser.tdp)
            fun('parser.pdp =', parser.pdp)
            fun('parser.false_positive =', parser.false_positive)
            fun('parser.false_negative =', parser.false_negative)
            fun('f"{parser.fixation_loss_error}/{parser.fixation_loss_total}" =', f"{parser.fixation_loss_error}/{parser.fixation_loss_total}")
            fun('parser.test_duration =', parser.test_duration)
            fun('parser.date =', parser.date)
            fun('parser.time =', parser.time)
            fun('parser.age =', parser.age)
            fun('parser.md =', parser.md)
            fun('parser.psd =', parser.psd)
            fun('parser.vfi =', parser.vfi)
            fun('parser.ght =', parser.ght)
        except Exception as e:
            _logger.exception("Error in parsing %s:", inp)

        if args.dump:
            with open(args.dump, "w", encoding="utf-8") as f:
                f.writelines(map(lambda x: repr(x) + "\n", parser._device.render_items))
