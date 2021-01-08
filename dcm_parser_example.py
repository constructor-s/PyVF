from pyvf.parse.dcm import HFADCMParser
import argparse
from itertools import chain
from glob import glob
import logging
_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Demo of HFA DCM SFA parsing")
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
        print(parser.pdf_parser.vf)
        print(parser.pdf_parser.td)
        print(parser.pdf_parser.pd)
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
