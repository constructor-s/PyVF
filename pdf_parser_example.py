from pyvf.parse.pdf import HFAPDFParser
import argparse
import logging
_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Demo of HFA PDF SFA parsing")
    parser.add_argument("-i", "--input", required=True, help="input.pdf")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig()
    _logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger('pyvf.parse.pdf').setLevel(logging.DEBUG if args.verbose else logging.INFO)

    _logger.debug("Input: %s", args.input)
    with open(args.input, "rb") as f:
        parser = HFAPDFParser(f)
    print(parser.laterality)
    print(parser.report_type)
    print(parser.pattern)
    print(parser.vf)
    print(parser.td)
    print(parser.pd)
    print(parser.false_positive)
    print(parser.false_negative)
    print(f"{parser.fixation_loss_error}/{parser.fixation_loss_total}")
    print(parser.test_duration)
    print(parser.date)
    print(parser.time)
    print(parser.age)
    print(parser.md)
    print(parser.psd)
    print(parser.vfi)
    print(parser.ght)
