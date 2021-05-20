"""
Anonymization script for PDF SFA files exported from Zeiss Forum.
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

from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from datetime import datetime, timedelta
from io import BytesIO
from dataclasses import dataclass
from typing import List
import numpy as np
import hashlib
import re
from copy import copy
from pdfminer.pdfinterp import PDFTextState, PDFGraphicState
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdftypes import PDFStream
import logging
_logger = logging.getLogger(__name__)


class HFAPDFParser:
    REGEX_PATTERN_242 = "\n".join((
        "30" + chr(176),
        "30" + chr(176),
        "30" + chr(176),
        r"(?P<vf>(<?\d+\n){54})(?P<vfimg>(.+\n){412})" +
        r"(?P<td>(-?\d+\n){52})(?P<pd>((-?\d+\n){52})|(MD Threshold exceeded.\nSee Total Deviation plot.\n))" +
        r"(?P<tdp>([10]\.\d+\n){52})(?P<pdp>(([10]\.\d+\n){52})|(MD Threshold exceeded.\nSee Total Deviation plot.\n))"
        "Total Deviation",
        "Pattern Deviation"
    ))  # Currently only used to parse the vf, td, pd, td probability, pd probability regions
    REGEX_COMPILED_242 = re.compile(REGEX_PATTERN_242)

    def __init__(self, fp):
        fp.seek(0)
        self.raw_pdf = fp.read()  # Save a copy in memory for anonymization

        parser = PDFParser(BytesIO(self.raw_pdf))
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = HFASFADevice(rsrcmgr)  # TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

        self.byte_sequences = device.byte_sequences
        self.text_sequences = device.text_sequences
        self._device = device
        self.regex_match = None
        if "24-2" in self.pattern:
            self.regex_match = HFAPDFParser.REGEX_COMPILED_242.search("\n".join(map(str, self._device.render_items)))

    def anonymize(self, anonymization_fun=lambda x: b""):
        import subprocess
        uncompress_process = subprocess.run(["pdftk", "-", "output", "-", "uncompress"],
                                            input=self.raw_pdf, capture_output=True)
        uncompressed_pdf = uncompress_process.stdout

        # Sometimes there are extra b"\\" (i.e. chr(92)) in uncompressed_pdf but not in snippet
        # For example, the original PDF is b'\x00*\x005\x00$\x00\\)\x00...'
        # but the snippet that we have is b"\x00*\x005\x00$\x00)\x00)..."
        # I am not sure why this is happening, as PDF is very complicated
        # Below is a not so efficient hack, but hopefully works
        # Find all PDF bytes that represent literal strings
        import re
        snippet_dict = {m.group(0).replace(b"\\", b""): m.group(0) for m in re.finditer(rb"\(.*\)Tj", uncompressed_pdf)}

        for snippet in (self.byte_sequences[self.text_sequences.index("Patient:") + 1],
                        self.byte_sequences[self.text_sequences.index("Patient ID:") + 1],
                        self.byte_sequences[self.text_sequences.index("Date of Birth:") + 1],
                        ):
            value = anonymization_fun(snippet)
            if isinstance(value, str):
                value = value.encode("UTF-8")

            if b"("+snippet+b")Tj" in snippet_dict:
                uncompressed_pdf = uncompressed_pdf.replace(snippet_dict[b"("+snippet+b")Tj"], b"("+value+b")Tj")
            elif b"("+snippet+b")Tj" in uncompressed_pdf:
                uncompressed_pdf = uncompressed_pdf.replace(b"("+snippet+b")Tj", b"("+value+b")Tj")
            else:
                raise ValueError("PDF anonymization failed: Cannot locate byte sequence to remove in PDF: " + repr(snippet))
            # Since this PDF is passing through pdftk compression again,
            # we actually don't have to maintain the same byte length

        anonymized_process = subprocess.run(["pdftk", "-", "output", "-", "compress"],
                                            input=uncompressed_pdf, capture_output=True)
        return HFAPDFParser(BytesIO(anonymized_process.stdout))

    def get_value(self, key, offset=1):
        if key not in self.text_sequences:
            _logger.debug("Cannot find %s in text_sequences. (PDF may have been modified/anonymized)", repr(key))
            return ""

        # key exists in self.text_sequences
        key_index = self.text_sequences.index(key)
        # value = self.text_sequences[key_index + offset]
        real_offset = offset
        i = 0
        while i < real_offset:
            i += 1
            if self.text_sequences[key_index + i] == ' ':
                real_offset += 1
        return self.text_sequences[key_index + real_offset]

    def get_value_list(self, key, offset_start=1, length=1):
        key_index = self.text_sequences.index(key)
        value = self.text_sequences[key_index + offset_start:key_index + offset_start + length]
        return value

    def get_value_try_multiple_methods(self, kwargs_list, validate_fun):
        """
        For slightly different PDF formats, some key locators may not work.
        This is a helper method wrapping the get_value method
        to conveniently retry different ways if invalid values were obtained
        """
        assert len(kwargs_list) > 0, "Must provide at least one kwargs in kwargs_list"
        success = False

        for kwargs in kwargs_list:
            try:
                value = self.get_value(**kwargs)
            except ValueError as e:
                _logger.debug("Get value failed with %s, continue: %s", kwargs, e)
                continue

            success = validate_fun(value)
            if success:  # Returns True if it is a valid value
                break
            else:
                continue

        assert success, f"Failed on all get value options: {str(kwargs_list)}. Last value was {repr(value)}."
        # noinspection PyUnboundLocalVariable
        return value


    @property
    def name(self):
        return self.get_value("Patient:")

    @property
    def dob(self):
        return self.get_value("Date of Birth:")

    @property
    def gender(self):
        return self.get_value("Gender:")

    @property
    def id(self):
        return self.get_value("Patient ID:")

    @property
    def laterality(self):
        value = self.get_value_try_multiple_methods((
            {"key": "Patient ID:", "offset": 2},
            {"key": "Version", "offset": -4},
        ), validate_fun=lambda value: value == "OS" or value == "OD")
        return value

    @property
    def report_type(self):
        # assert value == "Single Field Analysis", f"Report type {value} currently not supported"
        value = self.get_value_try_multiple_methods((
            {"key": "Patient ID:", "offset": 3},
            {"key": "Version", "offset": -3},
        ), validate_fun=lambda value: value == "Single Field Analysis")
        return value

    @property
    def pattern(self):
        value = self.get_value_try_multiple_methods((
            {"key": "Patient ID:", "offset": 4},
            {"key": "Version", "offset": -2},
        ), validate_fun=lambda value: "24-2" in value or "10-2" in value or "30-2" in value)
        return value

    @property
    def fixation_monitor(self):
        return self.get_value("Fixation Monitor:", offset=7)

    @property
    def fixation_target(self):
        return self.get_value("Fixation Target:", offset=7)

    @property
    def fixation_loss_error(self):
        return int(self.fixation_loss.split()[0].split("/")[0])

    @property
    def fixation_loss_total(self):
        return int(self.fixation_loss.split()[0].split("/")[1])

    @property
    def fixation_loss(self):
        return self.get_value("Fixation Losses:", offset=7)

    @property
    def false_positive(self):
        value = self.get_value("False POS Errors:", offset=7)
        if value == 'N/A':
            return "nan"
        return value

    @property
    def false_negative(self):
        value = self.get_value("False NEG Errors:", offset=7)
        if value == 'N/A':
            return "nan"
        return value

    @property
    def test_duration(self):
        value = self.get_value("Test Duration:", offset=7)
        t = datetime.strptime(value, "%M:%S")
        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        return delta.total_seconds()

    @property
    def fovea(self):
        return self.get_value("Fovea:", offset=7)

    @property
    def stimulus(self):
        return self.get_value("Stimulus:", offset=6)

    @property
    def background(self):
        return self.get_value("Background:", offset=6)

    @property
    def strategy(self):
        return self.get_value("Strategy:", offset=6)

    @property
    def pupil_diameter(self):
        return self.get_value("Pupil Diameter:", offset=6)

    @property
    def visual_acuity(self):
        if self.get_value("Visual Acuity:", offset=7) == "Date:":
            return None  # Visual acuity was skipped
        else:
            return self.get_value("Visual Acuity:", offset=6)

    @property
    def rx(self):
        return self.get_value("Date:", offset=-1)

    @property
    def date(self):
        value = self.get_value("Date:", offset=3)
        dt = datetime.strptime(value, "%b %d, %Y")
        return dt.date()

    @property
    def time(self):
        value = self.get_value("Time:", offset=3)
        dt = datetime.strptime(value, "%I:%M %p")
        return dt.time()

    @property
    def age(self):
        value = self.get_value("Age:", offset=3)
        return float(value)

    @property
    def n_vf_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 54
        else:
            raise NotImplementedError(f"n_vf_loc is not yet implemented for {self.pattern}")

    @property
    def n_td_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 52
        else:
            raise NotImplementedError(f"n_td_loc is not yet implemented for {self.pattern}")

    @property
    def vf(self):
        """

        Returns
        -------
        A list of visual field thresholds as float. "<0" are converted to -1.0
        """
        if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
            value_list = self.get_value_list("Total Deviation", offset_start=-4-self.n_td_loc-self.n_vf_loc, length=self.n_vf_loc)
        else:
            value_list = self.get_value_list("Total Deviation", offset_start=-self.n_td_loc*2-self.n_vf_loc, length=self.n_vf_loc)
        values = [float(i) if i != "<0" else -1.0 for i in value_list]
        assert all(map(lambda x: x>=-1, values))
        return values

    @property
    def td(self):
        if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
            value_list = self.get_value_list("Total Deviation", offset_start=-4-self.n_td_loc, length=self.n_td_loc)
        else:
            value_list = self.get_value_list("Total Deviation", offset_start=-self.n_td_loc*2, length=self.n_td_loc)
        return [float(i) for i in value_list]

    @property
    def pd(self):
        if self.get_value("Total Deviation", offset=-1) == "See Total Deviation plot.":
            return [float("nan")] * self.n_td_loc
        else:
            value_list = self.get_value_list("Pattern Deviation", offset_start=-1-self.n_td_loc, length=self.n_td_loc)
            return [float(i) for i in value_list]

    @property
    def tdp(self):
        """

        Returns
        -------
        List[float]
            Probability threshold in the total deviation map
        """
        if self.regex_match is not None:
            return list(map(float, self.regex_match.group('tdp').strip().split("\n")))
        else:
            _logger.warning("Parsing of TD probability map currently not supported for this file.")
            return [float("nan") for _ in range(self.n_td_loc)]

    @property
    def pdp(self):
        """

        Returns
        -------
        List[float]
            Probability threshold in the pattern deviation map
        """
        if self.regex_match is not None:
            match_lines = self.regex_match.group('pdp').strip().split("\n")
            if match_lines[0].strip() == "MD Threshold exceeded.":
                return [float("nan") for _ in range(self.n_td_loc)]
            return list(map(float, self.regex_match.group('pdp').strip().split("\n")))
        else:
            _logger.warning("Parsing of PD probability map currently not supported for this file.")
            return [float("nan") for _ in range(self.n_td_loc)]

    @property
    def ght(self):
        # Sometimes GHT is split across two lines
        pt1 = self.get_value("GHT:", offset=1)
        pt2 = self.get_value("GHT:", offset=2)
        if pt2 == "VFI:":
            return pt1
        else:
            return " ".join((pt1, pt2))

    @property
    def vfi(self):
        return self.get_value("VFI:", offset=1)

    @property
    def md(self):
        for key in ("MD:", "MD24-2:", "MD10-2:", "MD30-2:"):
            value = self.get_value(key, offset=1)
            if value:
                # If we can find a non-empty string
                return value
            else:
                # If not, keep searching
                _logger.debug("Cannot find %s, trying other keys.", key)
                continue

    @property
    def psd(self):
        for key in ("PSD:", "PSD24-2:", "PSD10-2:", "PSD30-2:"):
            value = self.get_value(key, offset=1)
            if value:
                # If we can find a non-empty string
                return value
            else:
                # If not, keep searching
                _logger.debug("Cannot find %s, trying other keys.", key)
                continue


class HFASFADevice(PDFLayoutAnalyzer):
    def __init__(self, rsrcmgr, pageno=1, laparams=None):
        super(HFASFADevice, self).__init__(rsrcmgr, pageno=pageno, laparams=laparams)
        self.byte_sequences = []
        self.text_sequences = []
        self.render_items = []

    def render_string(self, textstate, seq, ncs, graphicstate):
        super(HFASFADevice, self).render_string(textstate, seq, ncs, graphicstate)
        self.render_items.append(StringRenderItem(copy(textstate), seq, ncs, graphicstate))  # Must use copy, otherwise font objet is different later on
        font = textstate.font
        for obj in seq:
            if not isinstance(obj, bytes):
                # For PDF anonymized by VEM's Java software
                # some obj may no long be bytes but become invalid ints...
                # skip them
                _logger.debug("obj = %s is not of bytes type, skipping and appending an empty line", repr(obj))
                obj = b""
            self.byte_sequences.append(obj)
            self.text_sequences.append("".join([font.to_unichr(c) for c in font.decode(obj)]))

    def render_image(self, name, stream):
        super(HFASFADevice, self).render_image(name, stream)
        self.render_items.append(ImageRenderItem(name, stream))


@dataclass(frozen=True)
class StringRenderItem:
    textstate: PDFTextState
    seq: List[bytes]
    ncs: PDFColorSpace
    graphicstate: PDFGraphicState

    @property
    def decoded_seq(self):
        """
        Decode the bytes in seq using the font object.
        Sequences that cannot be parsed as replaced with empty b''.

        Returns
        -------
        List[str]
            List of decoded string
        """
        ret = []
        font = self.textstate.font
        for obj in self.seq:
            if not isinstance(obj, bytes):
                _logger.debug("obj = %s is not of bytes type, skipping and appending an empty line", repr(obj))
                obj = b""
            text = "".join([font.to_unichr(c) for c in font.decode(obj)])
            ret.append(text)
        return tuple(ret)

    def __str__(self):
        return "\n".join(self.decoded_seq)


@dataclass(frozen=True, repr=False)
class ImageRenderItem:
    name: str
    stream: PDFStream

    # Class attribute storing sha1 hash of image bytes to semantic meaning - pre-generated
    hash2str = {
        "2b3f91b0f6b384ae20fbfd6f056adae48c870ab3": 1.0,  # Decompressed
        "35be72552461ac1bbbf5cd5c6bfaaa4520af6da8": 0.05,  # Decompressed
        "1bfae8881f1ffbdf84ae1eecb1ddcc54e7fa1937": 0.02,  # Decompressed
        "534d50117dd5fb4ae70b05aeec3ec022b703ad3b": 0.01,  # Decompressed
        "6d396f13345b0b4d808c502db975e9d6b2987e88": 0.005,  # Decompressed
    }

    @property
    def decoded_value(self):
        """
        Get the numerical value that represents the semantic meaning of this image,
        if it can be interpreted

        Returns
        -------
        float
            Decoded value of the image
        """
        return ImageRenderItem.hash2str.get(self._get_decoded_image_hash(), float("nan"))

    @property
    def decoded_image(self):
        """
        Get the image representation of this object

        Returns
        -------
        np.ndarray
            Image generated from the stream bytes as a numpy array
        """
        if self.stream.data is None:
            self.stream.decode()
        buffer = self.stream.data
        if len(buffer) == self.stream.get("Length") + 1 and buffer[-1:] == b"\n":
            buffer = buffer[:-1]  # Usually there is an extra byte of b"\n" at the end for uncompressed stream...
        # assert len(buffer) == self.stream.get("Length"), f"Mismatch length of buffer ({len(buffer)}) and length in attribute ({self.stream.get('Length')})"
        w = self.stream.get("Width")
        h = self.stream.get("Height")
        c = self.stream.get("DecodeParms", {}).get("Colors", 1)
        if c == 1:
            # Sometimes in modified files, the image is uncompressed and does not specify how many colors/channels
            # Maybe could use self.stream.get("ColorSpace") == DeviceRGB (Need to find reference of DeviceRGB literal)
            c = len(buffer) // w // h
        bits_per_component = self.stream.get("BitsPerComponent")
        if bits_per_component == 8:
            dtype = np.uint8
        else:
            raise NotImplementedError(f"bits_per_component = {bits_per_component} is not yet implemented")

        # for filter_name, params in self.stream.get_filters():  # See pdfminer\image.py
        #     if filter_name == pdfminer.psparser.LIT("FlateDecode"):
        #         buffer = zlib.decompress(buffer)
        #     else:
        #         raise NotImplementedError(f"filter_name = {filter_name} is unsupported.")
        # # Lacks apply_png_predictor

        # if len(buffer) != w * h * bits_per_component / 8:
        #     _logger.error(f"Mismatch image shape ({w}, {h}) and buffer length ({len(buffer)}). Decoding failed.")
        #     return np.zeros((h, w), dtype=dtype)
        im = np.frombuffer(buffer, dtype=dtype).reshape(h, w, c)
        return im

    def _get_decoded_image_hash(self):
        return hashlib.sha1(self.decoded_image[..., 0].tobytes()).hexdigest()

    def __str__(self):
        return str(self.decoded_value)

    def __repr__(self):
        sup = super(ImageRenderItem, self).__repr__()
        return fr"{self.decoded_value}({self.name}, SHA1:{self._get_decoded_image_hash()})"+sup
