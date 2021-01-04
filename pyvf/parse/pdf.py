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


class HFAPDFParser:
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
        value = self.get_value("Patient ID:", offset=2)
        assert value == "OS" or value == "OD"
        return value

    @property
    def report_type(self):
        value = self.get_value("Patient ID:", offset=3)
        assert value == "Single Field Analysis", f"Report type {value} currently not supported"
        return value

    @property
    def pattern(self):
        return self.get_value("Patient ID:", offset=4)

    @property
    def fixation_monitor(self):
        return self.get_value("Fixation Monitor:", offset=7)

    @property
    def fixation_target(self):
        return self.get_value("Fixation Target:", offset=7)

    @property
    def fixation_loss_error(self):
        return int(self.fixation_losses.split()[0].split("/")[0])

    @property
    def fixation_loss_total(self):
        return int(self.fixation_losses.split()[0].split("/")[1])

    @property
    def fixation_loss(self):
        return self.get_value("Fixation Losses:", offset=7)

    @property
    def false_positive(self):
        return self.get_value("False POS Errors:", offset=7)

    @property
    def false_negative(self):
        return self.get_value("False NEG Errors:", offset=7)

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
            raise NotImplementedError()

    @property
    def n_td_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 52
        else:
            raise NotImplementedError()

    @property
    def vf(self):
        """

        Returns
        -------
        A list of visual field thresholds as float. "<0" are converted to -1.0
        """
        value_list = self.get_value_list("Total Deviation", offset_start=-self.n_td_loc-self.n_td_loc-self.n_vf_loc, length=self.n_vf_loc)
        values = [float(i) if i != "<0" else -1 for i in value_list]
        assert all(map(lambda x: x>=-1, values))
        return values

    @property
    def td(self):
        value_list = self.get_value_list("Total Deviation", offset_start=-self.n_td_loc-self.n_td_loc, length=self.n_td_loc)
        return [float(i) for i in value_list]

    @property
    def pd(self):
        value_list = self.get_value_list("Pattern Deviation", offset_start=-1-self.n_td_loc, length=self.n_td_loc)
        return [float(i) for i in value_list]

    @property
    def ght(self):
        return self.get_value("GHT:", offset=1)

    @property
    def vfi(self):
        return self.get_value("VFI:", offset=1)

    @property
    def md(self):
        return self.get_value("MD:", offset=1)

    @property
    def psd(self):
        return self.get_value("PSD:", offset=1)


class HFASFADevice(PDFLayoutAnalyzer):
    def __init__(self, rsrcmgr, pageno=1, laparams=None):
        super(HFASFADevice, self).__init__(rsrcmgr, pageno=1, laparams=None)
        self.byte_sequences = []
        self.text_sequences = []

    def render_string(self, textstate, seq, ncs, graphicstate):
        super(HFASFADevice, self).render_string(textstate, seq, ncs, graphicstate)
        font = textstate.font
        for obj in seq:
            self.byte_sequences.append(obj)
            self.text_sequences.append("".join([font.to_unichr(c) for c in font.decode(obj)]))
