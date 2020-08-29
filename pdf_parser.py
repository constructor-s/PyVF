import sys
from io import StringIO

from pdfminer.converter import TextConverter, PDFConverter, PDFLayoutAnalyzer
from pdfminer.layout import LAParams, LTContainer, LTText, LTTextBox, LTImage
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


class HFASFADevice(PDFLayoutAnalyzer):
    def __init__(self, rsrcmgr, pageno=1, laparams=None):
        super(HFASFADevice, self).__init__(rsrcmgr, pageno, laparams)
        self.byte_sequences = []
        self.text_sequences = []

    def render_string(self, textstate, seq, ncs, graphicstate):
        super(HFASFADevice, self).render_string(textstate, seq, ncs, graphicstate)
        font = textstate.font
        for obj in seq:
            self.byte_sequences.append(obj)
            self.text_sequences.append("".join([font.to_unichr(c) for c in font.decode(obj)]))


class HFASFATextParser:
    def __init__(self, text_sequences):
        self.text_sequences = text_sequences

    def get_value(self, key, offset_start=1):
        key_index = self.text_sequences.index(key)
        value = self.text_sequences[key_index + offset_start]
        return value

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
        value = self.get_value("Patient ID:", offset_start=2)
        assert value == "OS" or value == "OD"
        return value

    @property
    def report_type(self):
        value = self.get_value("Patient ID:", offset_start=3)
        assert value == "Single Field Analysis"
        return value

    @property
    def pattern(self):
        value = self.get_value("Patient ID:", offset_start=4)
        return value

    @property
    def n_vf_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 54

    @property
    def n_td_loc(self):
        if self.pattern == "Central 24-2 Threshold Test":
            return 52

    @property
    def vf(self):
        value_list = self.get_value_list("Age:", offset_start=7, length=self.n_vf_loc)
        for i in value_list:
            assert i == "<0" or float(i) is not None
        return value_list

    @property
    def td(self):
        value_list = self.get_value_list("Age:", offset_start=7+self.n_vf_loc, length=self.n_td_loc)
        for i in value_list:
            assert float(i) is not None
        return value_list

    @property
    def pd(self):
        value_list = self.get_value_list("Age:", offset_start=7 + self.n_vf_loc+self.n_td_loc, length=self.n_td_loc)
        for i in value_list:
            assert float(i) is not None
        return value_list


if __name__ == '__main__':
    import subprocess
    import io

    try:
        output = subprocess.run(["pdftk", "--version"], capture_output=True)
        print("Found pdftk")
        print(output.stdout.decode())
    except FileNotFoundError:
        print("Did not find pdftk, downloading...")
        import requests
        import zipfile
        import shutil
        r = requests.get("https://globalcdn.nuget.org/packages/pdftk.exe.1.44.0.nupkg")
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open("Tools/pdftk/bin/pdftk.exe") as fsrc, open("pdftk.exe", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            with zf.open("Tools/pdftk/bin/libiconv2.dll") as fsrc, open("libiconv2.dll", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
        output = subprocess.run(["pdftk", "--version"], capture_output=True)
        print("Found pdftk")
        print(output.stdout.decode())

    output = subprocess.run(["pdftk", sys.argv[1], "output", "-", "uncompress"], capture_output=True)
    raw_pdf = output.stdout

    # with open(sys.argv[1], 'rb') as in_file:
    with io.BytesIO(raw_pdf) as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = HFASFADevice(rsrcmgr)  # TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

        sfa = HFASFATextParser(device.text_sequences)
        print(sfa.name, sfa.dob, sfa.gender, sfa.id, sfa.laterality, sep="\n")
        print(sfa.vf)
        print(sfa.td)
        print(sfa.pd)

    # Anonymization
    for snippet in (device.byte_sequences[device.text_sequences.index("Patient:")+1],
                    device.byte_sequences[device.text_sequences.index("Patient ID:")+1],
                    device.byte_sequences[device.text_sequences.index("Date of Birth:")+1],
                    ):
        raw_pdf = raw_pdf.replace(snippet, b"")  # Since this PDF is passing through pdftk compression again, we actually don't have to manitain the same byte length

    output = subprocess.run(["pdftk", "-", "output", sys.argv[1]+".pdf", "compress"], input=raw_pdf)
