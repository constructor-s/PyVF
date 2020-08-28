import sys
from io import StringIO

from pdfminer.converter import TextConverter, PDFConverter, PDFLayoutAnalyzer
from pdfminer.layout import LAParams, LTContainer, LTText, LTTextBox, LTImage
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


class TextCollectorDevice(PDFLayoutAnalyzer):
    def receive_layout(self, ltpage):
        self.ltpage = ltpage
        self.items = []
        self.lines = []
        self.curr_x = None
        self.curr_y = None
        self.render(ltpage)

    def render(self, item):
        if isinstance(item, LTContainer):
            for child in item:
                self.render(child)
        elif isinstance(item, LTText):
            self.items.append(item)
            a, b, c, d, x, y = item.matrix
            if x == self.curr_x or y == self.curr_y:
                self.lines[-1].append(item.get_text())
            else:
                self.lines.append([item.get_text()])
            self.curr_x = x
            self.curr_y = y

        if isinstance(item, LTTextBox):
            return
        elif isinstance(item, LTImage):
            return
        else:
            return


with open(sys.argv[1], 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextCollectorDevice(rsrcmgr)  # TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)

    print('\n'.join(''.join(i) for i in device.lines))
