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
from .pdf import HFAPDFParser
import pydicom
from pydicom.tag import Tag
from io import BytesIO
import datetime
import logging
import numpy as np

SFA_DCM_ENCAPSULATED_TYPE = Tag(0x2201, 0x1000)
SFA_DCM_ENCAPSULATED_VERSION = Tag(0x2201, 0x1001)
SFA_DCM_ENCAPSULATED_INFO = Tag(0x2201, 0x1002)
SFA_DCM_REPORT_TYPE = Tag(0x22A1, 0x1001)
SFA_DCM_REPORT_TYPE = Tag(0x2501, 0x1000)  # Both of these fields seem to be the same...
SFA_DCM_UNKNOWN = Tag(0x2501, 0x1007)
SFA_DCM_UNKNOWN = Tag(0x2501, 0x1008)
SFA_DCM_PATTERN = Tag(0x7717, 0x1001)
SFA_DCM_STRATEGY = Tag(0x7717, 0x1002)
SFA_DCM_STIMULUS_SIZE = Tag(0x7717, 0x1003)
SFA_DCM_STIMULUS_COLOR = Tag(0x7717, 0x1004)
SFA_DCM_STIMULUS_BACKGROUND = Tag(0x7717, 0x1005)  # TODO: This may be swapped with above
SFA_DCM_UNKNOWN = Tag(0x7717, 0x1006)  # Value is "NOT_TESTED" - maybe this is fovea?
SFA_DCM_FL_TOTAL = Tag(0x7717, 0x1008)
SFA_DCM_FL_ERROR = Tag(0x7717, 0x1009)
SFA_DCM_FP = Tag(0x7717, 0x1010)
SFA_DCM_FN = Tag(0x7717, 0x1013)
SFA_DCM_MD = Tag(0x7717, 0x1016)
SFA_DCM_MDSIG = Tag(0x7717, 0x1017)
SFA_DCM_PSD = Tag(0x7717, 0x1018)
SFA_DCM_PSDSIG = Tag(0x7717, 0x1019)
SFA_DCM_GHT = Tag(0x7717, 0x1023)
SFA_DCM_FIXATION_MONITOR = Tag(0x7717, 0x1024)
SFA_DCM_FIXATION_TARGET = Tag(0x7717, 0x1025)
SFA_DCM_PUPIL_DIAMETER = Tag(0x7717, 0x1026)
SFA_DCM_RX = Tag(0x7717, 0x1027)
SFA_DCM_DATE = Tag(0x7717, 0x1032)
SFA_DCM_TIME = Tag(0x7717, 0x1033)
SFA_DCM_VFI = Tag(0x7717, 0x1034)

_logger = logging.getLogger(__name__)
EMPTY_PDF = b'%PDF-1.1\n%\xe2\xe3\xcf\xd3\n1 0 obj \n<<\n/Kids [2 0 R]\n/Count 1\n/Type /Pages\n>>\nendobj \n2 0 obj \n<<\n/Parent 1 0 R\n/Resources 3 0 R\n/MediaBox [0 0 612 792]\n/Contents [4 0 R]\n/Type /Page\n>>\nendobj \n3 0 obj \n<<\n/Font \n<<\n/F0 \n<<\n/BaseFont /Courier\n/Subtype /Type1\n/Type /Font\n>>\n>>\n>>\nendobj \n4 0 obj \n<<\n/Length 80\n>>\nstream\n1. 0. 0. 1. 50. 700. cm\nBT\n  /F0 16. Tf\n  (PDF content has been removed) Tj\nET \n\nendstream \nendobj \n5 0 obj \n<<\n/Pages 1 0 R\n/Type /Catalog\n>>\nendobj xref\n0 6\n0000000000 65535 f \n0000000015 00000 n \n0000000074 00000 n \n0000000182 00000 n \n0000000276 00000 n \n0000000409 00000 n \ntrailer\n\n<<\n/Root 5 0 R\n/Size 6\n>>\nstartxref\n459\n%%EOF\n'


def property_with_default(default=None):
    class custom_property(property):
        def __init__(self, fget, *args, **kwargs):
            def new_fget(*args, **kwargs):
                try:
                    return fget(*args, **kwargs)
                except KeyError:
                    return default
            super().__init__(fget=new_fget, *args, **kwargs)
    return custom_property


class HFADCMParser:
    def __init__(self, fp=None, dataset=None):
        if fp is not None:
            self.dataset = pydicom.dcmread(fp)
        elif dataset is not None:
            self.dataset = dataset

        pdf_fp = BytesIO(self.dataset.EncapsulatedDocument)
        self.pdf_parser = HFAPDFParser(pdf_fp)

    def anonymize(self, anonymization_fun=lambda x: ""):
        from copy import deepcopy
        dataset = deepcopy(self.dataset)  # Make copy
        dcmanonymize(dataset, anonymization_fun=anonymization_fun)
        try:
            anonymized_pdf_parser = self.pdf_parser.anonymize(anonymization_fun=anonymization_fun)
            dataset.EncapsulatedDocument = anonymized_pdf_parser.raw_pdf
        except ValueError as e:
            _logger.error(f"Could not parse PDF format. Replacing with empty PDF. ({str(e)})")
            dataset.EncapsulatedDocument = EMPTY_PDF

        return HFADCMParser(dataset=dataset)

    def save_as(self, filename):
        self.dataset.save_as(filename)

    @property_with_default()
    def name(self):
        return self.dataset.PatientName

    @property_with_default()
    def dob(self):
        return datetime.datetime.strptime(self.dataset.PatientBirthDate, "%Y%m%d")

    @property_with_default()
    def gender(self):
        return self.dataset.PatientSex

    @property_with_default()
    def id(self):
        return self.dataset.PatientID

    @property_with_default()
    def laterality(self):
        value = self.dataset.Laterality
        value = {"R": "OD", "L": "OS"}[value]
        assert value == "OS" or value == "OD"
        return value

    @property_with_default()
    def report_type(self):
        return self.dataset[SFA_DCM_REPORT_TYPE].value

    @property_with_default()
    def pattern(self):
        return self.dataset[SFA_DCM_PATTERN].value

    @property_with_default()
    def strategy(self):
        return self.dataset[SFA_DCM_STRATEGY].value

    @property_with_default()
    def fixation_monitor(self):
        return self.dataset[SFA_DCM_FIXATION_MONITOR].value

    @property_with_default()
    def fixation_target(self):
        return self.dataset[SFA_DCM_FIXATION_TARGET].value

    @property_with_default(default=np.nan)
    def fixation_loss_error(self):
        return self.dataset[SFA_DCM_FL_ERROR].value

    @property_with_default(default=np.nan)
    def fixation_loss_total(self):
        return self.dataset[SFA_DCM_FL_TOTAL].value

    @property_with_default(default=np.nan)
    def fixation_loss(self):
        return self.fixation_loss_error * 1.0 / self.fixation_loss_total

    @property_with_default(default=np.nan)
    def false_positive(self):
        return self.dataset[SFA_DCM_FP].value / 100.0

    @property_with_default(default=np.nan)
    def false_negative(self):
        return self.dataset[SFA_DCM_FN].value / 100.0

    @property_with_default()
    def stimulus_size(self):
        return self.dataset[SFA_DCM_STIMULUS_SIZE].value

    @property_with_default()
    def stimulus_color(self):
        return self.dataset[SFA_DCM_STIMULUS_COLOR].value

    @property_with_default()
    def stimulus_background(self):
        return self.dataset[SFA_DCM_STIMULUS_BACKGROUND].value

    @property_with_default()
    def pupil_diameter(self):
        return self.dataset[SFA_DCM_PUPIL_DIAMETER].value

    @property_with_default()
    def rx(self):
        return self.dataset[SFA_DCM_RX].value

    @property_with_default()
    def datetime(self):
        value = self.dataset.AcquisitionDateTime
        return datetime.datetime.strptime(value, "%Y%m%d%H%M%S.%f")

    @property_with_default()
    def ght(self):
        return self.dataset[SFA_DCM_GHT].value

    @property_with_default(default=np.nan)
    def vfi(self):
        return self.dataset[SFA_DCM_VFI].value

    @property_with_default(default=np.nan)
    def md(self):
        return self.dataset[SFA_DCM_MD].value

    @property_with_default()
    def mdsig(self):
        return self.dataset[SFA_DCM_MDSIG].value

    @property_with_default(default=np.nan)
    def psd(self):
        return self.dataset[SFA_DCM_PSD].value

    @property_with_default()
    def psdsig(self):
        return self.dataset[SFA_DCM_PSDSIG].value


def dcmanonymize(dataset, anonymization_fun):
    """
    Warning: This is only designed for Zeiss Forum SFA saved as DCM format
    First part copied from by: https://pydicom.github.io/pydicom/stable/auto_examples/metadata_processing/plot_anonymize.html#sphx-glr-auto-examples-metadata-processing-plot-anonymize-py

    Since age is important for setting baseline hill of vision, only the date of birth is reset to 01
    """
    ###############################################################################
    # We can define a callback function to find all tags corresponding to a person
    # names inside the dataset. We can also define a callback function to remove
    # curves tags.

    def person_names_callback(dataset, data_element):
        if data_element.VR == "PN":
            data_element.value = "anonymous"

    def curves_callback(dataset, data_element):
        if data_element.tag.group & 0xFF00 == 0x5000:
            del dataset[data_element.tag]

    ###############################################################################
    # We can use the different callback function to iterate through the dataset but
    # also some other tags such that patient ID, etc.

    dataset.PatientID = "id"
    dataset.walk(person_names_callback)
    dataset.walk(curves_callback)

    ###############################################################################
    # pydicom allows to remove private tags using ``remove_private_tags`` method

    # dataset.remove_private_tags()

    ###############################################################################
    # Data elements of type 3 (optional) can be easily deleted using ``del`` or
    # ``delattr``.

    if 'OtherPatientIDs' in dataset:
        delattr(dataset, 'OtherPatientIDs')

    if 'OtherPatientIDsSequence' in dataset:
        del dataset.OtherPatientIDsSequence

    ###############################################################################
    # For data elements of type 2, this is possible to blank it by assigning a
    # blank string.

    # tag = 'PatientBirthDate'
    # if tag in dataset:
    #     dataset.data_element(tag).value = '19000101'

    ###############################################################################
    # Custom code for visual field DCM anonymization
    dataset.PatientName = anonymization_fun(dataset.PatientName)
    dataset.ReferringPhysicianName = anonymization_fun(dataset.ReferringPhysicianName)
    dataset.PatientID = anonymization_fun(dataset.PatientID)
    dataset.PatientBirthDate = dataset.PatientBirthDate[:6] + '01'

    # Add a minimal valid PDF file: https://stackoverflow.com/a/17280876/6610243
    # dataset.EncapsulatedDocument = b'%PDF-1.\ntrailer<</Root<</Pages<</Kids[<</MediaBox[0 0 3 3]>>]>>>>>>'
