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

SFA_DCM_ENCAPSULATED_TYPE = Tag(0x2201, 0x1000)
SFA_DCM_ENCAPSULATED_VERSION = Tag(0x2201, 0x1001)
SFA_DCM_ENCAPSULATED_INFO = Tag(0x2201, 0x1002)
SFA_DCM_REPORT_TYPE = Tag(0x22A1, 0x1001)
SFA_DCM_REPORT_TYPE = Tag(0x2501, 0x1000)  # Both of these fields seem to be the same...
SFA_DCM_UNKNOWN = Tag(0x2501, 0x1007)
SFA_DCM_UNKNOWN = Tag(0x2501, 0x1008)
SFA_DCM_PATTERN = Tag(0x7717, 0x1002)
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
        anonymized_pdf_parser = self.pdf_parser.anonymize(anonymization_fun=anonymization_fun)
        dataset.EncapsulatedDocument = anonymized_pdf_parser.raw_pdf

        return HFADCMParser(dataset=dataset)

    def save_as(self, filename):
        self.dataset.save_as(filename)

    @property
    def name(self):
        return self.dataset.PatientName

    @property
    def dob(self):
        return datetime.datetime.strptime(self.dataset.PatientBirthDate, "%Y%m%d")

    @property
    def gender(self):
        return self.dataset.PatientSex

    @property
    def id(self):
        return self.dataset.PatientID

    @property
    def laterality(self):
        value = self.dataset.Laterality
        value = {"R": "OD", "L": "OS"}[value]
        assert value == "OS" or value == "OD"
        return value

    @property
    def report_type(self):
        return self.dataset[SFA_DCM_REPORT_TYPE].value

    @property
    def pattern(self):
        return self.dataset[SFA_DCM_PATTERN].value

    @property
    def fixation_monitor(self):
        return self.dataset[SFA_DCM_FIXATION_MONITOR].value

    @property
    def fixation_target(self):
        return self.dataset[SFA_DCM_FIXATION_TARGET].value

    @property
    def fixation_loss_error(self):
        if SFA_DCM_FL_ERROR in self.dataset: #check if the tag exists
            return self.dataset[SFA_DCM_FL_ERROR].value
        else:
            return 0

    @property
    def fixation_loss_total(self):
        if SFA_DCM_FL_TOTAL in self.dataset: #check if the tag exists
            return self.dataset[SFA_DCM_FL_TOTAL].value
        else:
            return 1

    @property
    def fixation_loss(self):
        return self.fixation_loss_error * 1.0 / self.fixation_loss_total

    @property
    def false_positive(self):
        return self.dataset[SFA_DCM_FP].value / 100.0

    @property
    def false_negative(self):
        return self.dataset[SFA_DCM_FN].value / 100.0

    @property
    def stimulus_size(self):
        return self.dataset[SFA_DCM_STIMULUS_SIZE].value

    @property
    def stimulus_color(self):
        return self.dataset[SFA_DCM_STIMULUS_COLOR].value

    @property
    def stimulus_background(self):
        return self.dataset[SFA_DCM_STIMULUS_BACKGROUND].value

    @property
    def pupil_diameter(self):
        return self.dataset[SFA_DCM_PUPIL_DIAMETER].value

    @property
    def rx(self):
        return self.dataset[SFA_DCM_RX].value

    @property
    def datetime(self):
        value = self.dataset.AcquisitionDateTime
        return datetime.datetime.strptime(value, "%Y%m%d%H%M%S.%f")

    @property
    def ght(self):
        return self.dataset[SFA_DCM_GHT].value

    @property
    def vfi(self):
        return self.dataset[SFA_DCM_VFI].value

    @property
    def md(self):
        return self.dataset[SFA_DCM_MD].value

    @property
    def mdsig(self):
        return self.dataset[SFA_DCM_MDSIG].value

    @property
    def psd(self):
        return self.dataset[SFA_DCM_PSD].value

    @property
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
