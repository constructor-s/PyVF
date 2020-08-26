"""
Parsing utilities for Zeiss Forum exports.
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
from pydicom.tag import Tag

SFA_DCM_ = Tag(0x7717, 0x1001)
SFA_DCM_ = Tag(0x7717, 0x1002)
SFA_DCM_ = Tag(0x7717, 0x1003)
SFA_DCM_ = Tag(0x7717, 0x1004)
SFA_DCM_ = Tag(0x7717, 0x1005)
SFA_DCM_ = Tag(0x7717, 0x1006)
SFA_DCM_ = Tag(0x7717, 0x1008)
SFA_DCM_ = Tag(0x7717, 0x1009)
SFA_DCM_ = Tag(0x7717, 0x1010)
SFA_DCM_ = Tag(0x7717, 0x1013)
SFA_DCM_MD = Tag(0x7717, 0x1016)
SFA_DCM_MDSIG = Tag(0x7717, 0x1017)
SFA_DCM_PSD = Tag(0x7717, 0x1018)
SFA_DCM_PSDSIG = Tag(0x7717, 0x1019)
SFA_DCM_GHT = Tag(0x7717, 0x1023)
SFA_DCM_ = Tag(0x7717, 0x1024)
SFA_DCM_ = Tag(0x7717, 0x1025)
SFA_DCM_ = Tag(0x7717, 0x1027)
SFA_DCM_ = Tag(0x7717, 0x1032)
SFA_DCM_ = Tag(0x7717, 0x1033)
SFA_DCM_VFI = Tag(0x7717, 0x1034)


def dcmanonymize(dataset):
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
    dataset.PatientName = 'anonymous^anonymous'
    dataset.ReferringPhysicianName = 'anonymous^anonymous'
    dataset.PatientID = "id"
    dataset.PatientBirthDate = dataset.PatientBirthDate[:6] + '01'

    # Add a minimal valid PDF file: https://stackoverflow.com/a/17280876/6610243
    dataset.EncapsulatedDocument = b'%PDF-1.\ntrailer<</Root<</Pages<</Kids[<</MediaBox[0 0 3 3]>>]>>>>>>'
