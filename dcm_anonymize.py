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

from argparse import ArgumentParser
import pydicom
import pyvf.parse

parser = ArgumentParser("Anonymize Zeiss Forum HFA visual field data")
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
args = parser.parse_args()

dataset = pydicom.dcmread(args.input)
pyvf.parse.dcmanonymize(dataset)
print(dataset.AcquisitionDateTime)
print(dataset[pyvf.parse.SFA_DCM_MD])
print(dataset[pyvf.parse.SFA_DCM_MDSIG])
print(dataset[pyvf.parse.SFA_DCM_PSD])
print(dataset[pyvf.parse.SFA_DCM_PSDSIG])
print(dataset[pyvf.parse.SFA_DCM_GHT])
print(dataset[pyvf.parse.SFA_DCM_VFI])
dataset.save_as(args.output)
