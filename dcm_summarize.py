"""
Batch save visual field result indices into a CSV format

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
import glob
import itertools
from collections import namedtuple
import datetime
import pandas as pd

parser = ArgumentParser("Summarize visual field results to CSV")
parser.add_argument("-i", "--input", type=str, required=True, nargs="+", help="One or more glob patterns of .dcm files")
parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file")
args = parser.parse_args()

VFResult = namedtuple("VFResult", ["id", "age", "eye", "md", "psd", "vfi", "ght"])

results = []
for f in itertools.chain(*map(glob.glob, args.input)):
    dataset = pydicom.dcmread(f)
    pyvf.parse.dcmanonymize(dataset)

    dob = datetime.datetime.strptime(dataset.PatientBirthDate, "%Y%m%d")
    acquisition = datetime.datetime.strptime(dataset.AcquisitionDateTime, "%Y%m%d%H%M%S.%f")

    result = VFResult(
        id=dataset.PatientID,
        age=(acquisition-dob).days,
        eye=dataset.Laterality,
        md=float(dataset[pyvf.parse.SFA_DCM_MD].value),
        psd=float(dataset[pyvf.parse.SFA_DCM_PSD].value),
        vfi=float(dataset[pyvf.parse.SFA_DCM_VFI].value),
        ght=dataset[pyvf.parse.SFA_DCM_GHT].value
    )
    results.append(result)

df = pd.DataFrame(results)
df.to_csv(args.output, index=False)
