"""
General parser and anonymization utilities

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
from .dcm import HFADCMParser
from pathlib import Path


def parse(filepath):
    """
    Factory method to construct the appropriate parser object based on the input file

    Parameters
    ----------
    filepath: str or Path

    Returns
    -------
    Corresponding parser object
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.suffix.lower() == ".pdf":
        with filepath.open("rb") as f:
            parser = HFAPDFParser(f)
    elif filepath.suffix.lower() == ".dcm":
        with filepath.open("rb") as f:
            parser = HFADCMParser(f)
    else:
        raise NotImplemented(f"{filepath} is not supported.")

    return parser
