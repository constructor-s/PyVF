"""

Copyright 2022 Bill Runjie Shi
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

from unittest import TestCase

from pyvf.plot import pretty_print_vf, get_pretty_print_grid
from pyvf.strategy import PATTERN_P24D2, PATTERN_P30D2, PATTERN_P10D2
import numpy as np

class Test_Printer(TestCase):
    def test_pretty_print_vf(self):
        print(pretty_print_vf(np.arange(76), pattern=PATTERN_P10D2, apply_style=True))
