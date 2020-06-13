# Converting the saplocmap data in the R visualFields to CSV format
# Also resets the loc indexing to zero based indexing for use in zero based 
# indexing languages
# Please see https://cran.r-project.org/web/packages/visualFields/visualFields.pdf
# and its associated licenses
#
#
# Copyright 2020 Bill Runjie Shi
# At the Vision and Eye Movements Lab, University of Toronto.
# Visit us at: http://www.eizenman.ca/
#
# This file is part of PyVF.
#
# PyVF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyVF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyVF. If not, see <https://www.gnu.org/licenses/>.


load("../../resources/R/visualFields/data/saplocmap.rda")
for (key in names(saplocmap)) {
  print(key)
  print(paste("saplocmap_", key, ".csv", sep=""))
  df = saplocmap[[key]]
  df$loc = df$loc - 1
  write.csv(df, file = paste("saplocmap_", key, ".csv", sep=""), row.names = FALSE)
}
