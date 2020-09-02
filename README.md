# PyVF
Python Visual Field simulation

_work in progress_

## Purpose

The current goal is to create a robust framework for testing visual field testing strategies and implementation of popular known strategies. This can then facilitate development of new strategies.

During this process, visual field analysis functions will be integrated. The goal is that portion can be used separately for visual field analysis.

Currently there is no active plan to interface with commercial perimeters for real subject testing. See [OPI - Open Perimetry Interface](https://github.com/turpinandrew/OPI).

## Usage
### Requirements

See [`requirements.txt`](requirements.txt). Latest version of Python (3.8) installed using an [Anaconda](https://www.anaconda.com/) environment using packages from `conda-forge` channel is recommended.

### An example

This performs a simulation of testing a range of true thresholds of a perfect responder, with a range of starting thresholds, using a 4-2 double staircase algorithm, on a single location. 

```
git clone https://github.com/constructor-s/PyVF.git
cd PyVF
python sim_ds_single_perfect.py
```

## Select References
1. [OPI - Open Perimetry Interface](https://github.com/turpinandrew/OPI) Turpin A, Artes PH, McKendrick AM. The Open Perimetry Interface: an enabling tool for clinical visual psychophysics. J Vis. 2012;12(11):22. Published 2012 Jan 1. doi:10.1167/12.11.22
2. [R visualFields package](https://github.com/cran/visualFields) Marín-Franch I & Swanson WH. The visualFields package: A tool for analysis and visualization of visual fields. Journal of Vision, 2013, 13(4):10, 1-12 
3. Turpin, A., McKendrick, A. M., Johnson, C. A., & Vingrys, A. J. (2003). Properties of Perimetric Threshold Estimates from Full Threshold, ZEST, and SITA-like Strategies, as Determined by Computer Simulation. Investigative Ophthalmology and Visual Science, 44(11), 4787–4795. https://doi.org/10.1167/iovs.03-0023
4. Heijl, A., Lindgren, G., & Olsson, J. (1987). Normal Variability of Static Perimetric Threshold Values Across the Central Visual Field. Archives of Ophthalmology, 105(11), 1544–1549. https://doi.org/10.1001/archopht.1987.01060110090039
5. Hoehn, R., Häckel, S., Kucur, S., Iliev, M. E., Abegg, M., & Sznitman, R. (2019). Evaluation of Sequentially Optimized Reconstruction Strategy in visual field testing in normal subjects and glaucoma patients. Investigative Ophthalmology & Visual Science, 60(9), 2477.

## Author
Bill Shi at [Vision and Eye Movements Lab](http://www.eizenman.ca/), University of Toronto

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
