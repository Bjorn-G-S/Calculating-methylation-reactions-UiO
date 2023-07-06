# Calculating-methylation-reactions


* [General](#general-info)
* [Purpose](#purpose)
* [Instalation](#installation)
* [How-to](#how-to)
* [Contact](#Contact)
* [License](#License)


## General

`Calculating-methylation-reactions` is a python program to be used for 
analyzing GCdata from excel recorded from a MTMTH experiment.
Currently `Calculating-methylation-reactions` is used at the section for catalysis at
the University of Oslo.

## Purpose

Analyze and visualize data from GC measurments taken during methylation experiments with ethylene and propylene

## Installation

Use the program by download to a directory accesible for you python enviornment. The enviornment need to have `python>=3.9.7`, `numpy>=1.21.2`, `pandas>=1.3.3`.


## How-to
How to use the program.


1. Import the program:
```
From MTMTH_excel import *
```
2. Define the directory of the files that are to be converted and the file with the x-vlaues:
```
obj = excel_MTMTH_Calc(Directory=XXXXXX)
```
3. Run the program. the following message will the apear:

## Dependencies

This script requires a python enviornment with the following packages, and the packaged these depend on:
```
python          (3.9.7)
pandas          (1.3.3)
numpy           (1.21.2)
matplotlib      (3.4.3)
ipywidgets      (7.6.5)
```

## Contact

For developer issues, please start a ticket in Github. You can also write to the dev team directly at  **b.g.solemsli@smn.uio.no**
#### Authors: 
Bjørn Gading Solemsli (@bjorngso).

## License
This script is under the MIT license scheme. 



