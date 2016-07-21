[![PyPI version](https://badge.fury.io/py/pyecog.svg)](https://badge.fury.io/py/pyecog)
## PyECoG
This module is for detecting epileptiform activity from single channel intracranial EEG (or ECoG) recordings.
Currently under heavy construction.

### Installing PyECoG
```{bash}
pip install pyecog
```

### Originally "networkclassifer"
This was originally a bunch of scripts for classifying brain-network states for acute models. The state in a given timewindow before light pulses that activated neurons. These scripts are still in the "lightcode" directory in case needed again.


### Usage:

    TODO:
    ndf/datahandler.py
     - bind parrallel ndf and convert ndf dir into one function
     - use the h5file class to clean up the code
     - add predictions to file should also be same as ndf - one file
     - everything needs to 

### Todo:
* currently very ndf file heavy
* gui for visualising traces and storing annotations

### Todo - more minor things
* AR coefs.
* Normalisation options
* Allow easy insertion of custom features
* speed up file conversion
* handle errors better
* make tests
* decide on how to store data in hdf5 files, currently in favour of one file per transmitter_id
* bad message filtering for fs != 512


