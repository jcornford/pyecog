[![PyPI version](https://badge.fury.io/py/pyecog.svg)](https://badge.fury.io/py/pyecog)
## PyECoG
This module is for detecting epileptiform activity from *single* channel intracranial EEG (or ECoG) recordings.
Currently under heavy construction! 

### Installing PyECoG
```{bash}
pip install pyecog
```

### How to use
- [Loading ndf files] (https://github.com/jcornford/pyecog/blob/master/documentation/demo_loading_ndfs_notebook.ipynb)

### Repository contents:
* NDF:          code is the current working directory.
* light_code:   contains old code. PyECoG was originally "networkclassifer", this is that code, kept for analysing further experiments.
* visualisation: contains a bunch of visualisation experiments


### Usage:



### Todo:
* Add windowed bandpower to features
* Glitch detection code is messy - relying on assignment bindings between attributes. 
* Parallel add_prediction_features method needs to have arguments passing to processes, e.g timewindow 
* add feature column headings to file...
* adding features to library file is currently not in parallel
* datahandler file is in need of refactoring
* logging can still spawn multiple processes in ipython notebooks (track this down)
* feature extractor file is in need of refactoring
* Gui for visualising traces and storing annotations - build on vispy/PyQtGraph?

### Long term to - more minor things
* AR coefs for condensed frequency .
* Allow easy insertion of custom features
* Write some tests already
* Decide on how to store data in hdf5 files, currently in favour of one file per transmitter_id



