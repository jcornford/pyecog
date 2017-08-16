[![PyPI version](https://badge.fury.io/py/pyecog.svg)](https://badge.fury.io/py/pyecog)
## PyECoG
This module is for detecting epileptiform activity from *single* channel intracranial EEG (or ECoG) recordings.
Currently under heavy construction!

Use python 3.5. 

### Recommended installation procedure for devlopment version from Github:

1. Install [Anaconda](https://www.continuum.io/downloads). Choose the Python 3 64-bit version for your operating system (Linux, Windows, or OS X).
  You can also use python 2, but just make sure your python version for the environment is 3. 
2. Make a new environment and install dependencies. To do this open a terminal windows (on Windows, a cmd prompt) and type or copy :
    ```{bash}
    conda create --name pyecog python=3.5 jupyter=1 scipy=0.18.1 numpy=1.11.2 scikit-learn=0.17.1 pandas=0.19.2 matplotlib=2 seaborn=0.7.1 h5py=2.6.0 xlrd=1 pyqt=5.6
    source activate pyecog_dev  # or just "activate pyecog_dev" if on windows
    pip install pyqtgraph==0.10
    pip install pyecog
    
3. Navigate to the folder in terminal/cmd, or open a terminal/cmd window at the extracted folder.

4. Finally, you are ready to run. You can open the pyecog gui type:

```{bash}
source activate pyecog  # or just "activate pyecog" if on windows
pyecog
```

### How to use - this needs to be updated
- note, some gui elements not implemented (open in jupyter notebook and low pass filter)
- [Loading ndf files] (https://github.com/jcornford/pyecog/blob/master/documentation_notebooks/demo_loading_ndfs_notebook.ipynb)

### Repository contents:
* NDF:           code is the current working directory.
* visualisation: contains a bunch of visualisation experiments


