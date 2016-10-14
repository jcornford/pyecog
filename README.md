[![PyPI version](https://badge.fury.io/py/pyecog.svg)](https://badge.fury.io/py/pyecog)
## PyECoG
This module is for detecting epileptiform activity from *single* channel intracranial EEG (or ECoG) recordings.
Currently under heavy construction!

Strongly recommended to use python 3.5, untested and likely (more) buggy with 2. 

### Installing PyECoG
It is advised to create a conda python 3.5 enviroment and to run:

```{bash}
pip install pyecog
```
### Dependency issues:

##### Installing PyQt4
You will also need to install PyQt4 to use the gui elements:
Unfortunately PyQt4 cannot be installed from pip (http://stackoverflow.com/questions/22640640/how-to-install-pyqt4-on-windows-using-pip), you will need to install from here:
https://riverbankcomputing.com/software/pyqt/download

Alternatively, for Windows you can download the appropriate package from here:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4
Then open command prompt and navigate to the folder that the package is in and enter:
```{bash}
pip install PyQt4-4.11.4-cp35-none-win_amd64.whl
```
If you do not have Microsoft Visual Studio installed, you may also need to install Visual C++ Redistributable for Visual Studio 2015 (https://www.microsoft.com/en-gb/download/details.aspx?id=48145)


##### Installing Pomegranate
This is required for the Hidden Markov decoding of classifier predictions. There are sometimes issues: (https://github.com/jmschrei/pomegranate/issues/114)

```python
38 from .hmm import *
...
ImportError: No module named utils
```
This can be solved with:

```bash
pip uninstall pomegranate
pip install pomegranate --no-cache-dir
```
On windows you will also need to install visual studio in order to install pomegranate.

### How to use
- [Loading ndf files] (https://github.com/jcornford/pyecog/blob/master/documentation_notebooks/demo_loading_ndfs_notebook.ipynb)

### Repository contents:
* NDF:          code is the current working directory.
* light_code:   contains old code. PyECoG was originally "networkclassifer", this is that code, kept for analysing further experiments.
* visualisation: contains a bunch of visualisation experiments




