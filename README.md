[![PyPI version](https://badge.fury.io/py/pyecog.svg)](https://badge.fury.io/py/pyecog)
## PyECoG
This module is for detecting epileptiform activity from *single* channel intracranial EEG (or ECoG) recordings.
Currently under heavy construction!

Use python 3.6, untested and likely (even more) buggy with 2. 



### Recommended installation procedure (using pip):

1. Install Anaconda. Choose the Python 3.6 64-bit version for your operating system (Linux, Windows, or OS X).
  You can also use python 2, but just make sure your python version for the environment is 3. 
2. Make a new environment and install dependencies: Open a terminal windows (on Windows,a cmd prompt) and type or copy :
    ```{bash}
    conda create --name pyecog_env python=3.5 jupyter=1 scipy=0.18.1 numpy=1.11.2 scikit-learn=0.17.1 pandas=0.19.2 matplotlib=2 seaborn=0.7.1 h5py=2.6.0 xlrd=1 bokeh=0.12.4 pyqt=5.6
    source activate pyecog_env  # or just "activate pyecog_env" if on windows
    pip install pyqtgraph==0.10
    pip install pomegranate==0.6.1
    ```
    #### WARNING: pomegranate often fails to build properly!
    To test your installation, try:
    ```{bash}
    python
    >>> import pomegranate
    ```
    If you get the following error:
    ```{bash}
    ImportError: No module named 'pomegranate.utils'
    ```
    This can be resolved by uninstalling and reinstalling:
    ```{bash}
    >>> quit()
    pip uninstall pomegranate
    pip install pomegranate --no-cache
    ```

3. Finally, you are set to run 
```{bash}
pip install pyecog
```


### Recommended procedure for running PyECoG from Github source code:
1. Follow steps 1 and 2 from above
2. Download the latest source code from [here](https://github.com/jcornford/pyecog/archive/master.zip) and extract.
3. Navigate to the folder in terminal/cmd, or open a terminal/cmd window at the extracted folder.
```{bash}
source activate pyecog_env  # or just "activate pyecog_env" if on windows
python
>>> import pyecog
>>> pyecog
<module 'pyecog' from 'YOUR-PATH_HERE/pyecog-master/pyecog/__init__.py'>
```

Like when installing using pip, you can load up a jupyter notebook or run scripts from here and import the module.
If you want to run the gui:
```{bash}
python
>>> pyecog.pyecog_main_gui.main()
```

You also have the option to run the gui from command line /terminal:

```{bash}
pyecog/visualisation/pyecog_main_gui.py 
```

### How to use
- [Loading ndf files] (https://github.com/jcornford/pyecog/blob/master/documentation_notebooks/demo_loading_ndfs_notebook.ipynb)

### Repository contents:
* NDF:          code is the current working directory.
* light_code:   contains old code. PyECoG was originally "networkclassifer", this is that code, kept for analysing further experiments.
* visualisation: contains a bunch of visualisation experiments

ps. maybe pomegranate problems can be solved by:q
cython-0.25.2 joblib-0.11 networkx-1.11


