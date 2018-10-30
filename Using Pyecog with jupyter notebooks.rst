==================================
Pyecog as a python module
==================================

Currently the smoothest way to use pyecog is to combine python code and the GUI. While it is possible to do
everything in the GUI, it is a little clunky and any uncaught error will currently crash the program.

It is suggested to use jupyter notebooks for ndf to h5 file conversion, feature extraction and the training & applying
of classifiers. Furthermore, by interacting with the files through python directly you do not restrict your
analysis to pre-coded routines.

Loading the pyecog module
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the tutorial notebooks below, we will be loading the pyecog module into python. The easiest way to do this
is by placing the notebook in the directory downloaded from github, e.g. Pyecog-Master as the pyecog module will be
found in this folder. However, if you want to run the notebook from else where on your computer you first need
to make sure that python can find the pyecog module using sys.path.append().
To do this copy the following code into a cell and run it (shift+enter).::

    import sys
    pyecog_path = '/home/jonathan/git_repos/pyecog' # replace this with the Pyecog-Master location
    sys.path.append(pyecog_path)


.. toctree::
   :maxdepth: 1
   :caption: Tutorial Notebooks:

   Notebook 1 Converting Ndf files to h5py.ipynb
