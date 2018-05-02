
How to install Pyecog
==================================

Note, there is a package on pip, however this has not been updated for a while, and will not work (probably). When the module is more stable I will release
on pip.

Installation procedure with an environment (recommended)
---------------------------------------------------------
1. Install [Anaconda](https://www.continuum.io/downloads). Choose the Python 3, 64-bit version for your operating system (Linux, Windows, or OS X). You can also download python 2, but it is recommended you use python version 3 for the conda environment (next step).

2. Make a new conda environment and install the dependencies for pyecog. To do this open a terminal windows (on Windows, a cmd prompt) and type or copy
   ::

       conda create --name pyecog_env python=3.5 jupyter=1 scipy=0.18.1 numpy=1.11.2 scikit-learn=0.18.1 pandas=0.19.2 matplotlib=2 seaborn=0.7.1 h5py=2.6.0 xlrd=1 pyqt=5.6 numba=0.37.0
       source activate pyecog_env  # or just "activate pyecog_env" if you are on windows
       pip install pyqtgraph==0.10

3. Click on the green "clone or download" button on the top right of the github project page: https://github.com/jcornford/pyecog/tree/master.


Download the zip file and extract it somewhere on your computer. Navigate to the extracted folder in terminal/cmd line or open a terminal/cmd window at the extracted folder.

4. Finally, you are ready to run. First activate your python environment that was made at step 2
   ::

       activate pyecog_env  # or  "source activate pyecog_env" if on a mac

5. And then run the gui with
   ::

       python start.py


Installing without a virtual python environment
------------------------------------------------

Replace step 2 with simply::
    conda install jupyter=1 scipy=0.18.1 numpy=1.11.2 scikit-learn=0.17.1 pandas=0.19.2 matplotlib=2 seaborn=0.7.1 h5py=2.6.0 xlrd=1 pyqt=5.6
    pip install pyqtgraph==0.10

 And then do not activate an enviroment before


Installing with pip
--------------------
This is for when the software is more developed. Reccomended to make the conda enviroment as before.

1 . After step 2 above, run::

   pip install pyecog

2. Ignore step 3. Open command window and activate the pyecog_env environment if you are using it. Type "pyecog" into the prompt. The Gui should load.
