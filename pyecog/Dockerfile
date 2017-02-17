FROM jupyter/minimal-notebook

MAINTAINER JCornford <jonathan.cornford@gmail.com>

USER root

## This is now an edited version of jupyter/scipy-notebook ##

# libav-tools for matplotlib anim
RUN apt-get update && \
    apt-get install -y --no-install-recommends libav-tools && \
    apt-get install python-qt4 -y  && \
    apt-get clean  && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER

# Install Python 3 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images - commented this out, don't know if this screw up
# pyqt4 - expect it will

RUN conda install --quiet --yes \
    'ipywidgets=5.2*' \
    'matplotlib=1.5*' \
    'pandas=0.19*' \
    'scipy=0.17*' \
    'seaborn=0.7*' \
    'bokeh=0.11*' \
    'hdf5=1.8.17' \
    'h5py=2.6*' \
    'scikit-learn=0.18*' \
    'xlrd'  && \


    #conda remove --quiet --yes --force qt pyqt && \
    conda clean -tipsy

# Activate ipywidgets extension in the environment that runs the notebook server
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Add shortcuts to distinguish pip for python2 and python3 envs
RUN ln -s $CONDA_DIR/bin/pip $CONDA_DIR/bin/pip3

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/

# Configure ipython kernel to use matplotlib inline backend by default
RUN mkdir -p $HOME/.ipython/profile_default/startup
#COPY mplimporthook.py $HOME/.ipython/profile_default/startup/

USER root


# Switch back to jovyan to avoid accidental container runs as root
USER $NB_USER

# below add the installation of pyecog
RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install pomegranate
RUN /opt/conda/bin/pip install pyqtgraph
RUN /opt/conda/bin/pip install pyecog

USER root
RUN apt-get install -y openssh-server libx11-dev libxext-dev x11-apps

# Switch back to jovyan to avoid accidental container runs as root
#USER $NB_USER