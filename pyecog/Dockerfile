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
RUN /opt/conda/bin/pip install pomegranate --no-cache-dir
RUN /opt/conda/bin/pip install pyqtgraph
RUN /opt/conda/bin/pip install pyecog

USER root
#RUN apt-get install -y openssh-server libx11-dev libxext-dev x11-apps
RUN apt-get update
RUN apt-get install -y openssh-server libx11-dev libxext-dev x11-apps
RUN mkdir /var/run/sshd
EXPOSE 22
ENV KEYFILE=/home/jovyan/.ssh/id_rsa
RUN mkdir /home/jovyan/.ssh; chown jovyan /home/jovyan/.ssh
RUN /usr/bin/ssh-keygen -q -t rsa -N "" -f $KEYFILE; cat $KEYFILE.pub >> /home/jovyan/.ssh/authorized_keys
RUN echo "IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config
RUN touch /home/jovyan/.Xauthority; chown jovyan /home/jovyan/.Xauthority
CMD ["/usr/sbin/sshd", "-D"]
RUN apt-get install -qqy x11-apps
ENV DISPLAY :0

# Switch back to jovyan to avoid accidental container runs as root
#USER $NB_USER

docker build -t xeyes - << __EOF__
FROM debian
RUN apt-get update
RUN apt-get install -qqy x11-apps
ENV DISPLAY $DISPLAY
CMD xeyes
__EOF__
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -ti -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e DISPLAY=$DISPLAY xeyes