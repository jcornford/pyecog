from .ndfconverter import NdfFile
from .h5loader import H5File
from .datahandler import DataHandler
from .classifier import FeaturePreProcesser
from .classifier import Classifier
from .classifier import load_classifier
#from .bokeh_visualisation import plot
#from .bokeh_visualisation import basic_plot
from .feature_extractor import FeatureExtractor
from .hmm import make_hmm_model

import os
import logging
#try:
#    logging.info('Re-intted')
#except:
logger = logging.getLogger()
logpath = os.getcwd()
#fhandler = logging.FileHandler(filename=os.path.join(os.path.split(logpath)[0], 'Datahandler.log'), mode='w')
fhandler = logging.FileHandler(filename=os.path.join(logpath, 'PyECoG_logfile.log'), mode='a+')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)