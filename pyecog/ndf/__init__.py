from .ndfconverter import NdfFile
from .h5loader import H5File
from .datahandler import DataHandler
from .classifier import FeaturePreProcesser
from .classifier import Classifier
from .classifier import ClassificationAlgorithm
from .hmm_pyecog import HMMBayes, HMM


from .feature_extractor import FeatureExtractor

from . import classifier_utils

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