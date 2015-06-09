import numpy as np
import featureBaseClass
import scipy.stats as st

class FreqFeatures():
    """
    Extracting frequency features

    """

    def __init__(self):
        pass

    def extract(self, data):
        features = np.log(np.absolute(np.fft.rfft(data, axis = 1)[:,1:48]))
        return features
