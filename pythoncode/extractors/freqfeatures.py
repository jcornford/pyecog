import numpy as np
import featureBaseClass
import scipy.stats as st
import sklearn

class FreqFeatures():
    """
    Extracting frequency features

    """

    def __init__(self):
        pass

    def extract(self, data):
        features = np.log(np.absolute(np.fft.rfft(data, axis = 1)[:,1:200]))
        features  = sklearn.preprocessing.scale(features, axis = 0)
        self.names = range(features.shape[1])
        print 'here'
        return features
