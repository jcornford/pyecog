import numpy as np
import FeatureBaseClass
import scipy.stats as st

class BasicFeatures():
    """
    Bunch of basic statistics features e.g. skew, kurtosis, coefficient of variation, coastline.

    """

    def __init__(self):
        pass

    def extract(self, data):
        kurtosis = st.kurtosis(data, axis=1)
        skew = st.skew(data, axis=1)
        variation = st.variation(data, axis=1)
        coastline = self._coastline(data)
        print variation[:,None].shape
        
        features = np.hstack((kurtosis[:,None], skew[:,None], variation[:,None], coastline[:,None]))
        Inan = np.where(np.isnan(features))
        Iinf = np.where(np.isinf(features))
        features[Inan] = 0
        features[Iinf] = 0
        
        return features
    def _coastline(self, data_array):
        print 'here'
        coastline = np.absolute(np.diff(data_array,axis = 1))
        features = np.sum(coastline, axis =1)
        normalised_coastline = (features-min(features))/(max(features)-min(features))
        return normalised_coastline