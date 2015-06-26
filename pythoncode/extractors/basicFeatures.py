import numpy as np
import featureBaseClass
import scipy.stats as st
import sklearn
from sklearn import preprocessing

class BasicFeatures():
    """
    Bunch of basic statistics features e.g. skew, kurtosis, coefficient of variation, coastline.
    As of 2015.06.19 data is only 0 meaned! Variation has been left alone

    """

    def __init__(self):
        #self.names = ['kurtosis','skew','std','coastline', 'norm_coastline', 'max_val','pos_crossings' ]
        self.names = ['std','norm_coastline', 'coastline','pos_crossings', 'moment6', 'skew']

    def extract(self, data):
        # data array is no longer zero meaned by default!
        kurtosis = st.kurtosis(data, axis=1)
        
        skew = st.skew(data, axis=1)
        # remeber skew is slightly fucked by 0 mean?
        # maybe tailed is better?
        std = np.std(data, axis=1)
        coastline = self._coastline(data)
        norm_std_data = data/np.std(data, axis = 1)[:,None]
        norm_coastline = self._coastline(norm_std_data)
        max_val = np.max(data, axis=1)

        all_std_data = data/np.std(data, axis = 0)
        pos_crossings = self._zero_crossings(all_std_data-4)

        moment6 = st.moment(data,6,axis=1)

        #3print variation[:,None].shape
        '''
        features = np.hstack((#kurtosis[:,None],
                              skew[:,None],
                              std[:,None],
                              coastline[:,None],
                              norm_coastline[:,None],
                              #max_val[:,None],
                              pos_crossings[:,None],
                              
                              ))
        '''
        features = np.hstack((
                              std[:,None],
                              norm_coastline[:,None],
                              coastline[:,None],
                              pos_crossings[:,None],
                              moment6[:,None],
                              skew[:,None],
                              
                              ))
        print features.shape
        Inan = np.where(np.isnan(features))
        Iinf = np.where(np.isinf(features))
        features[Inan] = 0
        features[Iinf] = 0
        features = self._normalize(features)
        return features
    def _zero_crossings(self, data_array):
        signs = np.sign(data_array)
        zero_crossings = abs(np.diff(signs,axis = 1)/2)
        result = np.sum(zero_crossings,axis=1)
        return result

    def _coastline(self, data_array):
        #print 'here'
        coastline = np.absolute(np.diff(data_array,axis = 1))
        features = np.sum(coastline, axis =1)
        normalised_coastline = (features-min(features))/(max(features)-min(features))
        return normalised_coastline
    def _normalize(self,series):
            a = np.min(series, axis = 0)
            b = np.max(series, axis = 0)
            return np.divide((series - a),(b - a))