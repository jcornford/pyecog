from abc import abstractmethod
class FeatureBaseClass(object):
    '''
    Abstract base class for feature extraction.
    '''
    
    @abstractmethod
    def extract(self,data_instance):
        '''
        Method to extract features.

        Returns:
        A 2d numpy array with features.
        '''
        raise NotImplementedError()
        
    def assert_features(self, features):
        assert(type(features)==np.ndarray)
        assert(len(features.shape)==1)
        assert(features.shape[0]>=1)

    def __str__(self):
        # subclass may override this. Be sure to make it readable.
        return type(self).__name__.split('.')[-1]