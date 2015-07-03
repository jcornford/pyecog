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
        mask = np.where(data[:,:-513]<-3,1,0)
        mask1 = np.roll(mask[:],1)
        mask = abs(np.subtract(mask,mask1))

        crossings = []
        for i in range(mask.shape[0]):
           crossings.append(np.where(mask[i,:])[0][1::2])
           # the second [] is to slice for the only one crossing



        psfreqs = []
        for ti in range(data.shape[0]):
            segments = []
            if len(crossings[ti])>0:
                for i in crossings[ti]:
                    segments.append(data[ti,i:i+512])
                segments = np.array(segments)
                res = np.mean((np.absolute(np.fft.rfft(segments, axis = 1)[:,20:40])), axis = 0)
                psfreqs.append(res)
            else:
                psfreqs.append(np.zeros([20]))
        psfreqs = np.array(psfreqs)
        self.names = range(psfreqs.shape[1])
        return psfreqs
        #features = np.log(np.absolute(np.fft.rfft(data, axis = 1)[:,1:200]))
        #features  = sklearn.preprocessing.scale(features, axis = 0)
        #print 'here'
        #return features

