import time
import sys

import pickle
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


from numpy import NaN, Inf, arange, isscalar, asarray, array # imports for the peakdet method


class FeatureExtractor():

    def __init__(self, dataset, subtract_baseline = True):
        self.start = time.clock()
        self.dataset = dataset
        self._baseline(subtract_baseline=subtract_baseline, threshold=0.04, window_size=100)
        self._event_dataset_stats()
        self._peaks_and_valleys()
        self._wavelet_features(fs = 512, frequencies = [1,5,10,15,20,30,60, 90])

        print("Stacking up..."),
        self.feature_array = np.hstack([self.event_stats,self.baseline_features,
                                   self.pkval_stats,self.meanpower])
        print 'DONE!'

        print self.feature_array.shape
        print 'took ', time.clock()-self.start, ' seconds to extract ', dataset.shape[0], 'feature vectors'


    def _event_dataset_stats(self):
        print("Extracting stats on event dataset..."),
        self.event_stats = np.zeros((len(self.event_dataset),7))
        self.event_stats_col_labels = ('min','max','mean','std-dev','skew','kurtosis','sum(abs(difference))')

        for i,trace in enumerate(self.event_dataset):
            if trace.shape[0]:
                # if not all baseline
                self.event_stats[i,0] = np.min(trace)
                self.event_stats[i,1] = np.max(trace)
                self.event_stats[i,2] = np.mean(trace)
                self.event_stats[i,3] = np.std(trace)
                self.event_stats[i,4] = st.kurtosis(trace)
                self.event_stats[i,5] = st.skew(trace)
                self.event_stats[i,6] = np.sum(np.absolute(np.diff(trace)))
            else:
                 # if all baseline
                self.event_stats[i,:] = np.NaN

        print "DONE", time.clock() - self.start

    def _baseline(self, subtract_baseline = True, threshold = 0.04, window_size = 100):
        '''
        Method calculates 2 feature lists and an event dataset:
            - self.event_dataset is the data window with 'baseline points removed'
            - baseline_length is the first feature list and is the number of datapoints within the window
              defined as being baseline using a rolling std deviation window and threshold
            - baseline_mean_diff is the second list and is the mean difference between baseline datapoint indexes.
        '''
        print 'Extracting baseline via rolling std...',
        array_std = np.std(self.rolling_window(self.dataset,window=window_size),-1)
        array_window = np.zeros([self.dataset.shape[0],window_size-1])
        rolling_std_array = np.hstack((array_window,array_std))
        masked_std_below_threshold = np.ma.masked_where(rolling_std_array > threshold, self.dataset)
        mean_baseline_vector = np.mean(masked_std_below_threshold,axis = 1)

        if subtract_baseline:
            dataset_after_subtraction_option = self.dataset - mean_baseline_vector[:,None]
        else:
            dataset_after_subtraction_option = self.dataset

        masked_std_above_threshold = np.ma.masked_where(rolling_std_array < threshold, dataset_after_subtraction_option)
        indexes = np.array(np.arange(self.dataset.shape[1]))

        self.event_dataset = [np.ma.compressed(masked_std_above_threshold[i,:]) for i in xrange(self.dataset.shape[0])]
        baseline_length = [len(np.ma.compressed(masked_std_below_threshold[i,:])) for i in xrange(self.dataset.shape[0])]
        baseline_diff_skew = [st.skew(np.diff(indexes[np.logical_not(masked_std_below_threshold[i].mask)])) for i in xrange(self.dataset.shape[0])]
        baseline_mean_diff = [np.mean(np.diff(indexes[np.logical_not(masked_std_below_threshold[i].mask)])) for i in xrange(self.dataset.shape[0])]
        #self.baseline_diff = [indexes[np.logical_not(masked_std_below_threshold[i].mask)] for i in xrange(dataset.shape[0])]

        self.baseline_features = np.vstack([baseline_length,baseline_mean_diff,baseline_diff_skew]).T
        print 'DONE', time.clock() - self.start
        print 'N.B using mean baseline difference - perhaps not best?'

    def _peaks_and_valleys(self):
        print ("Extracting peak/valley features..."),
        pk_nlist = []
        val_nlist = []

        for i in range(self.dataset.shape[0]):
            pk, val = self.peakdet(self.dataset[i,:], 0.5)
            pk_nlist.append(pk)
            val_nlist.append(val)

        # always 2 columns
        n_pks = []
        n_vals = []
        av_pk = []
        av_val = []
        av_range = []
        for i in range(len(pk_nlist)):
            n_pks.append(pk_nlist[i].shape[0])
            n_vals.append(val_nlist[i].shape[0])

            if pk_nlist[i].shape[0]:
                av_pk.append(np.mean(pk_nlist[i][:,1]))
            else:
                av_pk.append(np.NaN)

            if val_nlist[i].shape[0]:
                av_val.append(np.mean(val_nlist[i][:,1]))
            else:
                av_val.append(np.NaN)

            if val_nlist[i].shape[0] and pk_nlist[i].shape[0]:
                av_range.append(
                    abs(np.mean(pk_nlist[i][:,1])+np.mean(val_nlist[i][:,1]))
                    )
            else:
                av_range.append(np.NaN)

        n_pks = np.array(n_pks)
        n_vals = np.array(n_vals)
        av_pk = np.array(av_pk)
        av_val = np.array(av_val)
        av_range = np.array(av_range)
        self.pkval_stats = np.vstack([n_pks,n_vals,av_pk,av_val,av_range]).T
        print("DONE", time.clock()-self.start)
        print ('N.B Again - preassign?')

    def _wavelet_features(self, fs = 512, frequencies = [1,5,10,15,20,30,60, 90]):
        print("Extracting wavelet features from event dataset..."),
        window = int(fs*0.5)
        waveletList = []
        for freq in frequencies:
            wavelet, xaxis = self.complexMorlet(freq,fs,timewindow = (-0.1,0.1))
            waveletList.append(wavelet)

        power = []
        meanpower = []
        for i in range(len(self.event_dataset)):
            if self.event_dataset[i].shape[0]:
                convResults = self.convolve(self.event_dataset[i], waveletList)
                powerArray  = np.absolute(convResults)
                #print powerArray.shape
                power.append(powerArray)
                meanpower.append(np.mean(powerArray,axis = 1))

            else:
                power.append(np.ones((len(frequencies)))*np.NaN)
                meanpower.append(np.ones((len(frequencies)))*np.NaN)
        self.meanpower = np.array(meanpower)
        print("DONE", time.clock()-self.start)

        print("N.B. Still no post crossing power")

    @ staticmethod
    def rolling_window(array, window):
        """
        Remember that the rolling window actually starts at the window length in.
        """
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    @staticmethod
    def peakdet(v, delta, x=None):
        """
        function [maxtab, mintab]=peakdet(v, delta, x)
                [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                maxima and minima ("peaks") in the vector V.
                MAXTAB and MINTAB consists of two columns. Column 1
                contains indices in V, and column 2 the found values.

                With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
                in MAXTAB and MINTAB are replaced with the corresponding
               X-values.

                A point is considered a maximum peak if it has the maximal
                value, and was preceded (to the left) by a value lower by
                DELTA.
        """
        maxtab = []
        mintab = []
        if x is None:
            x = arange(len(v))
        v = asarray(v)
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN
        lookformax = True
        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        return array(maxtab), array(mintab)

    @staticmethod
    def complexMorlet(freq, samplingFreq, timewindow,
                  noWaveletCycles=6,freqNorm = False):
        '''
        freq              = wavelet frequency
        samplingFrequency = 'insert data sampling frequency here'
        noWaveletCycles   = 'reread variable name' - trades off temp and freq precision
        timewindow        = timewindow over which to calculate wavelet
        freqNorm          = Bool, decide whether to normlise over frequencies
        plotWavelet       = Plot the construction?
        '''
        n = noWaveletCycles
        f = freq
        timeaxis = np.linspace(timewindow[0],timewindow[1],samplingFreq*(2.*timewindow[1]))
        sine_wave = np.exp(2*np.pi*1j*f*timeaxis)
        gstd = n/(2.*np.pi*f)
        gauss_win = np.exp(-timeaxis**2./(2.*gstd**2.))

        A = 1 # just 1 if not normlising freqs
        if freqNorm:
            #gstd    = (4/(2*np.pi*f))**2# this is his?
            A = 1.0/np.sqrt((gstd*np.sqrt(np.pi)))
            print 'freqNorm is :',A,' for', f, ' Hz'

        wavelet = A*sine_wave*gauss_win
        #print wavelet.shape

        return wavelet, timeaxis

    @staticmethod
    def convolve(dataSection, wavelets):
        convResults = []
        for i in range(len(wavelets)):
            wavelet = wavelets[i]
            complexResult = np.convolve(dataSection,wavelet,'same')
            convResults.append(complexResult)

        convResults = np.array(convResults) # convert to a numpy array!
        #print convResults.shape
        return convResults





#dataset = pickle.load(open('../dataset','rb'))
#extractor = FeatureExtractor(dataset)


# begin piss-around
'''
def rolling_std(X, w):
    print time.clock(),'rolling std start'
    r = np.empty(X.shape)
    r.fill(np.nan)
    for i in range(w - 1, X.shape[0]):
        r[i] = np.std(X[(i-w+1):i+1])
    print time.clock(),'rolling std end'
    return r

def rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

r_std = rolling_std(dataset[6,:], 100)
threshold = 0.04
bis = np.where(r_std<threshold)[0]
#plt.plot(r_std, 'r')


print array_std.shape
print array_window.shape
rolling_std_array = np.hstack((array_window,array_std))
print rolling_std_array.shape

plt.plot(rolling_std_array[6,:], 'k')

plt.plot(dataset[6,:],'g')
plt.plot(np.arange(5120)[r_std<threshold],dataset[6,:][r_std<threshold],'rx')
plt.plot(np.arange(5120)[rolling_std_array[6,:]<threshold],dataset[6,:][rolling_std_array[6,:]<threshold],'kx')

print r_std.shape

baseline_point_subtraction = True
start  = time.clock()
threshold = 0.04
window_size = 100
array_std = np.std(rolling_window(dataset,window=window_size),-1)
array_window = np.zeros([dataset.shape[0],window_size-1])
rolling_std_array = np.hstack((array_window,array_std))
masked_std_below_threshold = np.ma.masked_where(rolling_std_array > threshold, dataset)

mean_baseline_vector = np.mean(masked_std_below_threshold,axis = 1)
if baseline_point_subtraction:
    dataset_after_subtraction_option = dataset - mean_baseline_vector[:,None]
masked_std_above_threshold = np.ma.masked_where(rolling_std_array < threshold, dataset_after_subtraction_option)
event_dataset = [np.ma.compressed(masked_std_above_threshold[i,:]) for i in xrange(dataset.shape[0])]

baseline_length = [len(np.ma.compressed(masked_std_below_threshold[i,:])) for i in xrange(dataset.shape[0])]

indexes = np.array(np.arange(5120))
baseline_diff = [indexes[np.logical_not(masked_std_below_threshold[i].mask)] for i in xrange(dataset.shape[0])]



#print baseline_diff_pre.shape, 'is diff_shape'
# need to grab baseine length, and also baseline

print baseline_length[6], 'is baseline len of 6'
for i in xrange(100):
    print baseline_diff[6][i+50]
    #print baseline_diff[6,i], 'is baseline diff'
    print masked_std_below_threshold[6,i+50],
    #print 'diff is', baseline_diff_pre[6,i+50]

baseline_dp = []
    for i in range(dataset.shape[0]):
        rstdi = rolling_std(dataset[i,:], 100,)# thresh = 0.04)
        e_index = np.where(rstdi>threshold)[0]
        b_index = np.where(rstdi<threshold)[0]
        baseline_dp.append(b_index)


    baseline_length = []
    baseline_mean_diff = []

    for dp_is in baseline_dp:
        blen_sec = dp_is.shape[0]/512.
        baseline_length.append(blen_sec)

        blen_diff = np.diff(dp_is)
        baseline_mean_diff.append(np.mean(blen_diff))

    baseline_length = np.array(baseline_length)
    baseline_mean_diff = np.array(baseline_mean_diff)
    print("Done baseline len")
    print time.clock()

print 'to do rolling etc, took ', time.clock() - start, 'seconds'
print rolling_std_array.shape

print mean_baseline_vector[6]

print mean_baseline_vector.shape
plt.plot(dataset_after_subtraction_option[6,:],'b')
#plt.plot(masked_std_below_threshold[6,:],'r')
plt.plot(masked_std_above_threshold[6,:],'kx')
plt.plot(event_dataset[6],'g')
#plt.plot(baseline_diff[6],'r')
plt.show()
'''

