from __future__ import print_function
import time
import sys

import pickle
import numpy as np
import scipy.stats as st
#import matplotlib.pyplot as plt


from numpy import NaN, Inf, arange, isscalar, asarray, array # imports for the peakdet method


class FeatureExtractor():

    def __init__(self, dataset, subtract_baseline = True):
        self.start = time.clock()
        self.dataset = dataset
        self._baseline(subtract_baseline=subtract_baseline, threshold=0.04, window_size=100)
        self._event_dataset_stats()
        self._peaks_and_valleys()
        self._wavelet_features(fs = 512, frequencies = [1,5,10,15,20,30,60, 90])

        print("Stacking up...")
        self.feature_array = np.hstack([self.event_stats,self.baseline_features,
                                   self.pkval_stats,self.meanpower])

        print(str(self.feature_array.shape))
        print('took '+ str(time.clock()-self.start)+ ' seconds to extract '+ str(dataset.shape[0])+ 'feature vectors')


    def _event_dataset_stats(self):
        print("Extracting stats on event dataset...")
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

        print("DONE"+ str(time.clock() - self.start))

    def _baseline(self, subtract_baseline = True, threshold = 0.04, window_size = 100):
        '''
        Method calculates 2 feature lists and an event dataset:
            - self.event_dataset is the data window with 'baseline points removed'
            - baseline_length is the first feature list and is the number of datapoints within the window
              defined as being baseline using a rolling std deviation window and threshold
            - baseline_mean_diff is the second list and is the mean difference between baseline datapoint indexes.
        '''
        print('Extracting baseline via rolling std...')
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
        print('DONE'+ str(time.clock() - self.start))
        #print 'N.B using mean baseline difference - perhaps not best?'

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
        #print ('N.B Again - preassign?')

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

        #print("N.B. Still no post crossing power")

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
            print('freqNorm is :'+str(A)+' for'+str(f) +' Hz')

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

