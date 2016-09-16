import time
import sys
import logging

import numpy as np
import scipy.stats as st
import scipy.stats as stats
#import matplotlib.pyplot as plt


from numpy import NaN, Inf, arange, isscalar, asarray, array # imports for the peakdet method

class FeatureExtractor():

    def __init__(self, dataset, fs, extract = True):
        '''
        Inputs:
            Dataset: should already be chunked into desired windows. Pass extract as False to select which features to run.
            Assuming this is on hour's worth

            fs: samling frequency
            extract: a flag to eaxtract all the features - default true

        Should we move the scaling to here - in order to leave the units correct elsewhere?
        '''

        self.start = time.clock()
        self.dataset = dataset
        self.chunk_len = 3600/self.dataset.shape[0]
        self.flat_data = np.ravel(dataset, order = 'C')
        self.fs = fs
        logging.debug('Fs passed to Feature extractor was: '+str(fs)+' hz')

        powerbands = [(1,4),(4,8),(8,12),(12,30),(30,50),(50,70),(70,120)]
        powerband_titles = [str(i)+'-'+str(ii)+' Hz' for i,ii in powerbands]

        if extract:
            self.get_basic_stats()
            #self.baseline( threshold=0.04, window_size=100)
            self.peaks_and_valleys()
            self.calculate_powerbands(self.flat_data, chunk_len= self.chunk_len, fs = fs,
                                      bands =  powerbands)


            self.feature_array = np.hstack([self.basic_stats,self.baseline_features,
                                       self.pkval_stats,self.meanpower])

            self.col_labels = np.hstack([self.basic_stats_col_labels, self.baseline_features_labels,
                                         self.pkval_labels, powerband_titles])

            timetaken = np.round((time.clock()-self.start), 2)
            logging.debug("Stacking up..."+str(self.feature_array.shape))
            logging.debug('Took '+ str(timetaken) + ' seconds to extract '+ str(dataset.shape[0])+ ' feature vectors')

    def get_basic_stats(self):
        logging.debug("Extracting basic stats...")
        self.basic_stats_col_labels = ('min', 'max', 'mean', 'std-dev', 'skew', 'kurtosis', 'sum(abs(difference))')

        self.basic_stats = np.zeros((self.dataset.shape[0],7))
        self.basic_stats[:,0] = np.amin(self.dataset, axis = 1)
        self.basic_stats[:,1] = np.amax(self.dataset, axis = 1)
        self.basic_stats[:,2] = np.mean(self.dataset, axis = 1)
        self.basic_stats[:,3] = np.std(self.dataset, axis = 1)
        self.basic_stats[:,4] = stats.kurtosis(self.dataset,axis = 1)
        self.basic_stats[:,5] = stats.skew(self.dataset, axis = 1)
        self.basic_stats[:,6] = np.sum(np.absolute(np.diff(self.dataset, axis = 1)))

    @staticmethod
    def calculate_powerbands(flat_data, chunk_len, fs, bands):
        '''
        Inputs:
            - flat_data: np.array, of len(arr.shape) = 1
            - chunk_len: length in seconds of chunk over which to calculate bp
            - fs       : sampling frequency in Hz
            - bands    : list of tuples, containing bands of interest.
            e.g. [(1,4),(4,8),(8,12),(12,30),(30,50),(50,70),(70,120)]

        Returns:
            - numpy array, columns correspond to powerbands.
            Maybe could switch out for pandas dataframe?

        Notes:
            - Band power units are "y unit"^2?... Aren't they Marco! ;]
            - using rfft so scaling factors added in are: (Marco could you complete this)
            - using a hanning window, so reflecting half of the first and last chunk in
            order to centre the window over the time windows of interest (and to not throw
            anything away)

        '''

        # first reflect the first and last half chunk length so we don't lose any time
        pad_dp = int((chunk_len/2) * fs)
        padded_data = np.pad(flat_data, pad_width=pad_dp, mode = 'reflect')

        # reshape data into array, stich together
        data_arr = np.reshape(padded_data,
                              newshape=(int(3600/chunk_len)+1, int(chunk_len*fs)), # +1 due to padding
                              order= 'C')
        data_arr_stiched = np.concatenate([data_arr[:-1], data_arr[1:]], axis = 1)

        # window the data with hanning window
        hanning_window = np.hanning(data_arr_stiched.shape[1])
        windowed_data  = np.multiply(data_arr_stiched, hanning_window)

        # run fft and get psd
        bin_frequencies = np.fft.rfftfreq(int(chunk_len*fs)*2, 1/fs) # *2 as n.points is doubled due to stiching
        A = np.abs(np.fft.rfft(windowed_data, axis = 1))
        psd = (2/.375)*(A/(2*fs))**2 # psd units are "y unit"V^2/hz
        # 2/.375 and 2*fs are due to rfft (Marco? comment here please)

        # now grab power bands from psd
        bin_width = np.diff(bins[:3])[1]
        pb_array = np.zeros(shape = (psd.shape[0],len(power_bands)))
        for i,band in enumerate(power_bands):
            lower_freq, upper_freq = band # unpack the band tuple
            band_indexes = np.where(np.logical_and(bins >= lower_freq, bins<=upper_freq))[0]
            bp = np.sum(psd[:,band_indexes], axis = 1)*bin_width
            pb_array[:,i] = bp

        return pb_array

    def rolling_window(array, window):
        """
        Remember that the rolling window actually starts at the window length in.
        """
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    def baseline(self, threshold = 1.5, window_size = 80):
        '''
        Window size should be fs dependent?

        - used to have mean baseline diff and also baseline diff skew for vincents model, kept in for now.
        '''

        #first_flatten_data_array
        flat_data = np.ravel(data, order = 'C')
        array_std = np.std(rolling_window(flat_data, window=window_size),-1)
        array_window_missing = np.zeros(window_size-1) # goes on at the front
        rolling_std_array = np.hstack((array_window_missing,array_std))


        masked_std_below_threshold = np.ma.masked_where(rolling_std_array > threshold, flat_data)
        masked_std_above_threshold = np.ma.masked_where(rolling_std_array < threshold, flat_data)
        std_below_arr = np.reshape(masked_std_below_threshold, newshape = data.shape, order = 'C')
        #std_above_arr = np.reshape(masked_std_above_threshold, newshape = data.shape, order = 'C')
        baseline_length = np.array([len(np.ma.compressed(std_below_arr[i,:])) for i in range(data.shape[0])])

        indexes = np.array(np.arange(data.shape[1]))
        baseline_diff_skew = np.array([st.skew(np.diff(indexes[np.logical_not(std_below_arr[i,:].mask)]))
                                       for i in range(data.shape[0])])
        baseline_mean_diff = np.array([np.mean(np.diff(indexes[np.logical_not(std_below_arr[i,:].mask)]))
                              for i in range(data.shape[0])])
        less_50_bl_datapoints = np.where(baseline_length<50)[0]
        baseline_mean_diff[less_50_bl_datapoints] = fs/(baseline_length[less_50_bl_datapoints]+2) # a hack to overcome if no baselines...

        print('***********')
        print(baseline_length[110:130])
        print(baseline_diff_skew[10:130])
        print(baseline_mean_diff[110:130])
        # what to do here with no mean diff or skew?! fully postive skewed? mean diff is the full window?
        # then we have one's as everthing is there - skew?

        baseline_features = np.vstack([baseline_length,baseline_mean_diff,baseline_diff_skew]).T
        baseline_features_labels = ['bl_len','bl_mean_diff', 'bl_diff_skew']

        return(baseline_features)

    def peaks_and_valleys(self):

        logging.debug("Extracting peak/valley features...")
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
        self.pkval_labels = ['n_pks','n_vals','av_pk','av_val','av_pkval_range']


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


