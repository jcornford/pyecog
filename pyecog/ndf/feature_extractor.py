import time
import sys
import logging

import numpy as np
import scipy.stats as stats

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
        self.powerbands = [(1,4),(4,8),(8,12),(12,30),(30,50),(50,70),(70,120)]
        self.powerband_titles = [str(i)+'-'+str(ii)+' Hz' for i,ii in self.powerbands]

        if extract:
            self.get_basic_stats()
            self.baseline(threshold = 1.5, window_size = 80) # window size needs to be frequency dependent
            self.peaks_and_valleys(delta_val = 3.0)
            self.bandpow_arr = self.calculate_powerbands(self.flat_data,
                                                         chunk_len=self.chunk_len,
                                                         fs = self.fs,
                                                         bands = self.powerbands)

            self.feature_array = np.hstack([self.basic_stats,
                                            self.baseline_features,
                                            self.pkval_stats,
                                            self.bandpow_arr])

            self.col_labels = np.hstack([self.basic_stats_col_labels,
                                         self.baseline_features_labels,
                                         self.pkval_labels,
                                         self.powerband_titles])

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
            Maybe could switch out for pandas dataframe in future?

        Notes:
            - Band power units are "y unit"^2
            - using rfft for slightly improved efficiency - a scaling factor for power
            computation needs to be included
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
        psd = 2*chunk_len*(2/.375)*(A/(fs*chunk_len*2))**2  # psd units are "y unit"^2/Hz
        # A/(fs*chunk_len*2) is the normalised fft transform of windowed_data
        # - rfft only returns the positive frequency amplitudes.
        # To compute the power we have to sum the squares of both positive and negative complex frequency amplitudes
        # .375 is the power of the Hanning window - the factor 2/.375 corrects for these two points.
        # 2*chunk_len - is a factor such that the result comes as a density with units "y unit"^2/Hz and not just "y unit"^2
        # this is just to make the psd usable for other purposes - in the next section bin_width*2*chunk_len equals 1.

        # now grab power bands from psd
        bin_width = np.diff(bin_frequencies[:3])[1] # this way is really not needed?
        pb_array = np.zeros(shape = (psd.shape[0],len(bands)))
        for i,band in enumerate(bands):
            lower_freq, upper_freq = band # unpack the band tuple
            band_indexes = np.where(np.logical_and(bin_frequencies > lower_freq, bin_frequencies<=upper_freq))[0]
            bp = np.sum(psd[:,band_indexes], axis = 1)*bin_width
            pb_array[:,i] = bp

        return pb_array

    @staticmethod
    def rolling_window(array, window):
        """
        Remember that the rolling window actually starts at the window length in.
        """
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    def baseline(self, threshold = 1.5, window_size = 80):
        '''
        #TODO: Window size should be fs dependent?

        - used to have mean baseline diff and also baseline diff skew for vincents model
          kept, diff (with a fudge for when few datapoints, but not skew.) Though from looking,
          skew seems very predictive of pre seizure?

        - keep in
        '''
        data = self.dataset

        #first_flatten_data_array
        flat_data = np.ravel(data, order = 'C')
        array_std = np.std(self.rolling_window(flat_data, window=window_size),-1)
        array_window_missing = np.zeros(window_size-1) # goes on at the front
        rolling_std_array = np.hstack((array_window_missing,array_std))

        masked_std_below_threshold = np.ma.masked_where(rolling_std_array > threshold, flat_data)
        masked_std_above_threshold = np.ma.masked_where(rolling_std_array < threshold, flat_data)
        std_below_arr = np.reshape(masked_std_below_threshold, newshape = data.shape, order = 'C')
        #std_above_arr = np.reshape(masked_std_above_threshold, newshape = data.shape, order = 'C')
        baseline_length = np.array([len(np.ma.compressed(std_below_arr[i,:])) for i in range(data.shape[0])])

        indexes = np.array(np.arange(data.shape[1]))
        baseline_diff_skew = np.array([stats.skew(np.diff(indexes[np.logical_not(std_below_arr[i,:].mask)]))
                                       for i in range(data.shape[0])])
        baseline_mean_diff = np.array([np.mean(np.diff(indexes[np.logical_not(std_below_arr[i,:].mask)]))
                              for i in range(data.shape[0])])

        # below is to handle when you have very few baseline points...
        # Not actually that satisfactory.
        less_bl_datapoints = np.where(baseline_length<100)[0]
        baseline_mean_diff[less_bl_datapoints] = self.fs/(baseline_length[less_bl_datapoints]+2) # a hack to overcome if no baselines...
        baseline_diff_skew[less_bl_datapoints] = 0.0  # a hack to overcome if no baselines...


        baseline_features = np.vstack([baseline_length, baseline_mean_diff,baseline_diff_skew]).T
        baseline_features_labels = ['bl_len','bl_mean_diff', 'bl_skew']

        return(baseline_features)

    def peaks_and_valleys(self, delta_val = 3.0):
        '''underlying peak det function is slow...
        the range and values dont seem to be that great...
        number a bit more interesting? nval and npk is often same but not alowas one to one

        Should really be using:
        https://gist.github.com/sixtenbe/1178136
        '''

        logging.debug("Extracting peak/valley features...")
        pk_nlist = []
        val_nlist = []

        for i in range(self.dataset.shape[0]):
            pk, val = self.peakdet(self.dataset[i,:], delta=delta_val)
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
        col_mins = np.nanmin(self.pkval_stats, axis = 0)
        col_mins[3] = np.nanmax(self.pkval_stats[:,3], axis =0) # hacky
        for row, col in np.argwhere(np.isnan(self.pkval_stats)):
            self.pkval_stats[row,col] = col_mins[col]

        self.pkval_labels = ['n_pks','n_vals','av_pk','av_val','av_pkval_range']

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
            x = np.arange(len(v))
        v = np.asarray(v)
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        lookformax = True
        for i in np.arange(len(v)):
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
        return np.array(maxtab), np.array(mintab)


