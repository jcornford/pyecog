import time
import sys
import logging
import warnings
import numpy as np
import scipy.stats as stats

class StdDevStandardiser():
    def __init__(self,data, std_sigfigs = 2):
        '''
        Calculates mode std dev and divides by it.

        Args:
            data:
            stdtw: time period over which to calculate std deviation
            std_sigfigs: n signfigs to round to

        '''
        logging.debug('Standardising dataset to mode std dev')
        #std_vector = self.round_to_sigfigs(np.std(data, axis = 1), sigfigs=std_sigfigs)
        std_vector = np.round(np.std(data, axis = 1), 0)
        #logging.debug(str(std_vector))
        std_vector = std_vector[std_vector != 0]
        if std_vector.shape[0] > 0:
            self.mode_std = stats.mode(std_vector)[0] # can be zero if there is big signal loss
            self.scaled = np.divide(data, self.mode_std)
            logging.debug(str(self.mode_std)+' is mode std of trace split into ')
        elif std_vector.shape[0] == 0:
            self.scaled = None
            logging.error(' File std is all 0, changed data to be None')

    @staticmethod
    def round_to_sigfigs(x, sigfigs):
        """
        N.B Stolen from stack overflow:
        http://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy

        Rounds the value(s) in x to the number of significant figures in sigfigs.
        Restrictions:
        sigfigs must be an integer type and store a positive value.
        x must be a real value or an array like object containing only real values.
        """
        #The following constant was computed in maxima 5.35.1 using 64 bigfloat digits of precision
        __logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1
        if not ( type(sigfigs) is int or np.issubdtype(sigfigs, np.integer)):
            raise TypeError( "RoundToSigFigs: sigfigs must be an integer." )
        if not np.all(np.isreal( x )):
            raise TypeError( "RoundToSigFigs: all x must be real." )
        if sigfigs <= 0:
            raise ValueError( "RoundtoSigFigs: sigfigs must be positive." )
        mantissas, binaryExponents = np.frexp(x)
        decimalExponents = __logBase10of2 * binaryExponents
        intParts = np.floor(decimalExponents)
        mantissas *= 10.0**(decimalExponents - intParts)
        return np.around(mantissas, decimals=sigfigs - 1 ) * 10.0**intParts

class FeatureExtractor():
    '''
    TODO:
     - Window on the baseline should be fs dependent
     - store the conversion factor for the scaling to std/ implement this all here, not in the
     - pandas df?
     - peakdet method is very slow, replace with  https://gist.github.com/sixtenbe/1178136? does it add much?
     - use of self and basically staticmethods is all jumbled up
     - flat data and chunked array is a little jumbled too
     - skew and mean diff of baseline is not actually working
    '''

    def __init__(self, data, fs, extract = True, run_peakdet = True):
        '''
        Inputs:
            - Data: should already be chunked into desired windows. Assumes this is on hour's worth.
            - fs: sampling frequency
            - extract: a flag to extract all the features - default true

        N.B Scaling to unit std should be done here!
        '''

        self.start = time.clock()
        standardiser = StdDevStandardiser(data,std_sigfigs=2)
        self.dataset = standardiser.scaled
        self.mode_std = standardiser.mode_std

        self.chunk_len = 3600/self.dataset.shape[0] # get this as an arg?
        self.flat_data = np.ravel(self.dataset, order = 'C')
        self.fs = fs

        logging.debug('Fs passed to Feature extractor was: '+str(fs)+' hz')
        # add in 120-160 if we have the fs for it:
        if self.fs < 512:
            self.powerbands = [(1,4),(4,8),(8,12),(12,30),(30,50),(50,70),(70,120)]
            self.powerband_titles = [str(i)+'-'+str(ii)+' Hz' for i,ii in self.powerbands]

        elif self.fs >= 512 :
            self.powerbands = [(1,4),(4,8),(8,12),(12,30),(30,50),(50,70),(70,120), (120,160)]
            self.powerband_titles = [str(i)+'-'+str(ii)+' Hz' for i,ii in self.powerbands]

        if extract:
            self.get_basic_stats()
            self.baseline(threshold = 1.5, window_size = 80) # window size needs to be frequency dependent

            self.bandpow_arr = self.calculate_powerbands(self.flat_data,
                                                         chunk_len=self.chunk_len,
                                                         fs = self.fs,
                                                         bands = self.powerbands)
            # this split is clunky, change this
            if run_peakdet:
                self.peaks_and_valleys(delta_val = 3.0)
                self.feature_array = np.hstack([self.basic_stats,
                                            self.baseline_features,
                                            self.pkval_stats,
                                            self.bandpow_arr])

                self.col_labels = np.hstack([self.basic_stats_col_labels,
                                             self.baseline_features_labels,
                                             self.pkval_labels,
                                             self.powerband_titles])
            else:
                self.feature_array = np.hstack([self.basic_stats,
                                            self.baseline_features,
                                            self.bandpow_arr])

                self.col_labels = np.hstack([self.basic_stats_col_labels,
                                             self.baseline_features_labels,
                                             self.powerband_titles])

            timetaken = np.round((time.clock()-self.start), 2)
            logging.debug("Stacking up..."+str(self.feature_array.shape))
            logging.debug('Took '+ str(timetaken) + ' seconds to extract '+ str(self.dataset.shape[0])+ ' feature vectors')

    def get_basic_stats(self):
        logging.debug("Extracting basic stats...")
        self.basic_stats_col_labels = ('min', 'max', 'mean', 'std-dev', 'kurtosis', 'skew', 'sum(abs(difference))')
        self.basic_stats = np.zeros((self.dataset.shape[0],7))
        self.basic_stats[:,0] = np.amin(self.dataset, axis = 1)
        self.basic_stats[:,1] = np.amax(self.dataset, axis = 1)
        self.basic_stats[:,2] = np.mean(self.dataset, axis = 1)
        self.basic_stats[:,3] = np.std(self.dataset, axis = 1)
        self.basic_stats[:,4] = stats.kurtosis(self.dataset,axis = 1)
        self.basic_stats[:,5] = stats.skew(self.dataset, axis = 1)
        self.basic_stats[:,6] = np.sum(np.absolute(np.diff(self.dataset, axis = 1)),axis = 1)

        # get rid of pesky NaNs, from interpolating over corruptions etc, in more controlled way than just imputing later
        self.basic_stats[:,4][np.isnan(self.basic_stats[:,4])] = -3.0
        self.basic_stats[:,5][np.isnan(self.basic_stats[:,5])] = 0.0
        self.basic_stats[:,4][np.isinf(self.basic_stats[:,4])] = -3.0
        self.basic_stats[:,5][np.isinf(self.basic_stats[:,5])] = 0.0


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
        - - `threshold has been chosen assuming the data has been scaled to mode std dev

        - used to have mean baseline diff and also baseline diff skew for vincents model
          kept, diff (with a fudge for when few datapoints, but not skew.) Though from looking,
          skew seems very predictive of pre seizure?

        - keep in
        '''
        data = self.dataset

        flat_data = np.ravel(data, order = 'C')
        array_std = np.std(self.rolling_window(flat_data, window=window_size),-1)
        array_window_missing = np.zeros(window_size-1) # goes on at the front
        rolling_std_array = np.hstack((array_window_missing,array_std))

        below_threshold_masked = np.ma.masked_where(rolling_std_array > threshold, flat_data)
        above_threshold_masked = np.ma.masked_where(rolling_std_array < threshold, flat_data)
        # masked_where puts a mask where the condition is met, so sign is other way to what you would expect
        # when using indexing, i.e.: above_threshold = flat_data[np.where(rolling_std_array > threshold)[0]]

        below_std_arr = np.reshape(below_threshold_masked, newshape = data.shape, order = 'C')
        above_std_arr = np.reshape(above_threshold_masked, newshape = data.shape, order = 'C')

        baseline_length = below_std_arr.count(axis = 1) # .count counts numb of non masked elements
        #print(baseline_length[:10])

        # build indexes array:
        indexes_row = np.arange(data.shape[1])
        indexes_arr = np.empty(data.shape)
        for i in range(data.shape[0]): indexes_arr[i, :] = indexes_row

        bl_indexes = np.ma.masked_array(indexes_arr, mask = below_std_arr.mask)

        # have to use lists and compressed, or diff ignores the jump
        # this is all very suboptimal feature engineering - MARCO?
        diff_arr_list = [np.diff(bl_indexes[i,:].compressed()) for i in range(data.shape[0])]
        baseline_index_diff_skew = np.array([stats.stats.skew(diff_arr_list[i]) for i in range(data.shape[0])])
        with warnings.catch_warnings():# suppress the mean of empty slice warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            baseline_mean_diff       = np.array([np.mean(diff_arr_list[i]) for i in range(data.shape[0])])

        # hereshould make diff array once - it is masked itself
        #baseline_index_diff_skew = stats.mstats.skew(np.diff(bl_indexes, axis = 1), axis = 1)
        #baseline_mean_diff       = np.mean(np.diff(bl_indexes, axis = 1), axis = 1)

        # below is to handle when you have very few baseline points..
        #  Not actually that satisfactory.
        less_bl_datapoints = np.where(baseline_length<100)[0]
        baseline_mean_diff[less_bl_datapoints] = self.fs/(baseline_length[less_bl_datapoints]+2) # a hack to overcome if no baselines...
        baseline_index_diff_skew[less_bl_datapoints] = 0.0

        #same just in case mask slips through somehow
        #if baseline_mean_diff.mask.any():
        #    baseline_mean_diff[baseline_mean_diff.mask] = self.fs/(baseline_length[less_bl_datapoints]+2)
        #if baseline_index_diff_skew.mask.any():
        #    baseline_index_diff_skew[baseline_index_diff_skew.mask]  = 0.0

        self.baseline_features = np.vstack([baseline_length, baseline_mean_diff,baseline_index_diff_skew]).T
        self.baseline_features_labels = ['bl_len','bl_mean_diff', 'bl_skew']
        #print(self.baseline_features)

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