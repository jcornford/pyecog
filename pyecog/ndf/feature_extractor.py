import time
import sys
import logging
import warnings
import numpy as np
import scipy.stats as stats


class StdDevStandardiser():
    def __init__(self, data, std_sigfigs=2):
        '''
        Calculates mode std dev of data and divides by it.

        # we should test median here
        Args:
            data: this should be chunked by timewindow
            std_sigfigs: n signfigs to round to, default 2. Important for the
            "resolution" of the mode standard dev.

        '''
        std_vector = np.round(np.std(data, axis=1), 0)
        std_vector = std_vector[std_vector != 0]

        if std_vector.shape[0] > 0:
            self.mode_std = stats.mode(std_vector)[0]
            self.scaled_data = np.divide(data, self.mode_std)
            logging.debug(str(self.mode_std) + ' is mode std of trace split into ')

        # recording might not contain any data that has std != 0
        elif std_vector.shape[0] == 0:
            self.scaled_data = None
            logging.error('File std is all 0, changed self.scaled_data to be None')


class FeatureExtractor():
    '''
    Notes:
     - Whatever calls this probably wants to store the conversion factor for the scaling to std.
     - Might be nice idea to return pd.dataframe
     - https://gist.github.com/sixtenbe/1178136 could be used for peak detections
    '''

    def __init__(self, data, fs, extract=True):
        '''
        Arguments:
            - data: this should already be chunked into desired windows over which to calculate features.
              Currently it is assumed that this is one hour.
            - fs: sampling frequency of the data
            - extract: a bool specifying whether to extract all the features on class init. If false,
              feature methods should be called directly.
        '''
        # first add noise to the data in the order of the least significant bit
        data = data + 0.4 * np.random.randn(data.shape[0], data.shape[1])

        std_dev_standardiser = StdDevStandardiser(data, std_sigfigs=2)
        self.dataset = std_dev_standardiser.scaled_data
        self.scale_coef_for_feature_extraction = std_dev_standardiser.mode_std

        self.chunk_len = 3600 / self.dataset.shape[0]  # get this as an arg?
        self.flat_data = np.ravel(self.dataset, order='C')
        self.fs = fs

        logging.debug('Standardising dataset by scaling with mode std dev' + str(self.scale_coef_for_feature_extraction))
        logging.debug('Fs passed to Feature extractor was: ' + str(fs) + ' hz')

        if self.fs < 512:
            self.powerbands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50), (50, 70), (70, 120)]
            self.powerband_titles = [str(i) + '-' + str(ii) + ' Hz' for i, ii in self.powerbands]

        elif self.fs == 512:
            self.powerbands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50), (50, 70), (70, 120), (120, 160)]
            self.powerband_titles = [str(i) + '-' + str(ii) + ' Hz' for i, ii in self.powerbands]

        elif self.fs == 1024:
            self.powerbands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50), (50, 70), (70, 110), (110, 150), (150, 200),
                               (200, 250), (250, 320)]
            self.powerband_titles = [str(i) + '-' + str(ii) + ' Hz' for i, ii in self.powerbands]

        if extract:
            self.get_basic_stats()
            self.baseline(threshold=1.5, window_size=80)  # window size needs to be frequency dependent

            self.bandpow_arr = self.calculate_powerbands(self.flat_data,
                                                         chunk_len=self.chunk_len,
                                                         fs=self.fs,
                                                         bands=self.powerbands)

            self.feature_array = np.hstack([self.basic_stats,
                                            self.baseline_features,
                                            self.bandpow_arr])

            self.col_labels = np.hstack([self.basic_stats_col_labels,
                                         self.baseline_features_labels,
                                         self.powerband_titles])

            logging.debug("Stacking up..." + str(self.feature_array.shape))

    def get_basic_stats(self):
        logging.debug("Extracting basic stats...")
        self.basic_stats_col_labels = ('min', 'max', 'mean', 'std-dev', 'kurtosis', 'skew', 'sum(abs(difference))')
        self.basic_stats = np.zeros((self.dataset.shape[0], 7))
        self.basic_stats[:, 0] = np.amin(self.dataset, axis=1)
        self.basic_stats[:, 1] = np.amax(self.dataset, axis=1)
        self.basic_stats[:, 2] = np.mean(self.dataset, axis=1)
        self.basic_stats[:, 3] = np.std(self.dataset, axis=1)
        self.basic_stats[:, 4] = stats.kurtosis(self.dataset, axis=1)
        self.basic_stats[:, 5] = stats.skew(self.dataset, axis=1)
        self.basic_stats[:, 6] = np.sum(np.absolute(np.diff(self.dataset, axis=1)), axis=1)

        # typically NaN if drop of signal, so use kurtosis of uniform dist.
        self.basic_stats[:, 4][np.isnan(self.basic_stats[:, 4])] = - 6 / 5

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
        pad_dp = int((chunk_len / 2) * fs)
        padded_data = np.pad(flat_data, pad_width=pad_dp, mode='reflect')

        # reshape data into array, stich together
        data_arr = np.reshape(padded_data,
                              newshape=(int(len(padded_data) / fs / chunk_len), int(chunk_len * fs)),
                              # changed in order to compensate for files which are not 3600seconds long
                              order='C')
        data_arr_stiched = np.concatenate([data_arr[:-1], data_arr[1:]], axis=1)

        # window the data with hanning window
        hanning_window = np.hanning(data_arr_stiched.shape[1])
        windowed_data = np.multiply(data_arr_stiched, hanning_window)

        # run fft and get psd
        bin_frequencies = np.fft.rfftfreq(int(chunk_len * fs) * 2, 1 / fs)  # *2 as n.points is doubled due to stiching
        A = np.abs(np.fft.rfft(windowed_data, axis=1))
        psd = 2 * chunk_len * (2 / .375) * (A / (fs * chunk_len * 2)) ** 2  # psd units are "y unit"^2/Hz
        # A/(fs*chunk_len*2) is the normalised fft transform of windowed_data
        # - rfft only returns the positive frequency amplitudes.
        # To compute the power we have to sum the squares of both positive and negative complex frequency amplitudes
        # .375 is the power of the Hanning window - the factor 2/.375 corrects for these two points.
        # 2*chunk_len - is a factor such that the result comes as a density with units "y unit"^2/Hz and not just "y unit"^2
        # this is just to make the psd usable for other purposes - in the next section bin_width*2*chunk_len equals 1.

        # now grab power bands from psd
        bin_width = np.diff(bin_frequencies[:3])[1]  # this way is really not needed?
        pb_array = np.zeros(shape=(psd.shape[0], len(bands)))
        for i, band in enumerate(bands):
            lower_freq, upper_freq = band  # unpack the band tuple
            band_indexes = np.where(np.logical_and(bin_frequencies > lower_freq, bin_frequencies <= upper_freq))[0]
            bp = np.sum(psd[:, band_indexes], axis=1) * bin_width
            pb_array[:, i] = bp

        return pb_array

    @staticmethod
    def rolling_window(array, window):
        """
        Remember that the rolling window actually starts at the window length in.
        """
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    def baseline(self, threshold=1.5, window_size=80, just_bl_len=True):
        '''
        #TODO: Window size should be fs dependent?
        - - `threshold has been chosen assuming the data has been scaled to mode std dev

        - used to have mean baseline diff and also baseline diff skew for vincents model
          kept, diff (with a fudge for when few datapoints, but not skew.) Though from looking,
          skew seems very predictive of pre seizure?

        '''
        flat_data = self.flat_data
        array_std = np.std(self.rolling_window(flat_data, window=window_size), -1)
        array_window_missing = np.zeros(window_size - 1)  # goes on at the front
        rolling_std_array = np.hstack((array_window_missing, array_std))
        below_threshold_masked = np.ma.masked_where(rolling_std_array > threshold, flat_data)
        above_threshold_masked = np.ma.masked_where(rolling_std_array < threshold, flat_data)
        # masked_where puts a mask where the condition is met, so sign is other way to what you would expect
        # when using indexing, i.e.: above_threshold = flat_data[np.where(rolling_std_array > threshold)[0]]
        below_std_arr = np.reshape(below_threshold_masked, newshape=self.dataset.shape, order='C')
        #above_std_arr = np.reshape(above_threshold_masked, newshape=self.dataset.shape, order='C')
        baseline_length = below_std_arr.count(axis=1)  # .count counts numb of non masked elements

        self.baseline_features = np.vstack([baseline_length]).T
        self.baseline_features_labels = ['bl_len']