import sys
import struct
import os
import time
import logging

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy import signal, stats
#from line_profiler import LineProfiler

if sys.version_info < (3,):
    range = xrange
'''
# decorator needed when profiling
def lprofile():
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)

                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner
'''
class NdfFile:
    """
    TODO:
     - glitch detection is a little messy, relying on bindings.
     - bad messages doesnt delete if 2 messages in window for one (minor but would be better to delete both)
     - Find out if they ever start recording half way through - for auto fs detection
     - Code up printing the ndf object __repr__
     - Clean up unused  __init__ attributes

    Class to load ndf binary files.

    The NDF file starts with a header of at least twelve bytes:
        - The first four bytes spell the NDF identifier " ndf". The identifier is then
          followed by three four-byte big-endian numbers.
        - The first number is the address of the meta-data string, i.e. the byte offset from
          the first byte of the file (indexed from 0). This is therefore the number of bytes
          to skip from the start of the file to begin reading the meta-data.
        - The second number is the address of the first data byte. The data extends to the end of the file.
        - The third number is the actual length of the meta-data string, as it was last written.
          If this number is zero, any routines dealing with the meta-data string must determine
          the length of the string themselves.

    The messages in the data recorder's message buffer are each four bytes long. The bytes of each message are listed in
    the table below. The Channel Number is used to identify the source of the message. Channel number zero is reserved for
    clock messages. Following the channel number, each message contains a sixteen-bit data word. In the case of transmitters
    , the sixteen-bit data word is a digitized voltage. The last byte of the message is a timestamp.

    Byte	Contents
    0	Channel Number
    1	Most Significant Data Byte
    2	Least Significant Data Byte
    3	Timestamp or Version Number

    All messages with channel number zero are clock messages. This channel acts as a reference clock that
    is subsequently used to align the data messages from the transmitter channels and do error correction.
    The messages in this channel are generated at a frequency of 128 Hz.

    Each ndf file typically encodes 1 hour of data at 512 Hz, although it can also encode data at other frequencies
    (e.g. 1024 Hz) it does so for up to 14 transmitters. Each channel sends a message roughly 4 times for every
    channel 0 message (because they are operating at 512 Hz, while the clock is at 128 Hz).

    """

    def __init__(self, file_path, verbose = False, fs = 'auto', amp_factor = 200):

        self.filepath = file_path

        #  some unused
        self.tid_set = set()
        self.tid_to_fs_dict = {}
        self.tid_raw_data_time_dict  = {}
        self.tid_data_time_dict = {}
        self.resampled = False

        self.file_label = file_path.split('/')[-1].split('.')[0]
        self.identifier = None
        self.data_address = None
        self.metadata = None

        self.t_stamps = None
        self.read_ids = None
        self.fs = fs

        self._n_possible_glitches = None
        self._glitch_count        = None
        self._plot_each_glitch    = None
        self.read_id = None

        self.verbose = verbose

        self.file_time_len_sec = 3600
        self.micro_volt_div = 0.4 # this is the dac units?

        # firmware dependent:
        self.clock_tick_cycle = 7.8125e-3  # the "big" clock messages are 128Hz, 1/128 = 7.8125e-3
        self.clock_division = self.clock_tick_cycle / 256.0 # diff values from one byte

        self._read_file_metadata()
        self.get_valid_tids_and_fs()

    def __getitem__(self, item):
        #assert type(item) == int
        assert item in self.tid_set, 'ERROR: Invalid tid for file'
        return self.tid_data_time_dict[item]

    def _read_file_metadata(self):
        with open(self.filepath, 'rb') as f:

            f.seek(0)
            self.identifier = f.read(4)
            assert (self.identifier == b' ndf')

            meta_data_string_address = struct.unpack('>I', f.read(4))[0]
            self.data_address = struct.unpack('>I', f.read(4))[0]
            meta_data_length = struct.unpack('>I', f.read(4))[0]

            if meta_data_length != 0:
                f.seek(meta_data_string_address)
                self.metadata = f.read(meta_data_length)
                # need to handle the fact it is in bytes?
                #print ('\n'.join(self.metadata.split('\n')[1:-2]))
                #print (self.metadata)
            else:
                print('meta data length unknown - not bothering to work it out...')

    def get_valid_tids_and_fs(self):
        """
        - Here work out which t_ids are in the file and their
          sampling frequency. Threshold of at least 5000 datapoints!
        """
        f = open(self.filepath, 'rb')
        f.seek(self.data_address)
        self._e_bit_reads = np.fromfile(f, dtype = 'u1')
        self.transmitter_id_bytes = self._e_bit_reads[::4]
        tid_message_counts = pd.Series(self.transmitter_id_bytes).value_counts()
        possible_freqs = [256,512,1024]
        for tid, count in tid_message_counts.iteritems():
            if count > 5000 and tid != 0: # arbitrary threshold to exclude glitches
                error = [abs(3600 - count/fs) for fs in possible_freqs]
                if self.fs == 'auto':
                    self.tid_to_fs_dict[tid] = possible_freqs[np.argmin(error)]
                else:
                    self.fs = float(self.fs)
                    self.tid_to_fs_dict[tid] = self.fs
                self.tid_set.add(tid)
                self.tid_raw_data_time_dict[tid]  = {}
                self.tid_data_time_dict[tid] = {}
        logging.info(self.filepath +' valid ids and freq are: '+str(self.tid_to_fs_dict))

    #@lprofile()
    def glitch_removal(self, plot_glitches=False, print_output=False,
                       plot_sub_glitches = False, tactic = 'mad'):
        """
        Tactics can either be 'std', 'mad','roll_med', 'big_guns'
        """
        for tid in self.read_ids:
            # create binding between tid data and the data to deglitch
            self.data_to_deglitch = self.tid_data_time_dict[tid]['data']
            self.time_to_deglitch = self.tid_data_time_dict[tid]['time']
            #print (self.data_to_deglitch is self.tid_data_time_dict[tid]['data'])
            self._n_possible_glitches = 0
            self._glitch_count        = 0
            self._plot_each_glitch = plot_sub_glitches

            if plot_glitches:
                plt.figure(figsize = (15, 4))
                plt.plot(self.time_to_deglitch , self.data_to_deglitch, 'k')
                plt.title('Full raw trace');plt.xlabel('Time (seconds)')
                plt.xlim(0,self.time_to_deglitch[-1])
                plt.show()

            if tactic == 'std':
                crossing_locations = self._stddev_based_outlier()
                self._check_glitch_candidates(crossing_locations)

            elif tactic == 'mad':
                crossing_locations = np.where(self._mad_based_outlier())[0]
                self._check_glitch_candidates(crossing_locations)

            elif tactic == 'roll_med':
                crossing_locations = np.where(self._rolling_median_based_outlier())[0]
                self._check_glitch_candidates(crossing_locations)

            elif tactic == 'big_guns':
                crossing_locations = np.where(self._rolling_median_based_outlier())[0]
                self._check_glitch_candidates(crossing_locations)
                crossing_locations = np.where(self._mad_based_outlier())[0]
                self._check_glitch_candidates(crossing_locations)
                crossing_locations = self._stddev_based_outlier()
                self._check_glitch_candidates(crossing_locations)
            else:
                print ('Please specify detection tactic: ("mad","roll_med","big_guns", "std")')
                raise

            logging.debug('Tid '+str(tid)+': removed '+str(self._glitch_count)+' datapoints as glitches. There were '+str(self._n_possible_glitches)+' possible glitches.')
            if self.verbose:
                print('Tid '+str(tid)+': removed '+str(self._glitch_count)+' datapoints as glitches. There were '+str(self._n_possible_glitches)+' possible glitches.')

            if plot_glitches:
                plt.figure(figsize = (15, 4))
                plt.plot(self.time_to_deglitch , self.data_to_deglitch, 'k')
                plt.title('De-glitched trace');plt.xlabel('Time (seconds)')
                plt.xlim(0,self.time_to_deglitch[-1] )
                plt.show()

            #self.tid_data_time_dict[tid]['data'] = self.data_to_deglitch[:]
            #self.tid_data_time_dict[tid]['time'] = self.time_to_deglitch[:]

    def _mad_based_outlier(self, thresh=3.5):
        """
        From stackoverflow?

        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
        """
        points = self.data_to_deglitch
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thresh

    def _rolling_median_based_outlier(self, threshold = 1):
        data = self.data_to_deglitch
        if len(data.shape) == 1:
            data = data[:, None]
        df = pd.DataFrame(data, columns=['raw'])
        df['rolling'] = df['raw'].rolling(window=10, center =
        True).median().fillna(method='bfill').fillna(method='ffill')
        difference = np.abs(df['raw'] - df['rolling'])
        #inlier_idx = difference < threshold
        outlier_idx = difference > threshold
        n_glitch = sum(abs(outlier_idx))
        if n_glitch > 200:
            logging.warning('Warning: more than 200 glitches detected! n_glitch = '+str(n_glitch))
        return outlier_idx


    def _stddev_based_outlier(self, x_std_threshold=10):
        std_dev = np.std(self.data_to_deglitch)
        mean_point = np.mean(self.data_to_deglitch)
        threshold = std_dev * x_std_threshold + mean_point
        crossing_locations = np.where(self.data_to_deglitch > threshold)[0]
        return crossing_locations

    def _check_glitch_candidates(self,crossing_locations, diff_threshold=10,):

        self._n_possible_glitches += len(crossing_locations)
        # check local difference is much bigger than the mean difference between points
        glitch_count = 0
        std_dev = np.std(self.data_to_deglitch)
        for location in crossing_locations:
            i = location - 1
            ii = location + 1
            try:
                if abs(self.data_to_deglitch[location] - self.data_to_deglitch[ii]) > diff_threshold * std_dev:
                    # plot glitches to be removed if plotting option is on
                    if self._plot_each_glitch:
                        plt.figure(figsize = (15, 4))
                        plt.plot(self.time_to_deglitch[location - 512:location + 512],
                                 self.data_to_deglitch[location - 512:location + 512], 'k')
                        plt.ylabel('Time (s)'); plt.title('Glitch '+str(glitch_count+1))
                        plt.show()
                    try:
                        value = self.data_to_deglitch[i] + (self.time_to_deglitch[location] - self.time_to_deglitch[i]) * (
                        self.data_to_deglitch[ii] - self.data_to_deglitch[i]) / (self.data_to_deglitch[ii] - self.data_to_deglitch[i])
                        self.data_to_deglitch[location] = value
                    except IndexError:
                        pass
                    glitch_count += 1

            except IndexError:
                pass
        self._glitch_count += glitch_count


    def correct_sampling_frequency(self):
        '''
        Remeber, this is acting on the modified data (bad message and glitch already)
        so self.tid_data_time dict
        :return:
        '''
        # this occurs after bad messages, so working with data ditc
        # first check that we are not interpolating datapoints for more than 1 second?

        for tid in self.read_ids:
            max_interp = max(np.diff(self.tid_data_time_dict[tid]['time']))
            try:
                assert max_interp < 2.0
            except:
                logging.warning('WARNING: You interpolated for greater than two seconds! ('+ str('{first:.2f}'.format(first = max_interp))+' sec)')
                logging.warning('File was '+str(os.path.split(self.filepath)[1])+ ', transmitter id was '+ str(tid))

            # do linear interpolation between the points, where !nan
            regularised_time = np.linspace(0, 3600.0, num= 3600 * self.tid_to_fs_dict[tid])
            not_nan = np.logical_not(np.isnan(self.tid_data_time_dict[tid]['data']))
            self.tid_data_time_dict[tid]['data'] = np.interp(regularised_time,
                                                             self.tid_data_time_dict[tid]['time'][not_nan],
                                                             self.tid_data_time_dict[tid]['data'][not_nan])

            self.tid_data_time_dict[tid]['time'] = regularised_time

            if self.verbose:
                print('Tid '+str(tid)+': regularised fs to '+str(self.tid_to_fs_dict[tid])+' Hz '+str(self.tid_data_time_dict[tid]['data'].shape[0]) +' datapoints')

        self._resampled = True


    def save(self, save_file_name = None):
        """
        Saves file in h5 format. Will only save the tid/tids that have loaded.
        Args:
            save_file_name:
        """
        if not save_file_name:
            hdf5_filename = self.filepath.strip('.ndf')+'_Tid_'+''.join(str([tid for tid in self.read_ids]))+ '.h5'
        else:
            hdf5_filename = save_file_name + '.h5'

        with h5py.File(hdf5_filename, 'w') as f:
            f.attrs['num_channels'] = len(self.read_ids)
            f.attrs['t_ids'] = list(self.read_ids)
            f.attrs['fs_dict'] = str(self.tid_to_fs_dict)
            file_group = f.create_group(os.path.split(self.filepath)[1][:-4])

            for tid in self.read_ids:
                transmitter_group = file_group.create_group(str(tid))
                transmitter_group.attrs['fs'] = self.tid_to_fs_dict[tid]
                transmitter_group.attrs['tid'] = tid
                transmitter_group.create_dataset('data',
                                                 data=self.tid_data_time_dict[tid]['data'],
                                                 compression = "gzip", dtype='f4',
                                                 chunks = self.tid_data_time_dict[tid]['data'].shape)
                transmitter_group.create_dataset('time',
                                                 data=self.tid_data_time_dict[tid]['time'],
                                                 compression = "gzip", dtype='f4',
                                                 chunks = self.tid_data_time_dict[tid]['time'].shape)
                transmitter_group.attrs["resampled"] = self._resampled

            f.close()

            #print f.attrs['fs_dict']
        if self.verbose:
            print('Saved data as:'+str(hdf5_filename)+ ' Resampled = ' + str(self._resampled))

    def _merge_coarse_and_fine_clocks(self):
        # convert timestamps into correct time using clock id
        t_clock_data = np.zeros(self.voltage_messages.shape)
        t_clock_data[self.transmitter_id_bytes == 0] = 1 # this is big ticks
        corse_time_vector = np.cumsum(t_clock_data) * self.clock_tick_cycle
        fine_time_vector = self.t_stamps_256 * self.clock_division
        self.time_array = fine_time_vector + corse_time_vector

    def load(self, read_ids = [],
             auto_glitch_removal = True,
             auto_resampling = True,
             auto_filter = True,
             scale_and_filter = False):
        '''
        N.B. Should run glitch removal before high pass filtering and auto resampling... If unhappy with glitches,
        turn off filtering and the resampling and then run their methods etc.

        Args:
            read_ids: ids to load, can be integer of list of integers
            auto_glitch_removal: to automatically detect glitches with default tactic median abs deviation
            auto_resampling: to resample fs to regular sampling frequency
            auto_filter : high pass filter traces at default 1 hz
            scale_and_filter: high pass filter (default 1 hz) and scale to mode std dev 5 second blocks of trace
            WARNING: This is more for visualisation of what the feature extractor is working on. TO keep things
            simple, when saving HDF5 files, save non-scaled.

        Returns:
            data and time is stored in self.tid_data_time_dict attribute. Access data via obj[tid]['data'].


        '''

        self.read_ids = read_ids
        logging.info('Loading '+ self.filepath +'read ids are: '+str(self.read_ids))
        if read_ids == [] or str(read_ids).lower() == 'all':
            self.read_ids = list(self.tid_set)
        if not hasattr(self.read_ids, '__iter__'):
            self.read_ids = [read_ids]


        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grabs time stamps, then get_file props has already read these ids
        self.t_stamps_256 = self._e_bit_reads[3::4]

        # read again, but in 16 bit chunks, grab messages
        f.seek(self.data_address + 1)
        self.voltage_messages = np.fromfile(f, '>u2')[::2]

        self._merge_coarse_and_fine_clocks() # this assigns self.time_array

        for read_id in self.read_ids:
            assert read_id in self.tid_set, "Transmitter %i is not a valid transmitter id" % read_id
            self.tid_raw_data_time_dict[read_id]['data'] = self.voltage_messages[self.transmitter_id_bytes == read_id] * self.micro_volt_div
            self.tid_raw_data_time_dict[read_id]['time'] = self.time_array[self.transmitter_id_bytes == read_id]

        # remove bad messages
        self._correct_bad_messages()

        if auto_glitch_removal:
            self.glitch_removal(tactic='mad')

        if auto_resampling:
            self.correct_sampling_frequency()
            # there should now be no nans surviving here!

        if auto_filter and not scale_and_filter:
            self.highpass_filter()

        if scale_and_filter:
            self.highpass_filter()
            self.standardise_to_mode_stddev()


    def highpass_filter(self, cutoff_hz = 1):
        '''
        Implements high pass digital butterworth filter, order 2.

        Args:
            cutoff_hz: default is 1hz
        '''
        for read_id in self.read_ids:
            fs = self.tid_to_fs_dict[read_id]
            nyq = 0.5 * fs
            cutoff_decimal = cutoff_hz/nyq

            logging.debug('Highpassfiltering, tid = '+str(read_id)+' fs: ' + str(fs) + ' at '+ str(cutoff_hz)+ ' Hz')
            data = self.tid_data_time_dict[read_id]['data']
            data = data - np.mean(data)    # remove mean to try and reduce any filtering artifacts
            b, a = signal.butter(2, cutoff_decimal, 'highpass', analog=False)
            filtered_data = signal.filtfilt(b, a, data)

            self.tid_data_time_dict[read_id]['data'] = filtered_data

    def standardise_to_mode_stddev(self, stdtw = 5, std_sigfigs = 2):
        '''
        Calculates mode std dev and divides by it.

        Args:
            stdtw: time period over which to calculate std deviation
            std_sigfigs: n signfigs to round to

        '''
        for read_id in self.read_ids:
            fs = self.tid_to_fs_dict[read_id]
            data = self.tid_data_time_dict[read_id]['data']

            logging.debug('Standardising to mode std dev, tid = '+str(read_id))

            reshaped = np.reshape(data, (int(3600/stdtw), int(stdtw*fs)))
            #std_vector = self.round_to_sigfigs(np.std(reshaped, axis = 1), sigfigs=std_sigfigs)
            std_vector = np.round(np.std(data, axis = 1), 0)
            std_vector = std_vector[std_vector != 0]
            if std_vector.shape[0] > 0:
                mode_std = stats.mode(std_vector)[0] # can be zero if there is big signal loss
                scaled = np.divide(data, mode_std)
                self.tid_data_time_dict[read_id]['data'] = scaled
                logging.debug(str(mode_std)+' is mode std of trace split into '+ str(stdtw)+ ' second chunks')
            elif std_vector.shape[0] == 0:
                self.tid_data_time_dict[read_id]['data'] =  None
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

    def _correct_bad_messages(self):
        '''
        - okay so we have 128hz as the clock...
        - fs / clock_rate is the n_messages between clocks
        - 256 / n_messages is the thing we are diving to get the residuals
        '''
        for tid in self.read_ids:
            transmitter_timestamps = self.t_stamps_256[self.transmitter_id_bytes == tid]
            fs = self.tid_to_fs_dict[tid]
            n_messages = fs/128 # 128 is clock
            expected_interval = 256/n_messages # 256 is bits (if 512hz fs this is 64)
            timestamp_moduli = transmitter_timestamps % expected_interval

            # now get params for reshaping...
            n_rows = int(fs*4)
            #n_rows = 2000
            n_fullcols = int(timestamp_moduli.size//n_rows)
            n_extra_stamps = timestamp_moduli.shape[0] - (n_rows*n_fullcols)
            end_moduli = timestamp_moduli[-n_extra_stamps:]
            if n_extra_stamps:
                reshaped_moduli = np.reshape(timestamp_moduli[:-n_extra_stamps], (n_rows, n_fullcols), order = 'F')
                # order F reshaped in a "fortran manner, first axis changing fastest", calculating down the columns here
                end_mean= ss.circmean(end_moduli, high = expected_interval)
                end_moduli_corrected = (end_moduli - end_mean)
                mean_vector = ss.circmean(reshaped_moduli, high=expected_interval, axis=0)
                moduli_array_corrected = (reshaped_moduli - mean_vector)
                drift_corrected_timestamp_moduli = np.concatenate([np.ravel(moduli_array_corrected, order = 'F'), end_moduli_corrected])

            elif n_extra_stamps == 0: # can be reshaped exactly
                reshaped_moduli = np.reshape(timestamp_moduli, (n_rows, n_fullcols), order = 'F')
                mean_vector = ss.circmean(reshaped_moduli, high=expected_interval, axis=0)
                moduli_array_corrected = (reshaped_moduli - mean_vector)
                drift_corrected_timestamp_moduli = np.ravel(moduli_array_corrected, order = 'F')

            drift_corrected_timestamp_moduli = np.absolute(drift_corrected_timestamp_moduli)
            self.drift_corrected_timestamp_moduli = drift_corrected_timestamp_moduli

            bad_message_locs = np.where(np.logical_and(drift_corrected_timestamp_moduli > 9,
                                                       drift_corrected_timestamp_moduli < (expected_interval-9)))[0]

            self.tid_data_time_dict[tid]['data'] = np.delete(self.tid_raw_data_time_dict[tid]['data'], bad_message_locs)
            self.tid_data_time_dict[tid]['time'] = np.delete(self.tid_raw_data_time_dict[tid]['time'], bad_message_locs)
            logging.debug('Tid ' +str(tid)+ ': Detected '+ str(len(bad_message_locs)) + ' bad messages out of '+ str(self.tid_raw_data_time_dict[tid]['data'].shape[0])
                       + ' Remaining : '+str(self.tid_data_time_dict[tid]['data'].shape[0]))

            if len(bad_message_locs) > 0.5*self.tid_raw_data_time_dict[tid]['data'].shape[0]:
                logging.error(' >half messages detected as bad messages. Probably change fs from auto to the correct frequency')

            if self.verbose:
                print ('Tid ' +str(tid)+ ': Detected '+ str(len(bad_message_locs)) + ' bad messages out of '+ str(self.tid_raw_data_time_dict[tid]['data'].shape[0])
                       + ' Remaining : '+str(self.tid_data_time_dict[tid]['data'].shape[0]))
            #if len(bad_message_locs) > 0.5*self.tid_raw_data_time_dict[tid]['data'].shape[0]:
            #    print('WARNING: >half messages detected as bad messages. Probably change fs from auto to the correct frequency')




