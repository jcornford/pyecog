import sys
import struct
import os
import time
import logging

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

if sys.version_info < (3,):
    range = xrange
try:
    from line_profiler import LineProfiler
    # decorator for profiling methods
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
except:
    pass

class NdfFile:
    """
    TODO:
     - glitch detection is a little messy, relying on bindings.
     - bad messages doesnt delete if 2 messages in window for one (minor but would be better to delete both)
     - Find out if they ever start recording half way through - for auto fs detection
     - Code up printing the ndf object __repr__
     - Clean up unused  __init__ attributes

    Note: currently 20,000 messages are required per tid to be assigned as "valid" tid

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
    def __init__(self, file_path, verbose = False, fs = 'auto', force_one_hour_file_length = True):

        self.filepath = file_path

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

        if force_one_hour_file_length:
            self.file_time_len_sec = 3600
        else:
            self.file_time_len_sec = None

        self.micro_volt_div = 0.4 # this is the dac units

        # firmware dependent:
        self.clock_tick_cycle = 7.8125e-3  # the "big" clock messages are 128Hz, 1/128 = 7.8125e-3
        self.clock_division = self.clock_tick_cycle / 256.0 # diff values from one byte

        self._read_file_metadata()
        self.get_valid_tids_and_fs()

    def __getitem__(self, item):
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
            else:
                print('meta data length unknown - not bothering to work it out...')
                # isn't it just (data_address - meta_data_string_address) ?

    def get_valid_tids_and_fs(self):
        """
        - Here work out which t_ids are in the file and their
          sampling frequency. Arbitary threshold of at least 5000 datapoints!
        """
        f = open(self.filepath, 'rb')
        f.seek(self.data_address)
        self._e_bit_reads = np.fromfile(f, dtype = 'u1')
        self.transmitter_id_bytes = self._e_bit_reads[::4]
        tid_message_counts = pd.Series(self.transmitter_id_bytes).value_counts()  # count how many different ids exist
        possible_freqs = [256,512,1024]
        for tid, count in tid_message_counts.iteritems():
            if count > 20000 and tid != 0: # arbitrary number of messages threshold to exclude glitch tids
                error = [abs(3600 - count/fs) for fs in possible_freqs]
                if self.fs == 'auto':
                    self.tid_to_fs_dict[tid] = possible_freqs[np.argmin(error)]
                else:
                    self.fs = float(self.fs)
                    self.tid_to_fs_dict[tid] = self.fs
                self.tid_set.add(tid)
                self.tid_raw_data_time_dict[tid]  = {}
                self.tid_data_time_dict[tid] = {}
        logging.info(self.filepath)
        logging.info('Valid ids and freq are: '+str(self.tid_to_fs_dict))

    #@lprofile()
    def glitch_removal(self, plot_glitches=False, print_output=False, plot_sub_glitches = False):
        """
        The idea is to identify large transients in the data
        #todo: eliminate the usage of bindings...
        """
        for tid in self.read_ids:
            # create binding between tid data and the data to deglitch
            self.data_to_deglitch = self.tid_data_time_dict[tid]['data']
            self.std_data_to_deglitch = np.std(self.data_to_deglitch)
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

            n_passes = 5
            passn = 1
            max_glitches = int(.00002*len(self.data_to_deglitch)) # define artbitary number of glitches to be happy with
            self.n_glitches = max_glitches
            while self.n_glitches >= max_glitches and passn <= n_passes: # if too many glitches are found: repeat
                crossing_locations = np.where(self._diff_based_outlier())[0]
                self._check_glitch_candidates(crossing_locations) # this updates self.n_gltiches
                if self.verbose:
                    print('Tid '+str(tid)+' pass ' + str(passn) + ': glitches found - ' + str(self.n_glitches) +
                          ' (max:' + str(max_glitches) + ')')
                    passn += 1
                    if plot_glitches:
                        plt.figure(figsize=(15, 4))
                        plt.plot(self.time_to_deglitch, self.data_to_deglitch, 'k')
                        plt.title('De-glitched trace')
                        plt.xlabel('Time (seconds)')
                        plt.xlim(0, self.time_to_deglitch[-1])
                        plt.show()

            logging.debug('Tid '+str(tid)+': removed '+str(self._glitch_count)+' datapoints as glitches. There were '
                          +str(self._n_possible_glitches)+' possible glitches.')
            if self.verbose:
                print('Tid '+str(tid)+': removed '+str(self._glitch_count)+' datapoints as glitches. There were '
                      +str(self._n_possible_glitches)+' possible glitches.')

            #self.tid_data_time_dict[tid]['data'] = self.data_to_deglitch[:]
            #self.tid_data_time_dict[tid]['time'] = self.time_to_deglitch[:]

    def _diff_based_outlier(self, thresh=.0000001):
        """
        Compute the absolute first difference of the signal and check for outliers
        """
        abs_diff_array = np.abs(np.diff(self.data_to_deglitch))
        if len(abs_diff_array.shape) == 1:
            abs_diff_array = abs_diff_array[:, None]
        meandiff = np.mean(abs_diff_array, axis=0)
        self.stddiff_data_to_deglitch = np.mean(abs_diff_array, axis=0)
        outliers = (abs_diff_array > -2 * np.log(thresh) * meandiff) + (abs_diff_array==0)
        return outliers

    #@lprofile()
    def _check_glitch_candidates(self,crossing_locations, diff_threshold=9,):
        '''
        Checks local difference is much bigger than the mean difference between points
        Args:
            crossing_locations: indexes of glitch
            diff_threshold: default is 10, uV?

        Returns:
        '''
        self._n_possible_glitches += len(crossing_locations)
        glitch_count = 0

        for i1 in crossing_locations:

            try:
                # first check if the next data point has a large transient or is flat lined
                cond1 = int(abs(self.data_to_deglitch[i1] - self.data_to_deglitch[i1+1]) > diff_threshold * self.stddiff_data_to_deglitch)
                cond2 = int(abs(self.data_to_deglitch[i1] - self.data_to_deglitch[i1-1]) > diff_threshold * self.stddiff_data_to_deglitch)
                cond3 = int(abs(self.data_to_deglitch[i1] - self.data_to_deglitch[i1+1]) == 0)
                cond4 = int(abs(self.data_to_deglitch[i1] - self.data_to_deglitch[i1-1]) == 0)

                if (cond1 + cond2 + cond3 + cond4)>=2 and not cond3+cond4 == 2:
                    i = i1 - 1
                    while i in crossing_locations:
                        i = i-1
                    # Correct for the fact that crossing_locations contain raise and peak times of the typical glitch
                    i = i+1

                    ii = i1+1
                    while ii in crossing_locations:
                        ii = ii + 1

                    # Correct only if surounding data points look normal
                    if abs(self.data_to_deglitch[i] - self.data_to_deglitch[ii])< 2*diff_threshold * self.stddiff_data_to_deglitch:

                        # plot glitches to be removed if plotting option is on
                        if self._plot_each_glitch:
                            plt.figure(figsize = (15, 4))
                            ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=3)
                            ax1.plot(self.time_to_deglitch[i:ii+1],
                                     self.data_to_deglitch[i:ii+1], 'r.-', zorder = 2)
                            ax1.set_xlabel('Time (s)'); ax1.set_title('Glitch '+str(glitch_count+1))

                        try:
                            removed = 'False'
                            if ii-i>1:  # ensure there are 3 points in the glitch, eliminates asymmetrical false positives
                                self.data_to_deglitch[i1] = (self.data_to_deglitch[i] + self.data_to_deglitch[ii])/2
                                glitch_count += 1
                                removed = 'True'

                            if self._plot_each_glitch:
                                ax1.plot(self.time_to_deglitch[i1 - 64:i1 + 64],
                                         self.data_to_deglitch[i1 - 64:i1 + 64], 'k-', zorder = 1, label = 'Glitch removed :'+removed)
                                ax1.legend(loc = 1)
                        except IndexError:
                            print('IndexError')
                            pass
                        if self._plot_each_glitch:
                            plt.show()


            except IndexError:
                print('IndexError')
                pass
        self.n_glitches     = glitch_count
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
            diff_array = np.diff(self.tid_data_time_dict[tid]['time'])
            second_gaps = diff_array[diff_array>1]
            seconds_missing = np.sum(second_gaps)
            n_second_gaps   = second_gaps.shape[0]
            if n_second_gaps >0:
                logging.debug('Tid '+str(tid)+': File contained '+ str(n_second_gaps)+
                              ' gaps of >1 second. Total Missing: '+str(seconds_missing)+ ' s.')
            try:
                assert max_interp < 2.0
            except:
                logging.warning(str(os.path.split(self.filepath)[1])+ ' Tid '+str(tid)+
                                ': You interpolated (at least once) for greater than two seconds! ('+
                                str('{first:.2f}'.format(first = max_interp))+' sec)')

            # do linear interpolation between the points, where !nan
            regularised_time = np.linspace(0, 3600.0, num= 3600 * self.tid_to_fs_dict[tid])
            not_nan = np.logical_not(np.isnan(self.tid_data_time_dict[tid]['data']))
            self.tid_data_time_dict[tid]['data'] = np.interp(regularised_time,
                                                             self.tid_data_time_dict[tid]['time'][not_nan],
                                                             self.tid_data_time_dict[tid]['data'][not_nan],)

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
                if self._resampled:
                    transmitter_group.attrs["time_arr_info_dict"] = str({'max_t':max(self.tid_data_time_dict[tid]['time']),
                                                                     'min_t':min(self.tid_data_time_dict[tid]['time']),
                                                                     'fs':   self.tid_to_fs_dict[tid]})
                else:
                    print('saving time array') # this shouldnt normally be run
                    transmitter_group.create_dataset('time',
                                                 data=self.tid_data_time_dict[tid]['time'],
                                                 compression = "gzip", dtype='f4',
                                                 chunks = self.tid_data_time_dict[tid]['time'].shape)

                transmitter_group.attrs["resampled"] = self._resampled

            f.close()

        if self.verbose:
            print('Saved data as:'+str(hdf5_filename)+ ' Resampled = ' + str(self._resampled))

    #@lprofile()
    def merge_coarse_and_fine_clocks(self):
        # would this be more effcient if just using lower precision?
        # convert timestamps into correct time using clock id
        t_clock_data = np.zeros(self.voltage_messages.shape, dtype='float32')
        t_clock_indices = np.where(self.transmitter_id_bytes == 0)[0]
        t_clock_data[t_clock_indices] = 1 # this is big ticks
        coarse_time_vector = np.multiply(np.cumsum(t_clock_data), self.clock_tick_cycle)
        fine_time_vector   = np.multiply(self.t_stamps_256, self.clock_division)
        self.time_array    = np.add(fine_time_vector,coarse_time_vector)

        # here account for occasions when transmitter with reset clock comes before clock 0 message
        try:
            clock_tstamp = self.t_stamps_256[t_clock_indices[0]] # clock timestamp is the same throughout
        except IndexError:
            print('ERROR: No clock messages!')
            return 1
        bad_timing = np.where(np.diff(self.time_array) == (256 + clock_tstamp) * 1 / 128 / 256)[0]  # this might suffer from numerical precision problems...
        self.time_array[bad_timing] = self.time_array[bad_timing] + 1 / 128
        return 0
        


    #@lprofile()
    def load(self, read_ids = [],
             auto_glitch_removal = True,
             auto_resampling = True,
             auto_filter = True,
             scale_to_mode_std = False):
        '''
        Notes:
            1. This assumes your file is 3600 seconds long. If it is not, it will append 0's to the end of your data.
            2. You should run glitch removal before high pass filtering and the auto resampling.
               If unhappy with glitches, you should turn off filtering and the resampling before
               running the glitch methods individually.

        Args:
            read_ids: ids to load, can be integer of list of integers
            auto_glitch_removal: to automatically detect glitches with default tactic median abs deviation
            auto_resampling: to resample fs to regular sampling frequency
            auto_filter : high pass filter traces at default 1 hz
            scale_to_mode_std: high pass filter (default 1 hz) and scale to mode std dev 5 second blocks of trace
            WARNING: This is more for visualisation of what the feature extractor is working on. TO keep things
            simple, when saving HDF5 files, save non-scaled.

        Returns:
            data and time is stored in self.tid_data_time_dict attribute. Access data via obj[tid]['data'].

        '''
        self.read_ids = read_ids
        logging.info('Loading '+ self.filepath +'read ids are: '+str(self.read_ids))
        if read_ids == [] or str(read_ids).lower() == 'all':
            self.read_ids = sorted(list(self.tid_set))
        if not hasattr(self.read_ids, '__iter__'):
            self.read_ids = [read_ids]

        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grabs time stamps, get_valid_tids_and_fs called on init has already has ids
        self.t_stamps_256 = self._e_bit_reads[3::4]

        # read again, but in 16 bit chunks, grab messages
        # f.seek(self.data_address + 1)
        # self.voltage_messages = np.fromfile(f, '>u2')[::2] #This seems sub-optimal: reading twice from disk
        self.voltage_messages = np.frombuffer(self._e_bit_reads[1:-1:].tobytes(), dtype='>u2')[::2]

        error_flag = self.merge_coarse_and_fine_clocks()  # this generates self.time_array
        if error_flag:
            return 1

        invalid_ids = []
        for read_id in self.read_ids:
            if read_id not in self.tid_set:
                invalid_ids.append(read_id)
            else:
                self.tid_raw_data_time_dict[read_id]['data'] = self.voltage_messages[self.transmitter_id_bytes == read_id] * self.micro_volt_div
                self.tid_raw_data_time_dict[read_id]['time'] = self.time_array[self.transmitter_id_bytes == read_id]

        for invalid_id in invalid_ids: self.read_ids.remove(invalid_id)
        if len(invalid_ids) != 0: print('Error: Invalid transmitter/s id passed to file: '+ str(invalid_ids))

        self.correct_bad_messages()

        if auto_glitch_removal:
            self.glitch_removal()

        if auto_resampling:
            self.correct_sampling_frequency()
            # there should now be no nans surviving here!

        if auto_filter:
            self.highpass_filter()

        if scale_to_mode_std: # not normally called
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

            logging.info('Highpassfiltering, tid = '+str(read_id)+' fs: ' + str(fs) + ' at '+ str(cutoff_hz)+ ' Hz')
            data = self.tid_data_time_dict[read_id]['data']
            data = data - np.mean(data)    # remove mean to try and reduce any filtering artifacts
            b, a = signal.butter(2, cutoff_decimal, 'highpass', analog=False)
            filtered_data = signal.filtfilt(b, a, data)

            self.tid_data_time_dict[read_id]['data'] = filtered_data

    def standardise_to_mode_stddev(self, stdtw = 5, std_sigfigs = 2):
        '''
        Calculates mode std dev and divides by it. This is not normally ran - only when dong the feature detection
        in feature extractor

        Args:
            stdtw: time period over which to calculate std deviation
            std_sigfigs: n signfigs to round to

        '''
        for read_id in self.read_ids:
            fs = self.tid_to_fs_dict[read_id]
            data = self.tid_data_time_dict[read_id]['data']

            logging.info('Standardising to mode std dev, tid = '+str(read_id))

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

    #@lprofile()
    def correct_bad_messages(self): #new
        '''
        Look for short inter-message-intervals to identify bad messages
        '''

        for tid in self.read_ids:
            # time diference between samples
            time_256  = self.t_stamps_256[self.transmitter_id_bytes == tid]

            tdiff = np.diff(time_256) % 256

            fs = self.tid_to_fs_dict[tid]
            sampling_period_bits = 256*128/fs # nbits*clock_frequency/fs_Hz = sampling_period_bits
            threshold = 9

            bad_message_index = np.nonzero(tdiff <= sampling_period_bits - threshold)[0][:, None]
            # make sure we won't have problems indexing stuff later:
            #bad_message_index[0]  = np.maximum(bad_message_index[0],1)
            #bad_message_index[-1] = np.minimum(bad_message_index[-1], len(tdiff)-2)

            bad_message_indices = np.hstack((bad_message_index, bad_message_index+1))
            base_message_indices = np.hstack((bad_message_index-1, bad_message_index-1))
            displace = np.argmax(np.abs(self.tid_raw_data_time_dict[tid]['data'][bad_message_indices] -
                                        self.tid_raw_data_time_dict[tid]['data'][base_message_indices]), axis=1)

            bad_message_locs = (bad_message_index + displace[:,None])

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
