import sys
import struct
import os
import time

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from line_profiler import LineProfiler


if sys.version_info < (3,):
    range = xrange

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

class NDFLoader:
    """
    TODO:
     - FILTERING!
     - Bad message seems unfinished (the theory at least)
     - Code up indexing the ndf instead of using load
     - Potentially need to inherit from class that can also be used for h5 loading (also .plot()?)

     - Glitch detection not working as expected with combined?
     - Speed up: Glitches(settle on tactic) / saving
     - Can you compress the saved h5 further?

     - Find out if they ever start recording half way through
     - Code up priting tthe ndf object __repr__

     - Write to log...
     - Clean up the __init__ attributes

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

    def __init__(self, file_path):

        self.filepath = file_path

        #  some unused

        self.tid_set = set()
        self.tid_to_fs_dict = {}
        self.tid_raw_data_dict  = {}
        self.tid_raw_time_dict  = {}
        self.tid_data_dict = {} # these get raw
        self.tid_time_dict = {}
        self.resampled = False

        self.file_label = file_path.split('/')[-1].split('.')[0]
        self.mean_point = None
        self.identifier = None
        self.data_address = None
        self.metadata = None

        self.channel_info = None

        self.tids = None
        self.t_stamps = None
        self.read_ids = None
        self.time_diff = None


        self._n_possible_glitches = None
        self._glitch_count        = None
        self._plot_each_glitch    = None
        self.read_id = None

        self.verbose = True

        self.file_time_len_sec = 3600
        adc_range = 2.7
        amp_factor = 300
        bit_size = 16
        self.volt_div = adc_range / (2 ** bit_size) / amp_factor * 1e3  # in mV unit

        # firmware dependent:
        self.clock_tick_cycle = 7.8125e-3  # the "big" clock messages are 128Hz, 1/128 = 7.8125e-3
        self.clock_division = self.clock_tick_cycle / 256.0 # diff values from one byte

        self._read_file_metadata()
        self._get_valid_tids_and_fs()

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
                #print ('\n'.join(self.metadata.split('\n')[1:-2]))

            else:
                print('meta data length unknown - not bothering to work it out...')

    def _get_valid_tids_and_fs(self):
        """
        - Here work out which t_ids are in the file and their
          sampling frequency. Threshold of at least 5000 datapoints!
        """
        f = open(self.filepath, 'rb')
        f.seek(self.data_address)
        self._e_bit_reads = np.fromfile(f, dtype = 'u1')
        self._transmitter_id_bytes = self._e_bit_reads[::4]
        tid_message_counts = pd.Series(self._transmitter_id_bytes).value_counts()
        possible_freqs = [256,512,1024]
        for tid, count in tid_message_counts.iteritems():
            if count > 5000 and tid != 0: # arbitrary threshold to exclude glitches
                error = [abs(3600 - count/fs) for fs in possible_freqs]
                self.tid_to_fs_dict[tid] = possible_freqs[np.argmin(error)]
                self.tid_set.add(tid)
    #@lprofile()
    def glitch_removal(self, plot_glitches=False, print_output=False,
                       plot_sub_glitches = False, tactic = 'mad'):
        """
        Tactics can either be 'std', 'mad','roll_med', 'big_guns'
        """
        for tid in self.tid_data_dict.keys():
            # use the badmessage filtered - also, if called first, will be resampled
            self.data_to_deglitch = self.tid_data_dict[tid]
            self.time_to_deglitch = self.tid_time_dict[tid]
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
                print 'hit1'
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

            if self.verbose:
                print('Tid '+str(tid)+': removed '+str(self._glitch_count)+' datapoints as glitches. There were '+str(self._n_possible_glitches)+' possible glitches.')

            if plot_glitches:
                plt.figure(figsize = (15, 4))
                plt.plot(self.raw_time, self.raw_data, 'k')
                plt.title('De-glitched trace');plt.xlabel('Time (seconds)')
                plt.xlim(0,self.raw_time[-1])
                plt.show()
        self.raw_data, self.raw_time = None, None



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
        inlier_idx = difference < threshold
        outlier_idx = difference > threshold
        n_glitch = sum(abs(outlier_idx))
        if n_glitch > 200:
            print ('Warning: more than 200 glitches detected! n_glitch = '+str(n_glitch))
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
        # this occurs after bad messages, so working with data ditc
        # first check that we are not interpolating datapoints for more than 1 second?

        for tid in self.tid_data_dict.keys():

            try:
                assert max(np.diff(self.tid_raw_time_dict[tid])) < 1.0
            except:
                print  ('oh fucky, you interpolated for greater than one second!')

            # do linear interpolation between the points
            regularised_time = np.linspace(0, 3600.0, num= 3600 * self.tid_to_fs_dict[tid])
            self.tid_data_dict[tid] = np.interp(regularised_time, self.tid_raw_time_dict[tid],
                                                self.tid_raw_data_dict[tid])
            self.tid_time_dict[tid] = regularised_time
            if self.verbose:
                print('Tid '+str(tid)+': corrected fs to '+str(self.tid_to_fs_dict[tid])+' Hz')

        self._resampled = True


    def save(self, save_file_name = None):
        """
        Saves file in h5 format. Will only save the tid that have loaded.
        Args:
            save_file_name:
        """
        if not save_file_name:
            hdf5_filename = self.filepath.strip('.ndf')+'_Tid_'+''.join(str([tid for tid in self.tid_data_dict.keys()]))+ '.h5'
        else:
            hdf5_filename = save_file_name + '.h5'
        hdf5_data = {}
        hdf5_time = {}
        with h5py.File(hdf5_filename, 'w') as f:
            f.attrs['num_channels'] = len(self.tid_set)
            f.attrs['t_ids'] = list(self.tid_set)
            f.attrs['fs_dict'] = str(self.tid_to_fs_dict)
            file_group = f.create_group(self.filepath.split('/')[-1][:-4])
            for tid in self.tid_data_dict.keys():
                transmitter_group = file_group.create_group(str(tid))

                data_to_save = self.tid_data_dict[tid] # bad messaged are here no longer distinction with
                time_to_save = self.tid_time_dict[tid] # resampling the data or not
                hdf5_data[tid] = transmitter_group.create_dataset('data', data=data_to_save, compression="gzip")
                hdf5_time[tid] = transmitter_group.create_dataset('time', data=time_to_save, compression="gzip")
                transmitter_group.attrs["resampled"] = self._resampled
            #print f.attrs['fs_dict']

        print('Saved data as:'+str(hdf5_filename)+ ' Resampled = ' + str(self._resampled))

    def _merge_coarse_and_fine_clocks(self):
        # convert timestamps into correct time using clock id
        t_clock_data = np.zeros(self.voltage_messages.shape)
        t_clock_data[self._transmitter_id_bytes == 0] = 1 # this is big ticks
        corse_time_vector = np.cumsum(t_clock_data) * self.clock_tick_cycle
        fine_time_vector = self.t_stamps_256 * self.clock_division
        self.time_array = fine_time_vector + corse_time_vector

    def load(self, read_ids = [], auto_glitch_removal = False):
        self.read_ids = read_ids
        if read_ids == [] or str(read_ids).lower() == 'all':
            self.read_ids = list(self.tid_set)
        if not hasattr(self.read_ids, '__iter__'):
            self.read_ids = [read_ids]

        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grabs time stamps
        # the get_file props has already read these ids
        transmitter_ids = self._transmitter_id_bytes
        self.t_stamps_256 = self._e_bit_reads[3::4]

        # read again, but in 16 bit chunks, grab messages
        f.seek(self.data_address + 1)
        self.voltage_messages = np.fromfile(f, '>u2')[::2]

        self._merge_coarse_and_fine_clocks() # this assigns self.time_array

        for read_id in self.read_ids:
            assert read_id in self.tid_set, "Transmitter %i is not a valid transmitter id" % read_id
            self.tid_raw_data_dict[read_id] = self.voltage_messages[self._transmitter_id_bytes == read_id] * self.volt_div
            self.tid_raw_time_dict[read_id] = self.time_array[self._transmitter_id_bytes == read_id]
        self._correct_bad_messages()

        if auto_glitch_removal:
            self.glitch_removal(tactic='mad')

    def _correct_bad_messages(self):
        '''
        Now Vectorised!
        '''
        for tid in self.tid_raw_data_dict.keys():

            transmitter_timestamps = self.t_stamps_256[self._transmitter_id_bytes == tid]
            timestamp_residuals = transmitter_timestamps%64 # as 60% 64 = 60, need to
            # use double ended threshold later! (>9 and <51)
            # actually is this the same thing?
            fs = self.tid_to_fs_dict[tid]
            n_rows = int(fs*4)
            n_fullcols = int(timestamp_residuals.size//n_rows)
            n_extra_stamps = timestamp_residuals.shape[0] - (n_rows*n_fullcols)
            end_residuals = timestamp_residuals[-n_extra_stamps:]
            reshaped_residuals = np.reshape(timestamp_residuals[:-n_extra_stamps], (n_rows, n_fullcols), order = 'F')
            # order F reshaped in a "fortran manner, first axis changing fastest"

            end_mode = stats.mode(end_residuals)[0][0]
            offset_end = (end_residuals - end_mode)%64

            mode_vector = stats.mode(reshaped_residuals, axis = 0)[0][0]
            offset_array = (reshaped_residuals - mode_vector )%64
            flattened_offset = np.concatenate([np.ravel(offset_array, order = 'F'), offset_end])
            # using 51 from Ali, but should be 54 actually
            bad_message_locs = np.where(np.logical_and(flattened_offset > 9,flattened_offset < 51 ))[0]
            self.tid_data_dict[tid] = np.delete(self.tid_raw_data_dict[tid], bad_message_locs)
            self.tid_time_dict[tid] = np.delete(self.tid_raw_time_dict[tid], bad_message_locs)
            if self.verbose:
                print ('Tid ' +str(tid)+ ': Detected '+ str(len(bad_message_locs)) + ' bad messages out of '+ str(self.tid_raw_data_dict[tid].shape[0]))

#start = time.time()
#fdir = '/Users/Jonathan/Dropbox/DataSharing_GL_SJ/'
#ndf = NDFLoader(fdir+'M1457172030.ndf')
#ndf.load('all')
#ndf.glitch_removal(tactic = 'mad')
#ndf.correct_sampling_frequency()
#print time.time() - start
#ndf.save()
#print time.time() - start