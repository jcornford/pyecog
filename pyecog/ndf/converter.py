import sys
import struct
import os
import time

import pandas as pd
import h5py

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

    def __init__(self, file_path, fs,
                 print_meta=False, auto_load = False):

        self.print_meta = print_meta
        self.filepath = file_path

        #  some unused
         
        self.tid_set = set()
        self.tid_to_fs_dict = {}
        self.file_label = file_path.split('/')[-1].split('.')[0]
        self.mean_point = None
        self.identifier = None
        self.data_address = None
        self.metadata = None

        self.channel_info = None
        self.data_dict = None
        self.tids = None
        self.t_stamps = None
        self.data = None
        self.time = None
        self.time_diff = None
        self.time_resampled = None
        self.data_resampled = None
        self.resampled = None
        self.resampled_fs = None
        self.read_id = None
        self.verbose = True
        self.auto_load = auto_load

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
                if self.print_meta:
                    print ('\n'.join(self.metadata.split('\n')[1:-2]))

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
        self.transmitter_id_bytes = self._e_bit_reads[::4]
        tid_message_counts = pd.Series(self.transmitter_id_bytes).value_counts()
        possible_freqs = [256,512,1024]
        for tid, count in tid_message_counts.iteritems():
            if count > 5000 and tid != 0: # arbitrary threshold to exclude glitches
                error = [abs(3600 - count/fs) for fs in possible_freqs]
                self.tid_to_fs_dict[tid] = possible_freqs[np.argmin(error)]
                self.tid_set.add(tid)

    def _mad_based_outlier(self, thresh=3.5):
        """
        From stackoverflow?

        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
        """
        points = self.raw_data
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def _rolling_median(self, threshold = 1, plot_glitches=False, print_output=False):
        if len(self.raw_data.shape) == 1:
            self.data_4med = self.raw_data[:, None]
        df = pd.DataFrame(self.data_4med, columns=['raw'])
        df['rolling'] = df['raw'].rolling(window=10, center =
        True).median().fillna(method='bfill').fillna(method='ffill')
        difference = np.abs(df['raw'] - df['rolling'])
        inlier_idx = difference < threshold
        outlier_idx = difference > threshold
        n_glitch = sum(abs(outlier_idx))
        if n_glitch > 200:
            print ('Warning: more than 200 glitches detected! n_glitch = '+str(n_glitch))
        return outlier_idx

    def _check_glitch_candidates(self,diff_threshold=10,):
        std_dev = np.std(self.raw_data)
        self.n_possible_glitches += len(self.crossing_locations)
        # check local difference is much bigger than the mean difference between points
        glitch_count = 0
        for location in self.crossing_locations: # std threshold crossings
            if location == 0:
                pass
                #print('Warning: if two glitches at start, correction will fail')
            i = location - 1
            ii = location + 1
            try:
                if abs(self.raw_data[location] - self.raw_data[ii]) > diff_threshold * std_dev:
                    # plot glitches to be removed if plotting option is on
                    if self.plot_each_glitch:
                        plt.figure(figsize = (15, 4))
                        plt.plot(self.raw_time[location - 512:location + 512],
                                 self.raw_data[location - 512:location + 512], 'k')
                        plt.ylabel('Time (s)'); plt.title('Glitch '+str(glitch_count+1))
                        plt.show()
                    try:
                        value = self.raw_data[i] + (self.raw_time[location] - self.raw_time[i]) * (
                        self.raw_data[ii] - self.raw_data[i]) / (self.raw_time[ii] - self.raw_time[i])
                        self.raw_data[location] = value
                    except IndexError:
                        pass
                    glitch_count += 1

            except IndexError:
                pass
        self.glitch_count += glitch_count


    def glitch_removal(self, x_std_threshold=10, plot_glitches=False, print_output=False,
                       plot_sub_glitches = False, tactic = 'mad'):
        """
        Seperate method, in order to adjust the thresholds manually

        Tactics can either be 'old', 'mad','roll_med', 'big_guns'
        """
        std_dev = np.std(self.raw_data)
        self.n_possible_glitches = 0
        self.glitch_count        = 0
        self.plot_each_glitch = plot_sub_glitches
        if plot_glitches:
            plt.figure(figsize = (15, 4))
            plt.plot(self.raw_time, self.raw_data, 'k')
            plt.title('Full raw trace');plt.xlabel('Time (seconds)')
            plt.xlim(0,self.raw_time[-1])
            plt.show()

        if tactic == 'old':
            mean_diff = np.mean(abs(np.diff(self.raw_data)))
            if not self.mean_point:
                self.mean_point = np.mean(self.raw_data)
            # identify candidate glitches based on std deviation
            threshold = std_dev * x_std_threshold + self.mean_point
            crossing_locations = np.where(self.data > threshold)[0]
            self.crossing_locations = crossing_locations
            self._check_glitch_candidates()


        elif tactic == 'mad':
            crossing_locations = np.where(self._mad_based_outlier())[0]
            self.crossing_locations = crossing_locations
            self._check_glitch_candidates()

        elif tactic == 'roll_med':
            crossing_locations = np.where(self._rolling_median())[0]
            self.crossing_locations = crossing_locations
            self._check_glitch_candidates()

        elif tactic == 'big_guns':
            crossing_locations = np.where(self._rolling_median())[0]
            self.crossing_locations = crossing_locations
            self._check_glitch_candidates()

            crossing_locations = np.where(self._mad_based_outlier())[0]
            self.crossing_locations = crossing_locations
            self._check_glitch_candidates()
            mean_diff = np.mean(abs(np.diff(self.raw_data)))
            if not self.mean_point:
                self.mean_point = np.mean(self.raw_data)
            # identify candidate glitches based on std deviation
            threshold = std_dev * x_std_threshold + self.mean_point
            crossing_locations = np.where(self.data > threshold)[0]
            self.crossing_locations = crossing_locations
            self._check_glitch_candidates()
        else:
            print ('Please specify detection tactic: ("mad","roll_med","big_guns", "old")')
            raise

        if print_output:
            print('Removed', self.glitch_count, 'datapoints detected as glitches. There were',self.n_possible_glitches, 'possible glitches.')

        if plot_glitches:
            plt.figure(figsize = (15, 4))
            plt.plot(self.raw_time, self.raw_data, 'k')
            plt.title('De-glitched trace');plt.xlabel('Time (seconds)')
            plt.xlim(0,self.raw_time[-1])
            plt.show()
        self.data = np.ravel(self.raw_data)




    def correct_sampling_frequency(self, resampling_fs=512.0, overwrite=False):
        # first check that we are not interpolating datapoints for more than 1 second?
        print 'Warning, change the sampling frequency to be automatic!'
        try:
            assert max(np.diff(self.raw_time)) < 1.0
        except:
            print  ('oh fucky, you interpolated for greater than one second!')
        self.time_diff = np.diff(self.raw_time)

        # do linear interpolation between the points
        print self.raw_time[-1]
        self.time_resampled = np.linspace(0, 3600.0, num= 3600 * resampling_fs)
        self.data_resampled = np.interp(self.time_resampled, self.raw_time, self.raw_data)
        self.resampled = True

        if overwrite:
            self.time = self.time_resampled[:]
            self.data = self.time_resampled[:]
            print('overwrite')

    def save(self, save_file_name = None, file_format='hdf5', sec_per_row=1, minimum_seconds=1):
        """
        Info on channels and their recording length can be found in the channel_info attribute.

        Currently accepting the following savefile format options:
            - hdf5

        Strongly recommended to save in hdf5 file format!

        Args:
            save_file_name:
            file_format:

        """
        #self.resampled_fs
        if file_format == 'hdf5':
            if not save_file_name:
                hdf5_filename = self.filepath.strip('.ndf')+'_Tid_'+str(self.read_id) + '.' + file_format
            else:
                hdf5_filename = save_file_name + '.' + file_format
            hdf5_data = {}
            hdf5_time = {}
            with h5py.File(hdf5_filename, 'w') as f:
                f.attrs['num_channels'] = len(self.data)
                file_group = f.create_group(self.filepath.split('/')[-1][:-4])

                transmitter_group = file_group.create_group(str(self.read_id))

                data_to_save = self.data_resampled if self.resampled else self.data
                time_to_save = self.time_resampled if self.resampled else self.time
                hdf5_data = transmitter_group.create_dataset('data', data=data_to_save, compression="gzip")
                hdf5_time = transmitter_group.create_dataset('time', data=time_to_save, compression="gzip")
                transmitter_group.attrs["resampled"] = self.resampled

        print('Saved data as:'+str(hdf5_filename))

    def _merge_coarse_and_fine_clocks(self):
        # convert timestamps into correct time using clock id
        t_clock_data = np.zeros(self.messages.shape)
        t_clock_data[self.transmitter_id_bytes == 0] = 1 # this is big ticks
        corse_time_vector = np.cumsum(t_clock_data) * self.clock_tick_cycle
        fine_time_vector = self.t_stamps_256 * self.clock_division
        self.time_array = fine_time_vector + corse_time_vector

    def load(self, read_id, auto_glitch_removal = False):
        self.read_id = read_id # do you need this?

        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grabs time stamps
        # the get_file props has already read these ids
        transmitter_ids = self.transmitter_id_bytes
        self.t_stamps_256 = self._e_bit_reads[3::4]

        # read again, but in 16 bit chunks, grab messages
        f.seek(self.data_address + 1)
        self.messages = np.fromfile(f, '>u2')[::2]

        self._merge_coarse_and_fine_clocks() # this assigns self.time_array

        self.raw_data = self.messages[self.transmitter_id_bytes == read_id] * self.volt_div
        self.raw_time = self.time_array[self.transmitter_id_bytes == read_id]
        self._correct_bad_messages()

    def _correct_bad_messages(self):
        '''
        Vectorised
        '''
        transmitter_timestamps = self.t_stamps_256[self.transmitter_id_bytes==self.read_id]
        timestamp_residuals = transmitter_timestamps%64 # as 60% 64 = 60, need to
        # use double ended threshold later! (>9 and <51)
        # actually is this the same thing?

        n_rows = int(self.fs*4)
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
        print self.raw_data.shape
        self.raw_data = np.delete(self.raw_data, bad_message_locs)
        self.raw_time = np.delete(self.raw_time, bad_message_locs)
        if self.verbose:
            print ('Detected '+ str(len(bad_message_locs)) + ' bad messages out of '+ str(self.raw_data.shape[0]))

fdir = '/Users/Jonathan/Dropbox/DataSharing_GL_SJ/'
ndf = NDFLoader(fdir+'M1457172030.ndf', 256.0, print_meta=True)
