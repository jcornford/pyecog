import sys
import struct
import os

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if sys.version_info < (3,):
    range = xrange

class NDFLoader:
    """
    Class to load ndf binary files.

    From open source instruments website:
    The NDF format is designed to make it easy for us to add one-dimensional data to an existing file. The letters NDF
    stand for Neuroscience Data Format. The NDF file starts with a header of at least twelve bytes. The first four bytes
    spell the NDF identifier " ndf".

    Three four-byte numbers follow the identifier. All are big-endian (most significant byte first). The first number is the
    address of the meta-data string. By address we mean the byte offset from the first byte of the file, so the first byte
    has address zero and the tenth byte has address nine. Thus the address of the meta-data string is the number of bytes
    we skip from the start of the file to get to the first character of the meta-data string. This address must be at least
    16 to avoid the header. The meta-data string is a null-terminated ASCII character string. The second number is the
    address of the first data byte. The data extends to the end of the file. To determine the length of the data, we obtain
     the length of the file from the local operating system and subtract the data address. The third number is the actual
     length of the meta-data string, as it was last written. If this number is zero, any routines dealing with the
     meta-data string must determine the length of the string themselves.

    The messages in the data recorder's message buffer are each four bytes long. The bytes of each message are listed in
    the table below. The Channel Number is used to identify the source of the message. Channel number zero is reserved for
    clock messages. In the case of Subcutaneous Transmitters, the channel number is the Transmitter Identification Number
    (TIN). Following the channel number, each message contains a sixteen-bit data word. In the case of SCTs, the
    sixteen-bit data word is a digitized voltage. The last byte of the message is a timestamp.

    Byte	Contents
    0	Channel Number
    1	Most Significant Data Byte
    2	Least Significant Data Byte
    3	Timestamp or Version Number

    The data recorder will never store a message with channel number zero unless that message comes from the clock.
    All messages with channel number zero are guaranteed to be clocks.

    """

    def __init__(self, file_path, time_interval_hours=(0, 1), print_meta=False):
        self.print_meta = print_meta
        self.filepath = file_path
        self.file_label = file_path.split('/')[-1].split('.')[0]
        self.mean_point = None
        self.identifier = None
        self.data_address = None
        self.metadata = None
        self.data_size = None
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

        self._get_file_properties()

        self.adc_range = 2.7
        self.amp_factor = 300
        self.bit_size = 16
        self.volt_div = self.adc_range / (2 ** self.bit_size) / self.amp_factor * 1e3  # in mV unit

        # firmware dependent check needed in future?
        self.clock_tick_cycle = 7.8125e-3  # per second
        self.clock_division = self.clock_tick_cycle / 256.0
        self.time_interval_hours = time_interval_hours
        self.time_interval = self.time_interval_hours * 3600  # convert hourly interval to seconds

    def _mad_based_outlier(self, thresh=3.5):
        """
        Not being used:

        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
        """
        points = self.data
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def _rolling_median(self, threshold = 1, plot_glitches=False, print_output=False):
        if len(self.data.shape) == 1:
            self.data_4med = self.data[:, None]
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
        std_dev = np.std(self.data)
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
                if abs(self.data[location] - self.data[ii]) > diff_threshold * std_dev:
                    # plot glitches to be removed if plotting option is on
                    if self.plot_each_glitch:
                        plt.figure(figsize = (15, 4))
                        plt.plot(self.time[location - 512:location + 512],
                                 self.data[location - 512:location + 512], 'k')
                        plt.ylabel('Time (s)'); plt.title('Glitch '+str(glitch_count+1))
                        plt.show()
                    try:
                        value = self.data[i] + (self.time[location] - self.time[i]) * (
                        self.data[ii] - self.data[i]) / (self.time[ii] - self.time[i])
                        self.data[location] = value
                    except IndexError:
                        pass
                    glitch_count += 1

            except IndexError:
                pass
        self.glitch_count += glitch_count


    def glitch_removal(self, x_std_threshold=10, plot_glitches=False, print_output=False, plot_sub_glitches = False, tactic = 'big_guns'):
        """
        Seperate method, in order to adjust the thresholds manually
        """
        std_dev = np.std(self.data)
        self.n_possible_glitches = 0
        self.glitch_count        = 0
        self.plot_each_glitch = plot_sub_glitches
        if plot_glitches:
            plt.figure(figsize = (15, 4))
            plt.plot(self.time, self.data, 'k')
            plt.title('Full raw trace');plt.xlabel('Time (seconds)')
            plt.xlim(0,self.time[-1])
            plt.show()

        if tactic == 'old':
            mean_diff = np.mean(abs(np.diff(self.data)))
            if not self.mean_point:
                self.mean_point = np.mean(self.data)
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
            mean_diff = np.mean(abs(np.diff(self.data)))
            if not self.mean_point:
                self.mean_point = np.mean(self.data)
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
            plt.plot(self.time, self.data, 'k')
            plt.title('De-glitched trace');plt.xlabel('Time (seconds)')
            plt.xlim(0,self.time[-1])
            plt.show()
        self.data = np.ravel(self.data)

    def _get_file_properties(self):
        with open(self.filepath, 'rb') as f:
            # no offset
            f.seek(0)
            # The NDF file starts with a header of at least twelve bytes. The first four bytes
            # spell the NDF identifier " ndf".
            self.identifier = f.read(4)
            assert (self.identifier == b' ndf')

            meta_data_string_address = struct.unpack('>I', f.read(4))[0]
            self.data_address = struct.unpack('>I', f.read(4))[0]
            meta_data_length = struct.unpack('>I', f.read(4))[0]

            if meta_data_length != 0:
                f.seek(meta_data_string_address)
                self.metadata = f.read(meta_data_length)
                if self.print_meta:
                    print(self.metadata)
            else:
                print('meta data length unknown - not bothering to work it out...',)# end=' ')
                print('skipping')

            file_size = os.path.getsize(self.filepath)
            self.data_size = file_size - self.data_address

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


    def load(self, read_id, auto_glitch_removal = False):
        """
        Args:
            - read_id: Transmitter id to be read from the ndf file.
        Returns:

        This is based on the following analysis :
        The time stamps for messages in a given channel fall at K + (64 x N) where N = 0, 1, 2 or 3, and K is an offset
        that drifts slowly over the course of an entire ndf but is reasonably stable for a few seconds at a time.
        So for a chunk of an ndf the "Clean up ndf" subroutine first estimates K as the mode of the residuals of
        [time stamp/64] for the messages in the channel of interest. It then goes back to the same chunk and asks if,
        for each message, the residual is within +/- 9 of the mode of K. If yes, accept the message as coming from the
        transmitter of interest and not stray or corrupted signal.

        That means that the tolerance is +/- [9/64], or approximately +/- 14%. I worked this out myself as giving a
        good trade-off of throwing out bad messages and leaving gaps, and I think it's in the same range as the
        tolerance that Kevan uses in his program.

        I am using chunks of 2000 points (i.e. ~4s) by default, but you can get an idea of what would work by
        calculating K and plotting it over a whole ndf file. You will see it drift up and down and loop as the clock on
        board the transmitter drifts relative to the computer clock, but usually without discontinuities, so it is
        basically constant over a few seconds.

        """
        self.read_id = read_id

        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grab ids and time stamps
        e_bit_reads = np.fromfile(f, 'u1')
        transmitter_ids = e_bit_reads[::4]
        self.tids = transmitter_ids
        self.t_stamps = e_bit_reads[3::4]

        '''
        # Here we find bad message
        bad_messages = []
        transmitter_timestamps = self.t_stamps[transmitter_ids==read_id]
        begin_chunk = 0
        end_chunk = begin_chunk + 2000
        for i in range(transmitter_timestamps.size//2000):
            # Read a 2000 long chunk of data
            begin_chunk = 2000*i
            end_chunk = min(begin_chunk+2000, transmitter_timestamps.size)
            chunk_times = transmitter_timestamps[begin_chunk:end_chunk]
            mode_k = stats.mode(chunk_times % 64).mode[0] # 64 as 8*64 is 256 - should be 8 messages per time reset.
            for j in range(chunk_times.size):
                offset = (int(chunk_times[j]) - mode_k) % 64 # actual offset minus most common offset
                # same as (time%64) - k.
                if offset > 9 and offset < 51:
                    bad_messages.append(j + begin_chunk)
        '''

        # read again, but in 16 bit chunks, grab messages
        f.seek(self.data_address + 1)
        self.messages = np.fromfile(f, '>u2')[::2]

        # convert timestamps into correct time using clock id
        t_clock_data = np.zeros(self.messages.shape)
        t_clock_data[transmitter_ids == 0] = 1
        clock_data = np.cumsum(t_clock_data) * self.clock_tick_cycle
        fine_time_array = self.t_stamps * self.clock_division
        self.time_array = fine_time_array + clock_data

        self.data = self.messages[transmitter_ids == read_id] * self.volt_div
        self.time = self.time_array[transmitter_ids == read_id]

        bad_messages = []
        transmitter_timestamps = self.t_stamps[transmitter_ids==read_id]
        begin_chunk = 0
        end_chunk = begin_chunk + 2000
        for i in range(transmitter_timestamps.size//2000):
            # Read a 2000 long chunk of data
            begin_chunk = 2000*i
            end_chunk = min(begin_chunk+2000, transmitter_timestamps.size)
            chunk_times = transmitter_timestamps[begin_chunk:end_chunk]
            mode_k = stats.mode(chunk_times % 64).mode[0] # 64 as 8*64 is 512 - should be 8 messages per time reset.
            for j in range(chunk_times.size):
                offset = (int(chunk_times[j]) - mode_k) % 64 # actual offset minus most common offset
                # same as (time%64) - k.
                if offset > 9 and offset < 51:
                    bad_messages.append(j + begin_chunk)

        self.data = np.delete(self.data, bad_messages)
        self.time = np.delete(self.time, bad_messages)

        if auto_glitch_removal:
            self.glitch_removal()

    def correct_sampling_frequency(self, resampling_fs=512.0, overwrite=False):
        # first check that we are not interpolating datapoints for more than 1 second?
        try:
            assert max(np.diff(self.time)) < 1.0
        except:
            print  ('oh fucky, you interpolated for greater than one second!')
        self.time_diff = np.diff(self.time)

        # do linear interpolation between the points
        self.time_resampled = np.linspace(0, self.time[-1], num=self.time[-1] * resampling_fs)
        self.data_resampled = np.interp(self.time_resampled, self.time, self.data)
        self.resampled = True

        if overwrite:
            self.time = self.time_resampled[:]
            self.data = self.time_resampled[:]
            print('overwrite')

