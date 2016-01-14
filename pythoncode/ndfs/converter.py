import os
import struct
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class NDFLoader:
    """
    Class to load ndf binary files.

    To implement:
        - Save as hdf5 file, assess file size in that format - 1mb smaller!
        - Read time of hdf5 file?
        - Interpolate missing data for stable 512 hz sampling.
        - Detect glitches and remove.
        - Add option to have baseline be one - remove median or something.
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
        self.data = {}
        self.time = {}

        self._get_file_properties()

        self.adc_range = 2.7
        self.amp_factor = 300
        self.bit_size = 16
        self.volt_div = self.adc_range / (2 ** self.bit_size) / self.amp_factor * 1e3  # in mV unit

        # firmware dependent check needed in future - this here or above?
        self.clock_tick_cycle = 7.8125e-3  # per second
        self.clock_division = self.clock_tick_cycle / 256.0
        self.time_interval_hours = time_interval_hours
        self.time_interval = self.time_interval_hours * 3600  # convert hourly interval to seconds

    def remove_points_based_on_timestamps(self, indexes=[]):
        """Remove data points based on timestamp.

        """
        if indexes == []:
            indexes = self.tids

        for id in indexes:
            begin_chunk = 0
            end_chunk = begin_chunk + 2000
            for i in range(np.length(self.data[id])//2000):
                # Read a 2000 long chunk of data
                begin_chunk = 0*i
                end_chunk = max(begin_chink+2000, np.length(self.data[id]))
                chunck_times = self.time[id][begin_chunk:end_chunk]





    def glitch_removal(self, index, x_std_threshold=10, diff_threshold=20, plot_glitches=False, print_output=False):
        mean_diff = np.mean(abs(np.diff(self.data[index])))
        if not self.mean_point:
            self.mean_point = np.mean(self.data[index])
        std_dev = np.std(self.data[index])

        # identify candidate glitches based on std deviation
        threshold = std_dev * x_std_threshold + self.mean_point
        crossing_locations = np.where(self.data[index] > threshold)[0]

        if plot_glitches:
            plt.plot(self.time[index], self.data[index])
            plt.title('Full trace')
            plt.show()

        # check local difference is much bigger than the mean difference between points
        glitch_count = 0
        for location in crossing_locations:
            if location == 0:
                print('Warning: if two glitches at start, correction will fail')
            i = location - 1
            ii = location + 1
            # if abs(np.diff(self.data[i:ii])).all() > mean_diff * diff_threshold:
            try:
                if abs(self.data[index][location] - self.data[index][ii]) > 10 * std_dev:
                    # plot glitche to be removed if plotting option is on
                    if plot_glitches:
                        plt.plot(self.time[index][location - 512:location + 512],
                                 self.data[index][location - 512:location + 512])
                        plt.show()
                    try:
                        value = self.data[index][i] + (self.time[index][location] - self.time[index][i]) * (
                        self.data[index][ii] - self.data[index][i]) / (self.time[index][ii] - self.time[index][i])
                        self.data[index][location] = value
                    except KeyError:
                        pass
                    glitch_count += 1
            except KeyError:
                pass

        if print_output:
            print('Removed', glitch_count, 'datapoints detected as glitches, with a threshold of', end=' ')
            print(x_std_threshold, 'times the std deviation. Therefore threshold was:', std_dev * x_std_threshold)
            print('above mean. Also used local difference between points, glitch was at least', diff_threshold, end=' ')
            print('greater than mean difference.')

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
                print('meta data length unknown - not bothering to work it out...', end=' ')
                print('skipping')

            file_size = os.path.getsize(self.filepath)
            self.data_size = file_size - self.data_address

    def save(self, file_format='hdf5', channels_to_save=(-1), fs=512, sec_per_row=1, minimum_seconds=1):
        """
        Default is to save all channels (-1). If you want to specify which channels to save
        pass in a tuple with desired channel numbers. e.g. (1,3,5,9) for channels 1, 3, 5 and 9.
        Info on channels and their recording length can be found in the channel_info attribute.

        Currently accepting the following savefile format options:
            - csv
            - xls
            - hdf5
            - npy
            - pickle

        Strongly recommended to save in hdf5 file format

        '''
        print 'WARNING: SAVE NOT FINISHED'

        savefile = h5py.File(self.filepath[:-4]+'.hdf5', 'w')
        hdf5_data = savefile.create_dataset('data', shape=self.data.shape, dtype='float')
        hdf5_time = savefile.create_dataset('time', shape = self.time.shape, dtype='float')
        #hdf5_data = savefile.create_dataset(self.file_label+'_data', shape=self.data.shape, dtype='float')
        #hdf5_time = savefile.create_dataset(self.file_label+'_time', shape = self.time.shape, dtype='float')

        if self.resampled:
            hdf5_data[:] = self.data_512hz
            hdf5_time[:] = self.time_512hz
        else:
            hdf5_data[:] = self.data
            hdf5_time[:] = self.time

        '''
        #implement multiple processes for saving
        dp_per_row = int(fs*sec_per_row)# could end up changing what they ask for...
        array = np.array(ndf.data_dict['9'])
        print 'cutting',array.shape[0]%dp_per_row, 'datapoints'
        row_index = array.shape[0]/dp_per_row # remeber floor division if int
        save_array = np.reshape(array[:row_index*dp_per_row],newshape = (row_index,dp_per_row))
        #probs dont need to change into an array before saving - but if new view, probs not big deal?
        '''

    def load(self, read_id=[]):
        """
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

        Args:
            read_id:

        Returns:

        """
        if read_id == []:
            read_id = set(self.tids)
            read_id.remove(0)

        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grab ids and time stamps
        e_bit_reads = np.fromfile(f,'u1')
        transmitter_ids = e_bit_reads[::4]
        self.tids = transmitter_ids
        self.t_stamps = e_bit_reads[3::4]

        # Here we find bad message
        bad_messages = {}
        for id in read_id:
            bad_messages[id] = []
            transmitter_timestamps = self.t_stamps[transmitter_ids==id]
            begin_chunk = 0
            end_chunk = begin_chunk + 2000
            for i in range(transmitter_timestamps.size//2000):
                # Read a 2000 long chunk of data
                begin_chunk = 2000*i
                end_chunk = min(begin_chunk+2000, transmitter_timestamps.size)
                chunk_times = transmitter_timestamps[begin_chunk:end_chunk]
                mod_k = stats.mode(chunk_times % 64).mode[0]
                for j in range(chunk_times.size):
                    offset = (int(chunk_times[j]) - mod_k) % 64
                    if offset > 9 and offset < 51:
                        bad_messages[id].append(j + begin_chunk)
        print(len(bad_messages[8]))
        print(len(self.t_stamps[transmitter_ids == 8]))

        # read again, but in 16 bit chunks, grab messages
        f.seek(self.data_address+1)
        self.messages = np.fromfile(f,'>u2')[::2]

        # convert timestamps into correct time using clock id
        t_clock_data = np.zeros(self.messages.shape)
        t_clock_data[transmitter_ids == 0] = 1
        clock_data = np.cumsum(t_clock_data) * self.clock_tick_cycle
        fine_time_array = self.t_stamps * self.clock_division
        self.time_array = fine_time_array + clock_data

        for id in read_id:
            self.data[id] = self.messages[transmitter_ids == id] * self.volt_div
            self.data[id] = np.delete(self.data[id], bad_messages[id])
            self.time[id] = self.time_array[transmitter_ids == id]
            self.time[id] = np.delete(self.time[id], bad_messages[id])
        # first check that we are not interpolating datapoints for more than 1 second?
        #assert max(np.diff(self.time)) < 1.0
        self.time_diff = np.diff(self.time)
        if max(np.diff(self.time)) < 1.0:
            print 'WARNING: assert max(np.diff(self.time)) < 1.0, would fail'

        # do linear interpolation between the points
        self.time_512hz = np.linspace(0,length,num=length*fs)
        self.data_512hz = np.interp(self.time_512hz,self.time,self.data)
        self.resampled = True

        if overwrite:
            self.time =  self.time_512hz[:]
            self.data = self.data_512hz[:]
            print 'overwrite'

            self.time[index] = self.time_512hz[:]
            self.data[index] = self.data_512hz[:]
            print('overwrite')


def main(filename):
    print("Reading : " + filename)
    start = time.clock()
    ndf = NDFLoader(filename)
    ndf.load([8])
    ndf.glitch_removal(index=8, plot_glitches=False, print_output=True)
    print((time.clock() - start) * 1000, 'ms to load the ndf file')

    start2 = time.clock()
    ndf.correct_sampling_frequency(index=8)
    print((time.clock() - start2) * 1000, 'ms to load resample')

    times = ndf.time[8] * 1000
    diffs = np.diff(times)
    fs_ac = 1000.0 / diffs
    print(times)
    print(diffs)

    plt.figure()
    plt.hist(fs_ac, bins=50, normed=True)
    # plt.xlim(0, 700)
    plt.xlabel('Instantaneous frequency (Hz)')
    plt.title(filename + ' instantaneous sampling frequencies')
    # plt.savefig('../../fs distribution.png')
    plt.show()

    # print (ndf.time_diff[:20]*1000)/(1/512.0*1000)
    # print np.std(ndf.time_diff[:20]*1000)
    print((1 / 512.0) * 1000)
    print(diffs / ((1 / 512.0) * 1000))
    # print (ndf.time_512hz[:20]*1000)/(1/512.0*1000)

    print(1000.0 / np.max(diffs))
    print(1000.0 / np.min(diffs))


if __name__ == "__main__":
    main(sys.argv[1])

'''
ndf.save()

start = time.clock()
file = h5py.File('/Users/Jonathan/Dropbox/M1445362612.hdf5', 'r')
data = file['data']
ndftime = file['time']
#return data
print (time.clock()-start)*1000, 'ms to load the hdf5 file'

'''

# plt.plot(data[:5120])
# plt.show()


# From open source instruments website
'''
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

 '''
