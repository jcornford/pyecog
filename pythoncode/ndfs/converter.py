import numpy as np
import os
import struct
from line_profiler import LineProfiler
import cProfile
import h5py


import matplotlib.pyplot as plt

def cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

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

class NDFLoader():
    '''
    Class to load ndf binary files.

    To implement:
        - Save as hdf5 file, assess file size in that format - 1mb smaller!
        - Read time of hdf5 file?
        - Interpolate missing data for stable 512 hz sampling.
        - Detect glitches and remove.
        - Add option to have baseline be one - remove median or something.
    '''

    def __init__(self, filepath, time_interval_hours = (0,1), print_meta = False):
        self.print_meta = print_meta
        self.filepath = filepath
        self.file_label = filepath.split('/')[-1].split('.')[0]
        print self.file_label
        self.identifier = None
        self.data_address = None
        self.metadata = None
        self.data_size = None
        self.channel_info = None
        self.data_dict = None

        self._get_file_properties()

        self.adc_range=2.7
        self.amp_factor=300
        self.bitsize=16
        self.volt_div=self.adc_range/(2**self.bitsize)/self.amp_factor*1e3 # in mV unit

        #firmware dependent check needed in future - this here or above?
        self.clock_tick_cycle=7.8125e-3 # per second
        self.clock_division=self.clock_tick_cycle/256.0
        self.time_interval_hours = time_interval_hours
        self.time_interval= self.time_interval_hours*3600; #convert hourly interval to seconds

    def glitch_removal(self, x_std_threshold = 40, plot_glitches = False ):
        print 'Code to remove glitches...'
        diff = abs(np.diff(self.data))
        mean_diff = np.mean(diff)
        threshold = np.std(diff)*x_std_threshold + mean_diff
        locs = np.where(diff>threshold,1,0)

        print 'Removed', np.sum(locs),'datapoints detected as glitches, with a threshold of',
        print x_std_threshold, 'times std deviation'
        #print self.time[locs ==1]

        if plot_glitches:
            ii = 1
            for ii in range(np.sum(locs)):
                i = self.time[locs ==1][ii]
                print i
                plt.plot(self.data[(i-20)*512:(i+20)*512])
                plt.show()
        print 'Need to now replace the glitches'

    def correct_sampling_frequency(self):
        print 'To do'


    def _get_file_properties(self):
        with open(self.filepath, 'rb') as f:
            # no offset
            f.seek(0)
            #The NDF file starts with a header of at least twelve bytes. The first four bytes
            #spell the NDF identifier " ndf".
            self.identifier = f.read(4)
            assert self.identifier == ' ndf'

            meta_data_string_address = struct.unpack('>I',f.read(4))[0]
            self.data_address = struct.unpack('>I',f.read(4))[0]
            meta_data_length = struct.unpack('>I',f.read(4))[0]

            if meta_data_length != 0:
                f.seek(meta_data_string_address)
                self.metadata = f.read(meta_data_length)
                if self.print_meta:
                    print self.metadata
            else:
                print 'meta data length unknown - not bothering to work it out...',
                print 'skipping'

            file_size = os.path.getsize(self.filepath)
            self.data_size = file_size - self.data_address

    def save(self, file_format = 'hdf5', channels_to_save = (-1), fs = 512, sec_per_row = 1, minimum_seconds = 1):
        '''
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

    def load(self,read_id):
        print 'currently only working in one transmitter mode'
        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        # read everything in 8bits, grab ids and time stamps
        e_bit_reads = np.fromfile(f,'u1')
        transmitter_ids = e_bit_reads[::4]
        self.tids = transmitter_ids
        self.t_stamps = e_bit_reads[3::4]

        # read again, but in 16 bit chunks, grab messages
        f.seek(self.data_address+1)
        self.messages = np.fromfile(f,'>u2')[::2]

        # convert timestamps into correct time using clock id
        self.clock_ticks = np.logical_not(transmitter_ids.astype('bool')).astype(int)
        #clock_ticks = np.where(transmitter_ids==0,1,0)
        self.clock_ticks= np.cumsum(self.clock_ticks)-1
        fine_time_array = self.t_stamps*self.clock_division
        coarse_time_array = self.clock_ticks*self.clock_tick_cycle
        self.time_array = fine_time_array+coarse_time_array

        if type(read_id) == int:
            self.data = self.messages[transmitter_ids==read_id]*self.volt_div
            self.time = self.time_array[transmitter_ids==read_id]

        elif type(read_id) == list:
            print 'WARNING: NOT FINISHED CODING'
            self.data_dict = {}
            for id in read_id:
                self.data_dict[str(id)] = self.messages[transmitter_ids==read_id]*self.volt_div
                self.time = self.time_array[transmitter_ids==read_id]

        elif read_id == 'all':
            print 'WARNING: NOT FINISHED CODING'
            self.data_dict = {}
            ids = set(self.tids)
            ids.remove(0)
            for id in ids:
                self.data_dict[str(id)] = self.messages[transmitter_ids==read_id]*self.volt_div
                self.time = self.time_array[transmitter_ids==read_id]


import time
dir = '/Users/Jonathan/Dropbox/'
start = time.clock()
ndf = NDFLoader(dir+'M1445362612.ndf')
ndf.load(8)
ndf.glitch_removal()
print (time.clock()-start)*1000, 'ms'
print ndf.data.shape
print ndf.time.shape
ndf.save()




start = time.clock()
file = h5py.File('/Users/Jonathan/Dropbox/M1445362612.hdf5', 'r')
data = file['data']
ndftime = file['time']
#return data
print (time.clock()-start)*1000, 'ms'



#plt.plot(data[:5120])
#plt.show()


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
3	Timestamp or Version Numbe

The data recorder will never store a message with channel number zero unless that message comes from the clock.
All messages with channel number zero are guaranteed to be clocks.

 '''