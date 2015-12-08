
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import cProfile

from line_profiler import LineProfiler

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

    def __init__(self, filepath, time_interval_hours = (0,1)):
        self.filepath = filepath
        self.identifier = None
        self.data_address = None
        self.metadata = None
        self.data_size = None
        self.channel_info = None
        self.data_dict = {}

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

    #@cprofile
    #@lprofile()
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
                #print self.metadata
            else:
                print 'meta data length unknown - not bothering to work it out...',
                print 'skipping'

            file_size = os.path.getsize(self.filepath)
            self.data_size = file_size - self.data_address
            #print 'data is ', self.data_size, 'bytes long'

    #@cprofile
    #@lprofile()
    #@lprofile()
    def save(self, file_format, channels_to_save = (-1), fs = 512, sec_per_row = 1, minimum_seconds = 1):
        '''
        Default is to save all channels (-1). If you want to specify which channels to save
        pass in a tuple with desired channel numbers. e.g. (1,3,5,9) for channels 1, 3, 5 and 9.
        Info on channels and their recording length can be found in the channel_info attribute.

        Currently accepting the following savefile format options:
            - csv
            - xls
            - hd5
            - npy
            - pickle

        '''
        print 'detailed method not written yet - check loading is correct and optimise it - \
        perhaps work out how many and pre-assign to array?'

        #implement multiple processes for saving
        dp_per_row = int(fs*sec_per_row)# could end up changing what they ask for...
        array = np.array(ndf.data_dict['9'])
        print 'cutting',array.shape[0]%dp_per_row, 'datapoints'
        row_index = array.shape[0]/dp_per_row # remeber floor division if int
        save_array = np.reshape(array[:row_index*dp_per_row],newshape = (row_index,dp_per_row))
        #probs dont need to change into an array before saving - but if new view, probs not big deal?

    @lprofile()
    #@cprofile
    def load(self,read_id=8):
        #print 'currently only working in one transmitter mode'
        f = open(self.filepath, 'rb')
        f.seek(self.data_address)

        e_bit_reads = np.fromfile(f,'u1')
        transmitter_ids = e_bit_reads[::4]
        self.tids = transmitter_ids
        self.t_stamps = e_bit_reads[3::4]#*self.clock_division

        f.seek(self.data_address+1)
        self.messages = np.fromfile(f,'>H')[::2]
        #print messages[:20]

        #clock_ticks = np.where(transmitter_ids==0,1,0)
        self.clock_ticks = np.logical_not(transmitter_ids.astype('bool')).astype(int)
        self.clock_ticks= np.cumsum(self.clock_ticks)-1
        fine_time_array = self.t_stamps*self.clock_division
        coarse_time_array = self.clock_ticks*self.clock_tick_cycle
        self.time_array = fine_time_array+coarse_time_array

        self.data = np.vstack((self.time_array[transmitter_ids==read_id],self.messages[transmitter_ids==read_id]))
        #self.data = self.messages[transmitter_ids==read_id]
        '''
        if selected_ids_list is None:
            ids = set(transmitter_id)
            ids.remove(0)
            print 'Not written code to properly analyse all yet, please supply value'
        '''

dir = '/Users/Jonathan/Dropbox/'
ndf = NDFLoader(dir+'M1445362612.ndf')
ndf.load()
print ndf.data.shape



'''
        %find clock message
        clock_tick_pos=find(id==0);
        clock_tick_pos(end+1)=length(id)+1;%tick,data,tick,...,tick,data,add_tick
        start_pos=floor(time_interval(1)/clock_tick_cycle)+1;
        duration=ceil(time_interval(2)/clock_tick_cycle);
        end_pos=start_pos+duration;
        if end_pos>length(clock_tick_pos)%trim later could stitch
                end_pos=length(clock_tick_pos)-1;
        end
        id=id(clock_tick_pos(start_pos):clock_tick_pos(end_pos));
        sample=bitand(temp(clock_tick_pos(start_pos):clock_tick_pos(end_pos)),bin2dec('00000000111111111111111100000000'));%sample
        sample=bitshift(sample,-8);
        time_stamp=bitand(temp(clock_tick_pos(start_pos):clock_tick_pos(end_pos)),bin2dec('00000000000000000000000011111111'));

        clear temp;

        %realign time stamps to proper time scale
        clock_tick_pos=clock_tick_pos(start_pos:end_pos)-clock_tick_pos(start_pos)+1;
        time_interval=1:(length(clock_tick_pos)-1);
        temp2=arrayfun(@(x,y,z)time_stamp(x+1:y-1)*clock_division+clock_tick_cycle*z,clock_tick_pos(1:end-1),clock_tick_pos(2:end),time_interval','UniformOutput',false);
        time_stamp=[time_stamp(1:clock_tick_pos(1)-1)*clock_division;cell2mat(temp2)];
        clear temp2;

        %get rid of clock tick lines
        clock_tick_pos(end)=[];
        sample(clock_tick_pos)=[];id(clock_tick_pos)=[];
        %seperate channels
        for channelidx=channels
            idx=find(id==channelidx);
            raw{channelidx}=[time_stamp(idx)/3600,volt_div*sample(idx)];%convert back to hours and corre
        '''


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

http://www.ncbi.nlm.nih.gov/pubmed/20172805
http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4360820/
 '''