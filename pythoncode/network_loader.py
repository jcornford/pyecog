
import glob
import os
from os.path import join
import numpy as np

class LFPData(object):
    """
    Class for one section of data before the light pulse.
    """
    def __init__(self, path,preprocess, window_len, fs = 512, label = None, downsample_rate = 20):
        self.fs = 512
        self.window_len = window_len*fs
        self.label = None
        self.filename = path

        self.alldata = np.load(path)

        #print self.alldata.shape
        if preprocess:
            # if more advanced can make a method for this
            #print "downsampled!"
            print(path[-35:-3]+'...'),
            self.alldata = self.alldata[::downsample_rate,:]
            print str(self.alldata.shape[0]/500)+'secs'


        self.lfpdata = self.alldata[:,1]


        if self.alldata.shape[1]>1:
            self.lightindexes = self._getLightIndexes(self.alldata[:,0])
            self.n_lightpulses = len(self.lightindexes)

            self.data_array = []
            self.label_array = []
            self.name_array = []
            #print len(self.lightindexes)
            for lightindextuple in self.lightindexes:
                #print lightindextuple
                i = lightindextuple[0] - (self.window_len) # in case we
                if i > 0:
                    data_section = self.lfpdata[i:lightindextuple[0]]
                    self.data_array.append(data_section)
                    self.label_array.append(label)
                    self.name_array.append(self.filename[:-3]+str(lightindextuple[0]/500))

            self.data_array = np.array(self.data_array)
            #self.data_array -= np.mean(self.data_array,axis = 0)
            # remove glitches
            #print 'remove glitches in lfpdata.py ran '
            self.threshold = np.std(self.data_array, axis = 1)*7
            self.mask = np.where(self.data_array>self.threshold[:,None],0,1)
            self.mask2 = np.where(self.data_array<-self.threshold[:,None],0,1)
            self.mask += self.mask2
            self.mask /= 2
            self.data_array *= self.mask
            ### Finish removing glitches ###
            self.label_array = np.array(self.label_array)
            self.name_array = np.array(self.name_array)
            #plt.figure(figsize=(12,6))
            #plt.plot(self.data_array.T)
            #plt.title(self.filename)
            assert len(self.data_array.shape) ==2



    def _getLightIndexes(self,light):
        threshold =  ((max(light)-min(light))/2)+min(light)
        mask = np.where(light>threshold,1,0)
        mask1 = np.roll(mask[:],1)
        mask = abs(np.subtract(mask,mask1))
        lightindexes = np.where(mask==1)[0]
        if not lightindexes.shape[0]%2 == 0:
            raise AssertionError('The file has an odd number of light pulses,\
                                 cannot automatically detect the pulse times')
        light_index_list = []
        for i in range(0,lightindexes.shape[0],2):
            pulse = (lightindexes[i],lightindexes[i+1])
            light_index_list.append(pulse)
        return light_index_list

class SeizureData(object):
    '''
    Class to load all files for a given name (e.g. all or animal)
    '''
    def __init__(self,base_dir, amount_to_downsample):
        '''
        base_dir is the path to the directory containing the network state
        folders. These folders need to be c1,c2,c3 etc
        '''
        self.fs = 512
        self.n_channels = 1
        self.amount_to_downsample = amount_to_downsample
        if not os.path.isdir(base_dir):
            raise ValueError('%s is not a directory.' % base_dir)
        self.base_dir = base_dir

    def load_data(self, n_states = 3,window = 10, preprocess = True):
        """
        Loads data for a network state and type of data into class variable.
        No output.

        Inputs:


        """
        self.n_states = n_states
        self.window = window
        self.filenames = self._get_filenames() # why the _again?
        #for filename in self.filenames:
            #print filename
        self.data_array_list = []
        self.label_colarray_list = []
        self.filename_list = []
        for i, filename in enumerate(self.filenames):
            #print float(i)/float(len(self.filenames))*100.," percent complete         \r",
            # Each call of _load_data_from_file appends data to features_train
            self.temp_data,self.temp_label, self.name = self._load_data_from_filename(filename,preprocess=preprocess)
            self.data_array_list.append(self.temp_data)
            self.label_colarray_list.append(self.temp_label[:])
            self.filename_list.append(self.name[:])
            #print self.temp_label
            #print self.temp_label.shape
        self.data_array = np.vstack((self.data_array_list))
        self.label_colarray = np.hstack((self.label_colarray_list))
        self.label_colarray = np.reshape(self.label_colarray, (self.label_colarray.shape[0],1))
        self.filename_list = np.hstack((self.filename_list))
        #print self.data_array.shape
        #print self.label_colarray.shape
        #print self.filename_list.shape
        #print "\nDone"

    def _get_filenames(self):
        filenames = glob.glob(join(self.base_dir,'*','*'))
        return filenames

    def _load_data_from_filename(self,filename, preprocess = True):
        for i in range(self.n_states):
            if filename.find('c'+str(i+1)) >-1:
                label = i+1
        lfp_data = LFPData(filename,preprocess, self.window, label = label, downsample_rate=self.amount_to_downsample)

        return lfp_data.data_array, lfp_data.label_array, lfp_data.name_array

    def extract_feature_array(self,feature_extractors = [None]):
        self.features = []
        self.feature_names = []
        for extractor in feature_extractors:
            self.features.append(extractor.extract(self.data_array))
            self.feature_names.append(extractor.names)
        self.feature_names = [item for sublist in self.feature_names for item in sublist] # flatten
        print self.feature_names, 'ARE THE FEATURES USED'
        self.features = np.hstack([self.features[i] for i in range(len(self.features))])