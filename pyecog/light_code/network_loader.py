import os
from os.path import join

import numpy as np

import utils


class LFPData(object):
    """
    Class for one section of data before the light pulse.
    """
    def __init__(self, path,preprocess, window_len, target_fs = 512, label = None, original_fs_dict = {}, filter = True, to_dwnsmple = False ):

        self.original_fs_dict = original_fs_dict
        self.target_fs = target_fs # this is not being used at the moment
        self.window_len = window_len*target_fs
        self.label = None
        self.filename = path

        print path
        self.alldata = np.load(path)
        print 'here'
        print self.alldata.shape
        #print self.alldata.shape
        if preprocess:

            key = self.filename.split('/')[-1].split('.')[0]
            print key
            try:
                self.original_fs = self.original_fs_dict[key]
            except KeyError:
                print 'Did not find key in fs dictonary!'
                # insert calculation code!

                if to_dwnsmple:
                    print 'got a shitty argument INCORRECT!'
                    self.original_fs =( to_dwnsmple * 500)/1000
                    print self.original_fs, 'is original rate in seconds'

                else:
                    exit(1)


            self.downsample_rate = int(self.original_fs * 1000/ float(self.target_fs)) # for converting between khz to 500 - in secconds


            # if more advanced can make a method for this
            #print "downsampled!"
            print(self.filename[-65:-3]+'...')
            self.fullfs = self.alldata[:,:]
            print self.alldata.shape, 'is now alldatashape'
            if filter:
                print 'filtering!'
                self.alldata[:,1] = utils.filterArray(self.alldata[:,1])
                print self.alldata.shape
                print self.downsample_rate, 'is downsample_rate'
            self.alldata = self.alldata[::self.downsample_rate,:]
            file_len = str(self.alldata.shape[0]/500)
            print file_len


        self.lfpdata = self.alldata[:,1]


        if self.alldata.shape[1]>1:
            self.lightindexes = self._getLightIndexes(self.fullfs[:,0])
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
                    self.name_array.append(self.filename[:-4]+'_'+str(lightindextuple[0]/float(self.target_fs))+'_outof_'+file_len)

            self.data_array = np.array(self.data_array)
            #self.data_array -= np.mean(self.data_array,axis = 0)
            # remove glitches
            #print 'remove glitches in lfpdata.py ran '
            print '*******', self.data_array.shape, 'is data array shape ********'
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



    def _getLightIndexes(self,light, plot = False):

        threshold =  ((max(light)-min(light))/2)+min(light)
        #threshold = 1000
        print threshold, 'is threshold', self.original_fs
        mask = np.where(light>threshold,1,0)
        mask1 = np.roll(mask[:],1)
        mask = abs(np.subtract(mask,mask1))
        lightindexes = np.where(mask==1)[0]
        print lightindexes.shape
        if not lightindexes.shape[0]%2 == 0:
            raise AssertionError('The file has an odd number of light pulses,\
                                 cannot automatically detect the pulse times')
        light_index_list = []

        if plot:
            import matplotlib.pyplot as plt
        for i in range(0, lightindexes.shape[0], 2):
            length = lightindexes[i+1]-lightindexes[i]
            if length > 9.5*self.original_fs*1000 and length < 10.5*self.original_fs*1000:
                pulse = (lightindexes[i]/self.downsample_rate,lightindexes[i+1]/self.downsample_rate)
                #print (pulse[1]-pulse[0])/float(512), 'seconds after ds'
                light_index_list.append(pulse)
                color = 'k'
            else:
                color = 'r'
            if plot:
                print i, (lightindexes[i+1]-lightindexes[i])/float(self.original_fs*1000)
                plt.plot(light[lightindexes[i]-1*self.original_fs*1000:lightindexes[i+1]+1000*self.original_fs], color = color)
            #print light[lightindexes[i]-512], light[lightindexes[i]+512]
        if plot:
            plt.show()
        return light_index_list

class SeizureData(object):
    '''
    Class to load all files in a base directory
    '''
    def __init__(self, base_dir_path, fs_dict = {}, target_fs=512,to_dwnsmple = False):
        '''

        '''
        self.target_fs = target_fs
        self.to_dwnsmple = to_dwnsmple # this is bullshit for the training data from vincent

        self.original_fs_dict = fs_dict

        if not os.path.isdir(base_dir_path):
            raise ValueError('%s is not a directory.' % base_dir_path)
        self.base_dir = base_dir_path

    def load_data(self, n_states = 3, window = 10, preprocess = True,):
        """
        Loads data for a network state and type of data into class variable.
        No output.

        Inputs:


        """
        self.n_states = n_states # this is a throw back from when we had c1, c2, c3 folders
        self.window = window
        self.filenames = self._get_filenames() # own method in case we want to get clever
        self.data_array_list = []
        self.label_colarray_list = []
        self.filename_list = []
        print self.filenames

        for i, filename in enumerate(self.filenames):
            print filename
            #print float(i)/float(len(self.filenames))*100.," percent complete         \r",
            # Each call of _load_data_from_file appends data to features_train
            self.temp_data,self.temp_label, self.name = self._load_data_from_filename(filename, preprocess=preprocess,to_dwnsmple = self.to_dwnsmple)
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

        filenames = [join(self.base_dir, name) for name in os.listdir(self.base_dir) if name[-4:] == '.npy']
        #print filenames

        if len(filenames) == 0:

            cfolders =  [os.path.join(self.base_dir, c_string) for c_string in os.listdir(self.base_dir) if c_string[0] == 'c']

            filenames = []
            for folder in cfolders:
                for name in os.listdir(folder):
                    if name[-4:] == '.npy':
                        filenames.append(os.path.join(folder, name))

        return filenames

    def _load_data_from_filename(self,filename, preprocess = True,to_dwnsmple = False ):

        for i in range(self.n_states):
            if filename.find('c'+str(i+1)) >-1: # in case pre-classified in folders
                label = i+1
            else:
                label = 0
        lfp_data = LFPData(filename,preprocess, self.window, label = label, target_fs=self.target_fs, original_fs_dict=self.original_fs_dict,to_dwnsmple = to_dwnsmple )

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