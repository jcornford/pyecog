
import glob
import os
from os.path import join
import numpy as np

from lfpData import LFPData

class LoadSeizureData(object):
    '''
    Class to load all files for a given name (e.g. all or animal)
    '''
    def __init__(self,base_dir):
        '''
        base_dir is the path to the directory containing the network state
        folders. These folders need to be c1,c2,c3 etc 
        '''
        self.fs = 512
        self.n_channels = 1
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
        for i, filename in enumerate(self.filenames):    
            #print float(i)/float(len(self.filenames))*100.," percent complete         \r",
            # Each call of _load_data_from_file appends data to features_train 
            self.temp_data,self.temp_label = self._load_data_from_filename(filename,preprocess=preprocess)
            self.data_array_list.append(self.temp_data)
            self.label_colarray_list.append(self.temp_label[:])
            #print self.temp_label
            #print self.temp_label.shape
        self.data_array = np.vstack((self.data_array_list))
        self.label_colarray = np.hstack((self.label_colarray_list))
        self.label_colarray = np.reshape(self.label_colarray, (self.label_colarray.shape[0],1))
        #print self.data_array.shape
        #print self.label_colarray.shape
        #print "\nDone"
        
    def _get_filenames(self):
        filenames = glob.glob(join(self.base_dir,'*','*'))    
        return filenames

    def _load_data_from_filename(self,filename, preprocess = True):
        for i in range(self.n_states):
            if filename.find('c'+str(i+1)) >-1:
                label = i+1
        lfp_data = LFPData(filename,preprocess, self.window, label = label)  
        
        return lfp_data.data_array, lfp_data.label_array
    
    def extract_feature_array(self,feature_extractors = [None]):
        self.features = []
        self.feature_names = []
        for extractor in feature_extractors:
            self.features.append(extractor.extract(self.data_array))
            self.feature_names.append(extractor.names)
        self.feature_names = [item for sublist in self.feature_names for item in sublist] # flatten
        print self.feature_names, 'ARE THE FEATURES USED'
        self.features = np.hstack([self.features[i] for i in range(len(self.features))])