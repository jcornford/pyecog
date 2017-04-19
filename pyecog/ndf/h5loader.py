import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd

class H5Dataset():
    """
    This is initially to just load up the h5 file converted by the ndf loader
    # todo everything should be simplified into one dictionary...
    """

    def __init__(self,fpath, mcode_id_str):
        self.fpath = fpath
        self.mcode_id_str = mcode_id_str
        self.h5dataset = None
        self.data = None
        self.time = None
        self.features_df = None
        self.features = None
        self.fs = None
        self.mode_std = None
        self.feature_col_labels = None
        self._load_data()

    def _load_data(self):
        with h5py.File(self.fpath, 'r+') as f:
            tid_dataset = f[self.mcode_id_str]

            for member in tid_dataset.keys():
                if str(member) == 'data':
                    self.data = np.array(tid_dataset['data'])
                if str(member) == 'time':
                    self.time = np.array(tid_dataset['time'])
                if str(member) == 'features':
                    self.features = np.array(tid_dataset['features'])
            for att_key in tid_dataset.attrs.keys():
                #['fs', 'tid', 'time_arr_info_dict', 'resampled', 'col_names', ;mode_std]
                if str(att_key) == 'col_names':
                    self.feature_col_labels = list(tid_dataset.attrs['col_names'])
                if str(att_key) == 'fs':
                    self.fs = tid_dataset.attrs['fs']
                if str(att_key) == 'mode_std':
                    self.mode_std = tid_dataset.attrs['mode_std']

            if self.time is None:
                time_arr_info_dict =  eval(tid_dataset.attrs['time_arr_info_dict'])
                self.time = np.linspace(time_arr_info_dict['min_t'],
                                        time_arr_info_dict['max_t'],
                                        num= time_arr_info_dict['max_t'] * time_arr_info_dict['fs'])
            if self.features is not None:
                self.features_df = pd.DataFrame(self.features, columns = [b.decode("utf-8")  for b in self.feature_col_labels])

    def plot(self):
        print ('Placeholder: Plot method to implement!')
        # have indexing argument...

class H5File():
    '''
    Class for reading h5 files:

    Use transmitter id to index which transmitter you want to access:
    h5obj = H5File(path_to_h5_file)here)
    h5obj[2] # for transmitter 2

    Attributes available are:
    -  h5obj[2].time
    -  h5obj[2].data

    # to do - first test that other modules dont treat as dictionary before finalising this.


    '''
    def __init__(self, filepath):
        self.filepath = filepath
        with h5py.File(self.filepath, 'r+') as f:

            if sys.version_info < (3,):
                self.attributes = dict(f.attrs.iteritems())
            else:
                self.attributes = dict(f.attrs.items())
            # eg. {'t_ids': array([ 1,  2 ]), 'fs_dict': '{1: 512.0, 2: 512.0, 'num_channels': 2}

            self.attributes['Mcode'] = list(f.keys())[0] # list in case there is more than one M1etc

    def __repr__(self):
        return 'Better formatting coming soon... \nAttributes:'+str(self.attributes)

    def __getitem__(self, tid):
        assert tid in self.attributes['t_ids'], 'ERROR: Invalid tid for file'
        tid_dataset = H5Dataset(self.filepath, self.attributes['Mcode']+'/'+str(tid))
        # again, retarded coding going on here - just have one master dict

        group_contents = {}
        group_contents['data'] = tid_dataset.data
        group_contents['time'] = tid_dataset.time

        group_contents['features'] = tid_dataset.features
        group_contents['feature_col_names'] = tid_dataset.feature_col_labels
        group_contents['mode_std'] = tid_dataset.mode_std
        group_contents['fs'] = tid_dataset.fs
        group_contents['features_df'] = tid_dataset.features_df

        return group_contents
