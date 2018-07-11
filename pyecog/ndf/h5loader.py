import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd

if sys.version_info < (3,):
    range = xrange
try:
    from utils import lprofile
except:
    pass

class H5Dataset():
    """
    This class is to represent recording for each hour in h5 file.

    Currently being used as
    """

    def __init__(self,fpath,tid):

        self.h5dataset = None
        self.data = None
        self.time = None
        self.features_df = None
        self.features = None
        self.fs = None
        self.scale_coef_for_feature_extraction = None
        self.feature_col_names = None
        self.load_data(fpath,tid)

    def load_data(self, fpath, tid):
        with h5py.File(fpath, 'r+') as f:
            mcode_id_str = list(f.keys())[0]+'/'+str(tid)
            if len(list(f.keys())) > 1:
                pass
                print('warning, more than one hour contained in h5 file')
            tid_dataset = f[mcode_id_str]

            for member in tid_dataset.keys():
                if str(member) == 'data':
                    # here instead of array could slice to avoid loading whole file
                    self.data = np.array(tid_dataset['data'])
                if str(member) == 'time':
                    self.time = np.array(tid_dataset['time'])
                if str(member) == 'features':
                    self.features = np.array(tid_dataset['features'])

            for att_key in tid_dataset.attrs.keys():
                # these will be ['fs', 'tid', 'time_arr_info_dict', 'resampled', 'col_names', ;mode_std]
                if str(att_key) == 'col_names':
                    self.feature_col_names = list(tid_dataset.attrs['col_names'])
                if str(att_key) == 'fs':
                    self.fs = tid_dataset.attrs['fs']
                if str(att_key) == 'scale_coef_for_feature_extraction':
                    self.scale_coef_for_feature_extraction = tid_dataset.attrs['scale_coef_for_feature_extraction']
                if str(att_key) == 'mode_std':
                    self.scale_coef_for_feature_extraction = tid_dataset.attrs['mode_std']

            if self.time is None:
                time_arr_info_dict =  eval(tid_dataset.attrs['time_arr_info_dict'])
                self.time = np.linspace(time_arr_info_dict['min_t'],
                                        time_arr_info_dict['max_t'],
                                        num= time_arr_info_dict['max_t'] * time_arr_info_dict['fs'])
            if self.features is not None:
                self.features_df = pd.DataFrame(self.features, columns = [b.decode("utf-8") for b in self.feature_col_names])

    def __getitem__(self, item):
        pass

class H5File():
    '''
    Class for reading h5 files:

    Example use:
    Transmitter id is used to index which transmitter you want to access:
        h5obj = H5File(path_to_h5_file)here)
        h5obj[2] # for transmitter 2

    This returns a dictionary for each transmitter id.

    Note:
        This code seems unnecessarily complicated, refactor and document required h5 format to facilitate code for
        loading from other file formats.
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
        return 'Attributes:'+str(self.attributes)

    def __getitem__(self, tid):
        assert tid in self.attributes['t_ids'], 'ERROR: Invalid tid for file. Valid tids are '+ str(self.attributes['t_ids'])
        tid_dataset = H5Dataset(self.filepath, tid)
        return tid_dataset.__dict__

