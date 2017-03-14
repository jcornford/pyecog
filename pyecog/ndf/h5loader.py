import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

class H5Dataset():
    """
    This is initially to just load up the h5 file converted by the ndf loader
    """

    def __init__(self,fpath, mcode_id_str):
        self.fpath = fpath
        self.mcode_id_str = mcode_id_str
        self.h5dataset = None
        self.data = None
        self.time = None
        self.features = None
        self.col_labels = None
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

            if self.time is None:
                time_arr_info_dict =  eval(tid_dataset.attrs['time_arr_info_dict'])
                self.time = np.linspace(time_arr_info_dict['min_t'],
                                        time_arr_info_dict['max_t'],
                                        num= time_arr_info_dict['max_t'] * time_arr_info_dict['fs'])

    def plot(self):
        print ('Placeholder: Plot method to implement!')
        # have indexing argument...

class H5File():
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
        group_contents = {}
        group_contents['data'] = tid_dataset.data
        group_contents['time'] = tid_dataset.time
        group_contents['features'] = tid_dataset.features
        try:
            group_contents['col_names'] = f[self.attributes['Mcode']+'/'+str(tid)].attrs['col_names'].astype(str)
        except:
            pass
        return group_contents
