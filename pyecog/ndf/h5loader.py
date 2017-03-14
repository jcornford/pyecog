import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

class H5Dataset():
    """
    This is to just load up the h5 file converted by the ndf loader
    """
    def __init__(self, h5dataset):
        self.h5dataset = h5dataset
        self.data = None
        self.time = None
        self.features = None
        self.col_labels = None
        self._load_data()

    def _load_data(self):
        for member in self.h5dataset.keys():
            if str(member) == 'data':
                self.data = np.array(self.h5dataset['data'])
            if str(member) == 'time':
                self.time = np.array(self.h5dataset['time'])
            if str(member) == 'features':
                self.features = np.array(self.h5dataset['features'])


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
            self.attributes['Mcode'] = list(f.keys())[0]

    def __repr__(self):
        return 'Better formatting coming soon... \nAttributes:'+str(self.attributes)

    def __getitem__(self, tid):
        assert tid in self.attributes['t_ids'], 'ERROR: Invalid tid for file'
        with h5py.File(self.filepath, 'r+') as f:
            tid_dataset = H5Dataset(f[self.attributes['Mcode']+'/'+str(tid)])
            group_contents = {}
            group_contents['data'] = tid_dataset.data
            group_contents['time'] = tid_dataset.time
            group_contents['features'] = tid_dataset.features
            try:
                group_contents['col_names'] = f[self.attributes['Mcode']+'/'+str(tid)].attrs['col_names'].astype(str)
            except:
                pass
        return group_contents
