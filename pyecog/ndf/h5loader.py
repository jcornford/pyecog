import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

class H5Dataset():
    """
    This is initially to just load up the h5 file converted by the ndf loader
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
        self.group_contents = {} # holding all the data (not just voltage) for each tid
        with h5py.File(self.filepath, 'r+') as f:
            if sys.version_info < (3,):
                self.attributes = dict(f.attrs.iteritems())
            else:
                self.attributes = dict(f.attrs.items())

            self.attributes['Mcode'] = list(f.keys())[0]
            for tid in self.attributes['t_ids']:
                tid_dataset = H5Dataset(f[self.attributes['Mcode']+'/'+str(tid)])

                self.group_contents[tid] = {}
                self.group_contents[tid]['data'] = tid_dataset.data
                self.group_contents[tid]['time'] = tid_dataset.time
                self.group_contents[tid]['features'] = tid_dataset.features
                try:
                    self.group_contents[tid]['col_names'] = f[self.attributes['Mcode']+'/'+str(tid)].attrs['col_names'].astype(str)
                except:
                    pass

    def __repr__(self):
        return 'Better formatting coming soon... \nAttributes:'+str(self.attributes)

    def __getitem__(self, item):
        #assert type(item) == int
        assert item in self.attributes['t_ids'], 'ERROR: Invalid tid for file'
        return self.group_contents[item]
