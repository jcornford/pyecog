import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pythoncode.extrator import FeatureExtractor
from converter import NDFLoader

class Main():
    def __init__(self):
        self.dir_path = '/Users/Jonathan/Dropbox/EEG/'
        self.annotation_file = 'seizures_Rat8_20-271015.xlsx'

        annotations = pd.read_excel(self.dir_path+self.annotation_file, index_col=False)
        annotations.columns = ['fname','start','end']
        self.annotations = annotations.dropna(axis=0, how='all')
        #self._load_ictal_raw()
        #self._load_ictal_hdf5()

        #self._load_inter_ictal_raw()
        #self._load_inter_ictal_hdf5()

        self._lda_plot()

    def _load_inter_ictal_hdf5(self):
        print 'going to lose time on cutting into arrays'
        print 'no normalisation'
        f = h5py.File('inter_ictal_file.hdf5', 'r')
        if_f = h5py.File('inter_ictal_features.hdf5', 'w')
        for key in f.keys():
            s_length = f[key].shape[0]/512
            trimmed_array = f[key][:s_length*512,1]
            data_array = np.reshape(trimmed_array, newshape = (s_length,512))
            print data_array.shape, 'is here'
            extractor = FeatureExtractor(data_array)
            features_dset = if_f.create_dataset(key+'_features', shape = extractor.feature_array.shape, dtype='float')
            features_dset = extractor.feature_array
            print extractor.feature_array.shape, 'is feauture array shape'

    def _load_ictal_hdf5(self):
        print 'going to lose time on cutting into arrays'
        print 'no normalisation'
        f = h5py.File('ictal_file.hdf5', 'r')
        if_f = h5py.File('ictal_features.hdf5', 'w')
        for key in f.keys():
            s_length = f[key].shape[0]/512
            trimmed_array = f[key][:s_length*512,1]
            data_array = np.reshape(trimmed_array, newshape = (s_length,512))
            print data_array.shape, 'is here'
            extractor = FeatureExtractor(data_array)
            features_dset = if_f.create_dataset(key+'_features', shape = extractor.feature_array.shape, dtype='float')
            features_dset = extractor.feature_array
            print extractor.feature_array.shape

    def _load_ictal_raw(self):
        self.f = h5py.File('ictal_file.hdf5', 'w')
        # only going to load a ndf with max of two seizures in it  - will fail with three!
        for i in xrange(self.annotations.shape[0]):
        #for i in xrange(1):
            fname = self.annotations.iloc[i,0]
            start = self.annotations.iloc[i,1]
            end   = self.annotations.iloc[i,2]
            print fname, end , start

            ndf =  NDFLoader(self.dir_path+'ndf/'+fname, print_meta=True)
            ndf.load()

            np.set_printoptions(precision=3, suppress = True)
            ictal_time = ndf.time[(ndf.time >= start) & (ndf.time <= end)]
            ictal_data = ndf.data[(ndf.time >= start) & (ndf.time <= end)]

            print ictal_data.shape[0]
            print ictal_time.shape
            try:
                self.fname_dset = self.f.create_dataset(fname, (ictal_data.shape[0],2), dtype='float')
            except:
                self.fname_dset = self.f.create_dataset(fname+'_'+str(2), (ictal_data.shape[0],2), dtype='float')

            self.fname_dset[:,0] = ictal_time
            self.fname_dset[:,1] = ictal_data

    def _load_inter_ictal_raw(self):
        iif = h5py.File('inter_ictal_file.hdf5', 'w')
        # only going to load a ndf with max of two seizures in it  - will fail with three!
        for i in xrange(self.annotations.shape[0]):
        #for i in xrange(1):
            fname = self.annotations.iloc[i,0]
            start = self.annotations.iloc[i,1]
            end   = self.annotations.iloc[i,2]
            print fname, end , start

            ndf =  NDFLoader(self.dir_path+'ndf/'+fname, print_meta=True)
            ndf.load()

            np.set_printoptions(precision=3, suppress = True)
            inter_ictal_time = ndf.time[(ndf.time <= start) | (ndf.time >= end)]
            inter_ictal_data = ndf.data[(ndf.time <= start) | (ndf.time >= end)]

            print inter_ictal_data.shape[0]
            print inter_ictal_time.shape
            try:
                fname_dset = iif.create_dataset(fname, (inter_ictal_data.shape[0],2), dtype='float')
            except:
                print 'here'
                fname = fname+'_2'
                fname_dset = iif.create_dataset(fname, (inter_ictal_data.shape[0],2), dtype='float')


            fname_dset[:,0] = inter_ictal_time
            fname_dset[:,1] = inter_ictal_data



if __name__ == '__main__':
    n = Main()
