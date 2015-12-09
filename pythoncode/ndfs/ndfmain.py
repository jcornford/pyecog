import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from pythoncode import utils as phd
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

        self._make_feature_array()
        self._impute_and_scale()
        self.run_lda()
        self.plot_lda()


    def _make_feature_array(self):
        ii_features = h5py.File('inter_ictal_features.hdf5', 'r')
        i_features = h5py.File('ictal_features.hdf5', 'r')
        ii_secs = 0
        i_secs = 0
        for key in ii_features.keys():
            #print ii_features[key][:]
            ii_secs += ii_features[key].shape[0]
        for key in i_features.keys():
            #print i_features[key][:]
            i_secs += i_features[key].shape[0]

        print ii_secs, 'interictal seconds and ', i_secs, 'ictal seconds'
        self.annotated_array = np.zeros(shape =(ii_secs+i_secs,24) )
        i = 0
        for key in ii_features.keys():
            sub_array = np.hstack((ii_features[key], np.zeros(ii_features[key].shape[0])[:,None]))
            self.annotated_array[i:i+ii_features[key].shape[0]] = sub_array
            i += ii_features[key].shape[0]

        for key in i_features.keys():
            sub_array = np.hstack((i_features[key], np.ones(i_features[key].shape[0])[:,None]))
            self.annotated_array[i:i+i_features[key].shape[0]] = sub_array
            i += i_features[key].shape[0]

        print self.annotated_array
        print self.annotated_array.shape

    def _impute_and_scale(self):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        i_annotated_array = self.imputer.fit_transform(self.annotated_array[:,:-1])

        self.std_scaler = preprocessing.StandardScaler()
        self.std_scaler.fit(i_annotated_array[:,:])
        self.iss_features = self.std_scaler.transform(i_annotated_array[:,:])

        self.labels = np.ravel(self.annotated_array[:,-1])

        print self.iss_features
        print self.labels


    def run_lda(self, n_components = 3):
        self.lda = LinearDiscriminantAnalysis(n_components = n_components, solver='eigen', shrinkage='auto')
        self.lda_iss_features = self.lda.fit_transform(self.iss_features,self.labels)

    def plot_lda(self):
        cs = ['b','r','k','grey','grey']
        plt.figure()
        ax2 = plt.subplot(111, projection = '3d')
        lda = self.lda_iss_features
        #for i in range(lda.shrange(4000):
            #print phd.mc[cs[int(self.labels[i])]]

            ax2.scatter(lda[-i,0], lda[-i,1],lda[-i,2], c = phd.mc[cs[int(self.labels[-i])]], edgecolor = 'k',
                       s = 20, linewidth = 0.1, depthshade = True, alpha = 0.8)

        ax2.set_xlabel("LD 1",)
        ax2.set_ylabel("LD 2", )
        ax2.set_zlabel("LD 3")
        ax2.view_init(elev=-174., azim=73)
        #ax2.set_zlim((-5,5))
        #ax2.set_xlim((-4.5,6))
        #ax2.set_ylim((-5,12))
        ax2.set_title('3d LDA projection')
        plt.show()


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
            features_dset[:] = extractor.feature_array
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
            features_dset[:] = extractor.feature_array
            print extractor.feature_array

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
