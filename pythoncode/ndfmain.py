from __future__ import print_function
import os
import pandas as pd
import numpy as np
import pickle
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
from sklearn.metrics import classification_report

from pythoncode import utils
from pythoncode.extractor import FeatureExtractor
from converter import NDFLoader
from pythoncode.classifier import NetworkClassifer

class Main():
    def __init__(self, normalise_seconds = True):
        self.dir_path = '/Users/Jonathan/Dropbox/EEG/'
        self.annotation_file = 'seizures_Rat8_20-271015.xlsx'

        self.normalise_seconds = True
        self.row_length = 512*1


        annotations = pd.read_excel(self.dir_path+self.annotation_file, index_col=False)
        annotations.columns = ['fname','start','end']
        self.annotations = annotations.dropna(axis=0, how='all')
        self._load_ictal_raw()
        self._extract_features_load_ictal_hdf5()

        #return None
        self._load_inter_ictal_raw()
        self._extract_features_load_inter_ictal_hdf5()

        self._make_feature_array()
        #self._impute_and_scale()
        #self.run_lda()
        #self.plot_lda()
        #self.marking()

        self.classify()
        self.stats()

    def stats(self):
        #self.X_train,self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.iss_features, self.labels, test_size=0.3, random_state=3)

        #classifier = NetworkClassifer(self.X_train, self.y_train,
         #                    self.X_test,self.y_test)
        classifier_trained = pickle.load(open('../saved_clf','rb'))
        ax5 = plt.subplot(111)
        ax5.set_title('Feature importance')
        ax5.set_ylabel('Importance (%)')
        import pandas as pd
        df = pd.DataFrame(classifier_trained.r_forest.feature_importances_*100,classifier_trained.feature_labels)
        df = df.sort(columns=0)
        df.plot(kind='bar', ax = ax5, rot = 80, legend = False, grid = False, color=utils.mc['r'])

        ax5.spines['right'].set_color('none')
        ax5.spines['top'].set_color('none')
            # Disable ticks.
        ax5.xaxis.set_ticks_position('bottom')
        ax5.yaxis.set_ticks_position('left')
        plt.tight_layout()

        plt.show()

        #y_true =
        #y_pred = []
        #target_names = ['inter-ictal', 'ictal']
        #classification_report(y_true, y_pred, target_names=target_names)

    def classify(self):

        self.X_train,self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.annotated_array[:,:-1],
                                                                                                np.ravel(self.annotated_array[:,-1]),
                                                                                                test_size=0.3,
                                                                                                random_state=3)

        classifier = NetworkClassifer(self.X_train, self.y_train,
                             self.X_test,self.y_test)
        classifier.run()

        f = open('../saved_clf','wb')
        pickle.dump(classifier,f)

        #classifier.pca(n_components = 3)
        #classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
        #classifier.lda_run()
        #classifier.pca_run()

    @staticmethod
    def _remove_glitches(data_array):
        '''
        DOESNT WORK!
        Probably needs updating! Doesn't replace with previous values
        or anything clever - just 0...
        Returns:

        '''

        threshold = np.std(data_array, axis = 1)*7
        mask = np.where(data_array>threshold[:,None],0,1)
        mask2 = np.where(data_array<-threshold[:,None],0,1)
        mask += mask2
        mask /= 2
        data_array *= mask
        return data_array


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

        #print ii_secs, 'interictal seconds and ', i_secs, 'ictal seconds'
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

        #print self.annotated_array
        #print self.annotated_array.shape

    def _impute_and_scale(self):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        i_annotated_array = self.imputer.fit_transform(self.annotated_array[:,:-1])

        self.std_scaler = preprocessing.StandardScaler()
        self.std_scaler.fit(i_annotated_array[:,:])
        self.iss_features = self.std_scaler.transform(i_annotated_array[:,:])

        self.labels = np.ravel(self.annotated_array[:,-1])

        #print self.iss_features.shape, 'is feature matrix'
        #print self.labels.shape, 'is labels number'


    def run_lda(self, n_components = 3):
        self.lda = LinearDiscriminantAnalysis(n_components = n_components, solver='eigen', shrinkage='auto')
        self.lda_iss_features = self.lda.fit_transform(self.iss_features,self.labels)

    def plot_lda(self):
        cs = ['b','r','k','grey','grey']
        plt.figure()
        ax2 = plt.subplot(111)
        lda = self.lda_iss_features
        ictal = lda[self.labels==1]
        inter_ictal = lda[self.labels==0]

        ax2.scatter(ictal[:,0], ictal[:,1], c = utils.mc['r'], edgecolor = 'k', s=20, linewidth=0.4, alpha=0.4,
                    zorder = 2,label = '1931 ictal seconds')

        ax2.scatter(inter_ictal[:,0],inter_ictal[:,1], c = utils.mc['b'],edgecolor = 'k', s=20, linewidth=0.4, alpha=0.4,
                    zorder = 1, label = '105163 inter-ictal seconds')
        ax2.set_xlabel("LD 1",)
        ax2.set_ylabel("LD 2", )
        #ax2.set_zlabel("LD 3")
        #ax2.view_init(elev=-174., azim=101)
        #ax2.set_zlim((-5,5))
        ax2.set_xlim((-3,9))
        ax2.set_ylim((-9,8))
        ax2.set_title("2d LDA projection")

        ax2.legend(frameon= True,ncol = 1, loc ='best', fontsize = 10)

        plt.show()


    def _extract_features_load_inter_ictal_hdf5(self):
        #print 'going to lose time on cutting into arrays'
        #print 'no normalisation'
        f = h5py.File('inter_ictal_file.hdf5', 'r')
        if_f = h5py.File('inter_ictal_features.hdf5', 'w')
        for key in f.keys():
            s_length = f[key].shape[0]/self.row_length
            trimmed_array = f[key][:s_length*self.row_length,1]
            data_array = np.reshape(trimmed_array, newshape = (s_length,self.row_length))
            #data_array = self._remove_glitches(data_array)
            if self.normalise_seconds:
                data_array = utils.normalise(data_array)
            extractor = FeatureExtractor(data_array)
            features_dset = if_f.create_dataset(key+'_features', shape = extractor.feature_array.shape, dtype='float')
            features_dset[:] = extractor.feature_array
            #print extractor.feature_array.shape

    def _extract_features_load_ictal_hdf5(self):
        #print 'going to lose time on cutting into arrays'
        #print 'no normalisation'
        f = h5py.File('ictal_file.hdf5', 'r')
        if_f = h5py.File('ictal_features.hdf5', 'w')
        for key in f.keys():
            s_length = f[key].shape[0]/self.row_length
            trimmed_array = f[key][:s_length*self.row_length,1]
            data_array = np.reshape(trimmed_array, newshape = (s_length,self.row_length))
            #data_array = self._remove_glitches(data_array)
            if self.normalise_seconds:
                data_array = utils.normalise(data_array)
            extractor = FeatureExtractor(data_array)
            features_dset = if_f.create_dataset(key+'_features', shape = extractor.feature_array.shape, dtype='float')
            features_dset[:] = extractor.feature_array
            #print extractor.feature_array

    def _load_ictal_raw(self):
        self.f = h5py.File('ictal_file.hdf5', 'w')
        # only going to load a ndf with max of two seizures in it  - will fail with three!
        for i in xrange(self.annotations.shape[0]):
        #for i in xrange(1):
            fname = self.annotations.iloc[i,0]
            start = self.annotations.iloc[i,1]
            end   = self.annotations.iloc[i,2]
            #print fname, end , start

            ndf =  NDFLoader(self.dir_path+'ndf/'+fname, print_meta=True)
            ndf.load(8)
            ndf.glitch_removal(plot_glitches=False, print_output=True)
            ndf.correct_sampling_frequency()

            np.set_printoptions(precision=3, suppress = True)
            ictal_time = ndf.time[(ndf.time >= start) & (ndf.time <= end)]
            ictal_data = ndf.data[(ndf.time >= start) & (ndf.time <= end)]


            #print ictal_data.shape[0]
            #print ictal_time.shape
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
            #print fname, end , start

            ndf =  NDFLoader(self.dir_path+'ndf/'+fname, print_meta=True)
            ndf.load(8)
            ndf.glitch_removal(plot_glitches=False, print_output=True)
            ndf.correct_sampling_frequency()

            np.set_printoptions(precision=3, suppress = True)
            inter_ictal_time = ndf.time[(ndf.time <= start) | (ndf.time >= end)]
            inter_ictal_data = ndf.data[(ndf.time <= start) | (ndf.time >= end)]

            #print inter_ictal_data.shape[0]
            #print inter_ictal_time.shape
            try:
                fname_dset = iif.create_dataset(fname, (inter_ictal_data.shape[0],2), dtype='float')
            except:
                #print 'here'
                fname = fname+'_2'
                fname_dset = iif.create_dataset(fname, (inter_ictal_data.shape[0],2), dtype='float')

            fname_dset[:,0] = inter_ictal_time
            fname_dset[:,1] = inter_ictal_data



if __name__ == '__main__':
    n = Main()
