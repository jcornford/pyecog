import pickle
import os
import pythoncode.utils as utils

import numpy as np
import pandas as pd

from pythoncode.network_loader import SeizureData
from pythoncode.extrator import FeatureExtractor
from pythoncode.make_pdfs import plot_traces_hdf5


import matplotlib.pyplot as plt
import h5py

import utils


class Predictor():

    '''
    Todo:
    '''

    def __init__(self, clf_pickle_path, fs_dict_path='../pickled_fs_dictionary'):

        #self.fs_dict  = pickle.load(open(fs_dict_path,'rb'))
        #for key in self.fs_dict:
            #print key, self.fs_dict[key]
        pickle.load(open('/Volumes/LACIE SHARE/pickled_classifier','rb'))
        self.classifier = pickle.load(open(clf_pickle_path,'rb'))
        self.r_forest = self.classifier.r_forest
        self.r_forest_lda = self.classifier.r_forest_lda
        self.lda = self.classifier.lda

        print self.lda
        #print self.r_forest_lda


    def assess_states(self, raw_path = None, downsample_rate = None, savestring = 'example',
                      threshold = 65,
                      raw_load = True,
                      saved_path = None,
                      make_pdfs = True):
        self.raw_path = raw_path
        self.threshold = '65' # 'sureity' threshold
        self.savestring = savestring

        with h5py.File(raw_path , 'r') as hf:

            for key in hf.attrs.keys():
                print key, hf.attrs[key]
            print hf.items()

            for ndfkey in hf.keys():
                print ndfkey, 'is hf key'
                datadict = hf.get(ndfkey)

            for tid in datadict.keys():

                time = np.array(datadict[tid]['time'])
                data = np.array(datadict[tid]['data'])
                #print npdata.shape

            print data.shape

            index = data.shape[0]/ (5120/2)
            print index, 'is divded by 5120'

            data_array = np.reshape(data[:(5120/2)*index], (index,(5120/2),))
            print data_array.shape
            #plt.figure(figsize = (20,10))
            #plt.plot(data_array[40,:])

            #plt.show()
        self.data_array = data_array




        self.norm_data = utils.normalise(self.data_array)
        #self.norm_data = self.data_array
        self.norm_data = utils.filterArray(self.norm_data, window_size=7,order = 3)
        feature_obj = FeatureExtractor(self.norm_data)

        i_features = self.classifier.imputer.transform(feature_obj.feature_array)
        iss_features = self.classifier.std_scaler.transform(i_features)
        lda_iss_features = self.lda.transform(iss_features)

        np.set_printoptions(precision=3, suppress = True)

        #self.pred_table = self.r_forest.predict_proba(iss_features)*100
        #self.preds = self.r_forest.predict(iss_features)

        self.pred_table = self.r_forest_lda.predict_proba(lda_iss_features)*100
        self.preds = self.r_forest_lda.predict(lda_iss_features)

        self.predslist = list(self.preds) # why need this?
        self.predslist[self.predslist == 4] = 'Baseline'
        self.max_preds = np.max(self.pred_table, axis = 1)
        #print pred_table
        self.threshold_for_mixed = np.where(self.max_preds < int(self.threshold),1,0) # 1 when below
        #self._string_fun2()
        #self._write_to_excel()

        if make_pdfs:
            #path_to_create = '/Volumes/LaCie/Albert_ndfs/hdf5/pdfs/'+self.raw_path.split('/')[-1][:-5]
            #print path_to_create
            #os.mkdir(path_to_create,0755)
            self.plot_pdfs()

    def plot_pdfs(self):
        plot_traces_hdf5(self.norm_data,
                    labels = self.preds,
                    #savestring = '/Volumes/LaCie/Albert_ndfs/hdf5/pdfs/'+self.raw_path.split('/')[-1][:-5]+'/'+self.savestring,
                    #savestring = '/Volumes/LaCie/Albert_ndfs/training_data/rpdfs/'+self.savestring,
                    savestring  = '/Volumes/LaCie/Gabriele/pdfs_pred/',
                    prob_thresholds= self.threshold_for_mixed,
                    trace_len_sec= 5)

x = Predictor( clf_pickle_path = '/Volumes/LaCie/Albert_ndfs/pickled_classifier_20160302')
x = Predictor( clf_pickle_path ='/Volumes/LaCie/Albert_ndfs/pickled_classifier_t5dbs')


dirpath = '/Volumes/LaCie/Albert_ndfs/hdf5/'
dirpath = '/Volumes/LaCie/Albert_ndfs/training_data/raw_hdf5s/'

dirpath = '/Volumes/LaCie/Gabriele/hdf5s'  # this is for gabriele's stuff
makepdfs = True

#filepath = dirpath + 'M1453331811.hdf5'
for filename in os.listdir(dirpath):
        if filename.endswith('.hdf5'):
            filepath = os.path.join(dirpath, filename)

            x.assess_states(raw_path = filepath,
                            savestring = 'ndf_pred_'+filepath.split('/')[-1][:-5]+'_',
                            raw_load = False,
                            make_pdfs= makepdfs)


