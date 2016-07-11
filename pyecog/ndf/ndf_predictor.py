from __future__ import print_function
import pickle
import os

import numpy as np
import h5py

import utils
from pyecog.light_code.extractor import FeatureExtractor
from pyecog.light_code.make_pdfs import plot_traces


class Predictor():


    '''
    CURRENTLY DEPRICATED

    '''

    def __init__(self, clf_pickle_path):
        self.classifier = pickle.load(open(clf_pickle_path,'rb'))
        self.r_forest = self.classifier.r_forest
        self.r_forest_lda = self.classifier.r_forest_lda
        self.lda = self.classifier.lda

    def load_traces(self, filepath):
        with h5py.File(filepath , 'r') as hf:

            self.data_array = data_array
            self.norm_data = utils.normalise(self.data_array)
            self.norm_data = utils.filterArray(self.norm_data, window_size=7, order = 3)

    def extract_features(self):
        self.feature_obj = FeatureExtractor(self.norm_data)

    def assess_states(self,
                      pdf_savedir = 'example',
                      savestring = 'demo',
                      threshold = 65,
                      make_pdfs = True,
                      format = 'png'):

        self.threshold = '65' # 'sureity' threshold
        self.savestring = savestring

        #self.norm_data = self.data_array

        i_features = self.classifier.imputer.transform(self.feature_obj.feature_array)
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
            plot_traces(self.norm_data, labels = self.preds, savepath = pdf_savedir+savestring,format_string = format)

    @ staticmethod
    def _normalise(series):
        a = np.min(series, axis=1)
        b = np.max(series, axis=1)
        return np.divide((series - a[:, None]), (b-a)[:,None])

#x = Predictor( clf_pickle_path = '/Volumes/LaCie/Albert_ndfs/pickled_classifier_20160302')

def main():
    x = Predictor( clf_pickle_path ='/Volumes/LaCie/Albert_ndfs/pickled_classifier_t5dbs')


    dirpath = '/Volumes/LaCie/Albert_ndfs/hdf5/'
    dirpath = '/Volumes/LaCie/Albert_ndfs/training_data/raw_hdf5s_/'

    #dirpath = '/Volumes/LaCie/Gabriele/hdf5s_id_7'  # this is for gabriele's stuff
    dirpath = '/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.14/hdf5s'
    makepdfs = False

    #filepath = dirpath + 'M1453331811.hdf5'
    for filename in os.listdir(dirpath):
            if filename.endswith('.hdf5'):
                filepath = os.path.join(dirpath, filename)

                x.assess_states(savestring = 'ndf_pred_'+filepath.split('/')[-1][:-5]+'_',
                                make_pdfs= makepdfs)

if __name__ == "__main__":
    main()
