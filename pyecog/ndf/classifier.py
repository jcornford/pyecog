
import sys
import os
import multiprocessing
try:
    import cPickle as pickle
except:
    import pickle

import traceback
import time
import logging

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Imputer, StandardScaler, normalize

from . import hmm_pyecog
from .classifier_utils import get_predictions_cross_val
from . import classifier_utils

class Library():
    """Class for library, with features already calculated, to be used for training clf"""

    def __init__(self, library_path):
        with h5py.File(library_path, 'r+') as f:
            self.keys = list(f.keys())
            self.keys = np.random.permutation(self.keys)

            self.fs_list = [f[key].attrs['fs'] for key in self.keys]
            self.tids = [f[key].attrs['tid'] for key in self.keys]
            self.fname_array = np.vstack(
                [np.vstack([fname for i in range(f[fname + '/features'].shape[0])]) for fname in self.keys])
            self.y = np.hstack(f[name + '/labels'] for name in self.keys)
            self.X_colnames = f[self.keys[0]].attrs['feature_col_names'].astype(str)
            self.X = np.vstack(f[name + '/features'] for name in self.keys)

            self.n_files = len(self.keys)
            self.n_examples = self.X.shape[0]

        self.df = None
        self.df_tidy = None

    def get_dataframe(self):
        self.df = pd.DataFrame(self.X, columns=self.X_colnames)
        self.df['ictal'] = self.y
        return self.df

    def get_tidy_dataframe(self):
        self.df_tidy = pd.melt(self.get_dataframe(), id_vars=['ictal'])
        return self.df_tidy

    def plot_tidy_dataframe(self, n_col=5):
        self.get_tidy_dataframe()
        pal = {1: "red", 0: "grey"}
        g = sns.FacetGrid(self.df_tidy, col="variable", hue='ictal', palette=pal, col_wrap=n_col, sharex=False,
                          sharey=False)
        g.map(sns.distplot, "value", hist=True, norm_hist=True, kde=False)
        g.add_legend()
        return g


class FeaturePreProcesser():
    def __init__(self, X, X_colnames):
        self.df = pd.DataFrame(X, columns=X_colnames)

    def transform(self):
        '''
        Here do the particular transformations to make data better behaved (logs etc).
        Requires the colnames of the feature dataframe...
        '''
        preprocessed_lib_df = self.df

        # handle powerband columns
        hzcols = preprocessed_lib_df.filter(like='H').columns
        preprocessed_lib_df[hzcols] = preprocessed_lib_df[hzcols].add(10 ** -6).apply(np.log)
        newlog_col_dict = {col: 'log_' + col for col in hzcols}
        preprocessed_lib_df.rename(columns=newlog_col_dict, inplace=True)

        # handle min-max
        preprocessed_lib_df['min'] = preprocessed_lib_df['min'].apply(np.abs).add(10 ** -6).apply(np.log)
        preprocessed_lib_df['max'] = preprocessed_lib_df['max'].apply(np.abs).add(10 ** -6).apply(np.log)
        min_max_col_dict = {'min': 'log_abs_min', 'max': 'log_max'}
        preprocessed_lib_df.rename(columns=min_max_col_dict, inplace=True)

        # handle kurtosis
        preprocessed_lib_df.loc[:, 'kurtosis'][preprocessed_lib_df['kurtosis'] > 50] = 50  # probably dont need
        # add three as kurtsis on ones etc is -3
        preprocessed_lib_df['kurtosis'] = preprocessed_lib_df['kurtosis'].add(3).add(10 ** -6).apply(np.log)
        preprocessed_lib_df.rename(columns={'kurtosis': 'log_kurtosis', 'bl_mean_d': 'log_bl_mean_d'}, inplace=True)

        return preprocessed_lib_df.values, preprocessed_lib_df.columns

    def fit_transform_std_scaler(self, X):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(X)
        X = self.imputer.transform(X)

        self.std_scaler = StandardScaler()
        self.std_scaler.fit(X)
        X = self.std_scaler.transform(X)
        return X

    def fit_std_scaler(self, X):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(X)
        X = self.imputer.transform(X)

        self.std_scaler = StandardScaler()
        self.std_scaler.fit(X)

    def transform_std_scaler(self, X):
        X = self.imputer.transform(X)
        X = self.std_scaler.transform(X)
        return X

    def transform_features(self, X, cols):
        df = pd.DataFrame(X, columns = cols)
        preprocessed_lib_df = df

        # handle powerband columns
        hzcols = preprocessed_lib_df.filter(like='H').columns
        preprocessed_lib_df[hzcols] = preprocessed_lib_df[hzcols].add(10 ** -6).apply(np.log)
        newlog_col_dict = {col: 'log_' + col for col in hzcols}
        preprocessed_lib_df.rename(columns=newlog_col_dict, inplace=True)

        # handle min-max
        preprocessed_lib_df['min'] = preprocessed_lib_df['min'].apply(np.abs).add(10 ** -6).apply(np.log)
        preprocessed_lib_df['max'] = preprocessed_lib_df['max'].apply(np.abs).add(10 ** -6).apply(np.log)
        min_max_col_dict = {'min': 'log_abs_min', 'max': 'log_max'}
        preprocessed_lib_df.rename(columns=min_max_col_dict, inplace=True)

        # handle kurtosis
        preprocessed_lib_df.loc[:, 'kurtosis'][preprocessed_lib_df['kurtosis'] > 50] = 50  # probably dont need
        # add three as kurtsis on ones etc is -3
        preprocessed_lib_df['kurtosis'] = preprocessed_lib_df['kurtosis'].add(3).add(10 ** -6).apply(np.log)
        preprocessed_lib_df.rename(columns={'kurtosis': 'log_kurtosis', 'bl_mean_d': 'log_bl_mean_d'}, inplace=True)

        X = preprocessed_lib_df.values

        X = self.imputer.transform(X)
        X = self.std_scaler.transform(X)
        return X

class ClassificationAlgorithm():
    '''Class to combine a descriminative algorithm with a HMM model'''

    def __init__(self, descriminative_model_object, hmm_class_def):
        self.descriminative_model = descriminative_model_object
        self.hmm = hmm_class_def

    def fit(self, X, y, downsample_bl_factor=1, upsample_seizure_factor=1):
        """
        Fit descrim algo and then HMM.

        Todo: handle resampling and class weights for descrim classifier.
        """

        if downsample_bl_factor == 1 and upsample_seizure_factor == 1:
            Xclf, yclf = X, y
            #print('here')
        else:
            counts = pd.Series(y).value_counts()
            target_resample = (int(counts[0] / downsample_bl_factor),
                               int(counts[1] * upsample_seizure_factor))
            Xclf, yclf = classifier_utils.resample_training_dataset(X, y, target_resample)
            print('resample', target_resample)

        self.fit_hmm(X, y)
        self.descriminative_model.fit(Xclf, yclf) # class weights to be added here too
        print('Clf stage 3 complete: descriminative_model fit')

    def fit_hmm(self, X, y):
        self.hmm.A = self.hmm.get_state_transition_probs(y)
        print('Clf stage 1 complete: HMM hidden state transition probabilities')

        if isinstance(self.hmm, hmm_pyecog.HMMBayes):
            print('Clf stage 2 complete: no emission probabilities needed')
            pass

        if isinstance(self.hmm, hmm_pyecog.HMM):
            cv_preds = get_predictions_cross_val(X, y, self.descriminative_model)
            # the function get_predictions_cross_val returns (labels, probabilties)
            self.hmm.phi_mat = self.hmm.get_state_emission_probs(y, cv_preds[0])
            print('Clf stage 2 complete: HMM emission probabilities')

    @staticmethod
    def apply_threshold(posterior, threshold):
        postive_class_predictions = posterior[:, 1] > threshold
        return postive_class_predictions.astype(int)

    def predict(self, X, threshold):
        if isinstance(self.hmm, hmm_pyecog.HMM):
            # hmm expects labels
            class_labels = self.descriminative_model.predict(X)
            posterior = self.hmm.forward_backward(class_labels.T).T
        else:
            # hmm expects postive class probabilites
            class_probs = self.descriminative_model.predict_proba(X)
            posterior = self.hmm.forward_backward(class_probs.T).T
        return self.apply_threshold(posterior, threshold)

def load_classifier(filepath):
    f = open(filepath, 'rb')
    clf = pickle.load(f)
    return clf

class Classifier():

    """ We need: save, load, add data (with warm start). """

    def __init__(self, library_path):
        self.lib = Library(library_path)
        self.pyecog_scaler = None
        self.y_to_label_dict = {0: 'baseline', 1: 'ictal'}
        self.lib.ystr_ = pd.Series(self.lib.y).map(self.y_to_label_dict).values

    def save(self, fname):
        fname = fname if fname.endswith('.p') else fname+'.p'
        f = open(fname,'wb')
        pickle.dump(self,f)

    @property
    def label_value_counts(self):
        return pd.Series(self.lib.y).value_counts()
    @property
    def y(self):
        return self.lib.y
    @property
    def X(self):
        return self.lib.X

    @property
    def algo(self):
        return self.__algo
    @algo.setter
    def algo(self,algo_object):
        """Algo object an be anything that has predict and fit methods"""
        self.__algo = algo_object

    def train(self, downsample_bl_factor=1, upsample_seizure_factor=1):
        self.algo.fit(self.X,self.y,
                      downsample_bl_factor=downsample_bl_factor,
                      upsample_seizure_factor=upsample_seizure_factor)

    def predict(self, X, threshold):
        return self.algo.predict(X,threshold)


    def preprocess_features(self):
        self.pyecog_scaler = FeaturePreProcesser(self.lib.X, self.lib.X_colnames)
        self.lib.X, self.lib.X_colnames = self.pyecog_scaler.transform()
        self.lib.X = self.pyecog_scaler.fit_transform_std_scaler(self.lib.X)


    def predict_directory(self,prediction_dir,
                            output_csv_filename,
                            overwrite_previous_predicitions = True,
                            posterior_thresh = 0.5,
                            gui_object = None):

        classifier_utils.predict_dir(prediction_dir,
                                     output_csv_filename,
                                     classfier_object = self,
                                     overwrite_previous_predicitions = overwrite_previous_predicitions,
                                     posterior_thresh = posterior_thresh,
                                     gui_object=gui_object)




