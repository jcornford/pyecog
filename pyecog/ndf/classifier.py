
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

from sklearn.utils import resample
from sklearn.preprocessing import Imputer, StandardScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn import metrics as sk_metrics
from sklearn.calibration import CalibratedClassifierCV

class Library():
    """Class for library, with features already calculated, to be used in training clf"""

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
        This requires the colnames of the dataframe...
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


class Classifier():
    """ We need: save, load, add data (with warm start). """

    def __init__(self, library_path):
        self.lib = Library(library_path)
        self.y_to_label_dict = {0: 'baseline', 1: 'ictal'}
        self.lib.ystr_ = pd.Series(self.lib.y).map(self.y_to_label_dict).values

        # self.prepare_library_data() - should be run

    def prepare_library_data(self):
        self.pyecog_scaler = FeaturePreProcesser(self.lib.X, self.lib.X_colnames)
        self.lib.X, self.lib.X_colnames = self.pyecog_scaler.transform()
        self.lib.X = self.pyecog_scaler.fit_transform_std_scaler(self.lib.X)

    def train_basic_rf_oob(self):
        """Just a temp method to get basic idea of performance"""
        y = pd.Series(self.lib.y)
        y_str = y.map(self.y_to_label_dict)

        target_counts = pd.Series(y_str).value_counts()
        print(target_counts)

        rf = RandomForestClassifier(n_estimators=600, oob_score=True,
                                    random_state=7, n_jobs=-1, class_weight=None,
                                    max_features='auto')  # when you have None, coastline massive
        rf.fit(self.lib.X, y_str)
        oob_preds = get_preds_from_rf_oob(rf)
        print(sk_metrics.classification_report(y_str, oob_preds))
        cm, n_cm = labelled_confusion_matrix(y_str, oob_preds, rf)
        print(cm, '\n', n_cm)
        self.feature_importances = get_feature_imps(rf, clf.lib.X_colnames)
        self.rf = rf
        # sk_metrics.f1_score(y_str,oob_preds,average='binary', pos_label='ictal')
        # sk_metrics.f1_score(y_str,oob_preds,average='binary', pos_label='baseline')

        pass