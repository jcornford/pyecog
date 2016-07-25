import sys
import os
import multiprocessing

import traceback
import time
import logging

import h5py
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.preprocessing import Imputer, StandardScaler,normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv

from . import hmm

class FeaturePreProcesser():
    def __init__(self):
        pass

    def fit(self,X):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(X)
        X = self.imputer.transform(X)

        self.std_scaler = StandardScaler()
        self.std_scaler.fit(X)

    def fit_transform(self, X):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(X)
        X = self.imputer.transform(X)

        self.std_scaler = StandardScaler()
        self.std_scaler.fit(X)
        X = self.std_scaler.transform(X)

        return X
    def transform(self, X):
        X = self.imputer.transform(X)
        X = self.std_scaler.transform(X)
        return X

class Classifier():

    def __init__(self, library_path):
        with h5py.File(library_path, 'r+') as f:
            self.keys = list(f.keys())
            self.keys = np.random.permutation(self.keys)
            self.tids = [f[key].attrs['tid'] for key in self.keys]
            self.fname_array = np.vstack([np.vstack([fname for i in range(f[fname+'/features'].shape[0])]) for fname in self.keys])
            self.features = np.vstack( f[name+'/features'] for name in self.keys)
            self.labels   = np.vstack( f[name+'/labels'] for name in self.keys)
            #self.traces   = np.vstack( f[name+'/data'] for name in self.keys)

        self.cleaner = FeaturePreProcesser()
        self.cleaner.fit(self.features)
        self.features = self.cleaner.transform(self.features)

    def train(self):
        #self.emission_probs = self.local_cv(self.features, self.labels)
        print('Emission probs')
        print (self.emission_probs)
        self.transition_probs = hmm.get_state_transition_probs(self.labels)
        print ('Transition probs')
        print (self.transition_probs)
        #self.hm_model = hmm.make_hmm_model(self.emission_probs,self.transition_probs)


        counts = pd.Series(np.ravel(self.labels[:])).value_counts().values
        target_resample = (counts[1]*50,counts[1]*3)
        print (counts, target_resample)
        #target_resample = (5000,500)
        res_y, res_x = self.resample_training_dataset(self.labels, self.features,
                                                    sizes = target_resample)
        print (res_y.shape, res_x.shape) # incorect!
        #rf = RandomForestClassifier(n_jobs=-1, n_estimators= 2000, oob_score=True, bootstrap=True)
        #rf.fit(res_x, np.ravel(res_y))

    def make_hmm_model(self):
        self.emission_probs = self.local_cv(self.features, self.labels)
        print('Emission probs')
        print (self.emission_probs)
        self.transition_probs = hmm.get_state_transition_probs(self.labels)
        print ('Transition probs')
        print (self.transition_probs)
        #self.hm_model = hmm.make_hmm_model(self.emission_probs,self.transition_probs)

    def resample_training_dataset(self, labels, feature_array, sizes = (5000,500)):
        """
        Inputs:
            - labels
            - features
            - sizes: tuple, for each class (0,1,etc)m the number of training chunks you want.
            i.e for 500 seizures, 5000 baseline, sizes = (5000, 500), as 0 is baseline, 1 is Seizure
        Takes labels and features an
        """
        if len (labels.shape) == 1:
            labels = labels[:, None]

        resampled_labels = []
        resampled_features = []
        for i,label in enumerate(np.unique(labels.astype('int'))):
            class_inds = np.where(labels==label)[0]

            class_labels = labels[class_inds]
            class_features = feature_array[class_inds,:]

            if class_features.shape[0] < sizes[i]: # need to oversample
                class_features_duplicated = np.vstack([class_features for i in range(int(sizes[i]/class_features.shape[0]))])
                class_labels_duplicated  = np.vstack([class_labels for i in range(int(sizes[i]/class_labels.shape[0]))])
                n_extra_needed = sizes[i] - class_labels_duplicated.shape[0]
                extra_features = resample(class_features, n_samples =  n_extra_needed,random_state = 7, replace = False)
                extra_labels = resample(class_labels, n_samples =  n_extra_needed,random_state = 7, replace = False)

                boot_array  = np.vstack([class_features_duplicated,extra_features])
                boot_labels = np.vstack([class_labels_duplicated,extra_labels])

            else: # more need to undersample
                boot_array  = resample(class_features, n_samples =  sizes[i],random_state = 7, replace = False)
                boot_labels = resample(class_labels,   n_samples =  sizes[i],random_state = 7, replace = False)

            resampled_features.append(boot_array)
            resampled_labels.append(boot_labels)
        # stack both up...
        resampled_labels = np.vstack(resampled_labels)
        resampled_features = np.vstack(resampled_features)
        #print(pd.Series(resampled_labels[:,0]).value_counts())
        return resampled_labels, resampled_features

    def local_cv(self, X, y, nfolds = 5):
        '''
        - X has been imputed and cleaned etc...
        - Also, we are using default RF, should nest some
        tuning within it!

        Returns:
            - HMM model emission probs.
            - ToDO: Error (though HMM?)
            - ToDO: best RF parameters...
            - ToDO: potentially threshold
            - ToDO: best resampling?
        '''
        logging.info('Starting stratified cross validation with '+ str(nfolds)+ ' folds!' )

        emission_matrixes_list = [] # to hold the result from the k fold

        if len(y.shape) != 1:
            y = np.ravel(y)

        self.printProgress(0,nfolds, prefix = 'Cross validation:', suffix = 'Complete', barLength = 50)
        fold = 1
        skf = cv.StratifiedKFold(y, nfolds)
        for train_ix,test_ix in skf:
            logging.info('Jonny needs to log the CV resutls!')
            # index for the k fold
            X_train = X[train_ix,:]
            y_train = y[train_ix]
            X_test  = X[test_ix,:]
            y_test  = y[test_ix]

            # work out desired resampling
            y_counts = pd.Series(y_train).value_counts().values
            target_resample = (y_counts[1]*50,y_counts[1]*3)
            samp_y, samp_x = self.resample_training_dataset(y_train, X_train, sizes = target_resample)

            # train clf on resampled
            rf = RandomForestClassifier(n_jobs=-1, n_estimators= 100, oob_score=True, bootstrap=True)
            rf.fit(samp_x, np.ravel(samp_y))

            # see how well it did
            train_emitts = rf.predict(X_train)
            test_emitts  = rf.predict(X_test)
            binary_states = np.where(y_test==0,0,1) # this is for when multiple labels
            emission_matrix = hmm.get_state_emission_probs(test_emitts, binary_states)
            emission_matrixes_list.append(emission_matrix)
            self.printProgress(fold,nfolds, prefix = 'Cross validation:', suffix = 'Complete', barLength = 50)
            fold += 1


        ems = np.stack(emission_matrixes_list, axis = 2)
        mean_emitt_matrix = np.mean(ems, axis = 2)
        return mean_emitt_matrix

    def printProgress (self, iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : number of decimals in percent complete (Int)
            barLength   - Optional  : character length of bar (Int)
        """
        filledLength    = int(round(barLength * iteration / float(total)))
        percents        = round(100.00 * (iteration / float(total)), decimals)
        bar             = '█' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write('\n')
            sys.stdout.flush()