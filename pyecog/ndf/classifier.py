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

from sklearn.utils import resample
from sklearn.preprocessing import Imputer, StandardScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn import metrics
try:
    from . import hmm
except:
    print('Problem import hmm module - presumably do not have pomegranate insalled?')

from .h5loader import H5File

def load_classifier(filepath):
    f = open(filepath, 'rb')
    clf = pickle.load(f)
    return clf

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
    ''' we drop the actual data... so this is a lot smaller than the library used to train it '''
    def __init__(self, library_path):
        with h5py.File(library_path, 'r+') as f:
            self.keys = list(f.keys())
            self.keys = np.random.permutation(self.keys)
            self.tids = [f[key].attrs['tid'] for key in self.keys]
            self.fname_array = np.vstack([np.vstack([fname for i in range(f[fname+'/features'].shape[0])]) for fname in self.keys])
            self.features = np.vstack( f[name+'/features'] for name in self.keys)
            self.labels   = np.hstack( f[name+'/labels'] for name in self.keys)
            self.feature_names = f[self.keys[0]].attrs['col_names'].astype(str)

        self.cleaner = FeaturePreProcesser()
        self.cleaner.fit(self.features)
        self.features = self.cleaner.transform(self.features)

    def save(self, fname = 'pickled_clf' ):
        fname = fname if fname.endswith('.p') else fname+'.p'
        f = open(fname,'wb')
        pickle.dump(self,f)

    def load(self, fname):
        print('This is a function of the module, not a particular classifier! (Naturally will not exist)')
        return 0

    def train(self, downsample_bl_by_x = 3,
                    upsample_seizure_by_x = 1,
                    ntrees=800, n_cores = -1,
                    n_emission_prob_cvfolds = 3):
        
        if upsample_seizure_by_x != 1:
            print('Warning: you are upsampling minority class - oob error cannot be trusted' )
        
        counts = pd.Series(np.ravel(self.labels[:])).value_counts().values
        target_resample = (int(counts[0]/self.downsample_bl_factor),counts[1]*self.upsample_seizure_factor)

        logging.info('Training classifier: ')
        logging.info('Resampling training - upsampling seizure seconds, downsampling baseline periods...')
        res_y, res_x = self.resample_training_dataset(self.labels, self.features,
                                                      sizes = target_resample)

        logging.info('Training Random Forest on resampled data')
        self.clf =  RandomForestClassifier(n_jobs=n_cores, n_estimators= ntrees, oob_score=True, bootstrap=True)
        self.clf.fit(res_x, np.ravel(res_y))

        logging.info('Getting HMM params')
        self.make_hmm_model(n_emission_prob_cvfolds) # uses normal data, you should pass downsampling params?

        print ('********* oob results on resampled data *******')
        self.oob_preds = np.round(self.clf.oob_decision_function_[:,1])
        print('ROC_AUC score: '+str(metrics.roc_auc_score(np.ravel(res_y), self.clf.oob_decision_function_[:,1])))
        print('Recall: '+str(metrics.recall_score(np.ravel(res_y), self.oob_preds)))
        print('F1: '+str(metrics.f1_score(np.ravel(res_y), self.oob_preds)))
        print(metrics.classification_report(np.ravel(res_y),self.oob_preds))

        self.feature_weightings = sorted(zip(self.clf.feature_importances_, self.feature_names),reverse = True)

    def make_hmm_model(self, n_emission_prob_folds = 3):
        # how to get these bad boys...
        self.emission_probs = self.get_cross_validation_emission_probs(self.features,
                                                                       self.labels,
                                                                       nfolds = n_emission_prob_folds)
        print('Emission probs')
        print (self.emission_probs)
        self.transition_probs = hmm.get_state_transition_probs(self.labels)
        print ('Transition probs')
        print (self.transition_probs)
        self.hm_model = hmm.make_hmm_model(self.emission_probs,self.transition_probs)
        
    def get_cross_validation_emission_probs(self, X, y, nfolds = 3):
        '''
        - X has been imputed and cleaned etc...
        - Also, we are using default RF, should nest some
        tuning within it! - too long

        Returns:
            - HMM model emission probs.
            - ToDO: Error (though HMM?)
            - ToDO: best RF parameters...
            - ToDO: potentially threshold
            - ToDO: best resampling?
        '''
        logging.info('Starting stratified cross validation with '+ str(nfolds)+ ' folds!' )

        emission_matrices_list = [] # to hold the result from the k fold

        if len(y.shape) != 1: # ravel labels if they need it
            y = np.ravel(y)

        self.printProgress(0,nfolds, prefix = 'Getting hmm emission probs:', suffix = 'Complete', barLength = 30)
        fold = 1

        skf = cv.StratifiedKFold(y, nfolds, random_state= 7)
        precision, recall, f1 = [],[],[]
        oob_precision, oob_recall, oob_f1 = [],[],[]
        for train_ix,test_ix in skf:

            # index for the k fold
            X_train = X[train_ix,:]
            y_train = y[train_ix]
            X_test  = X[test_ix,:]
            y_test  = y[test_ix]

            # work out desired resampling - this needs to be inheritied
            y_counts = pd.Series(y_train).value_counts().values
            target_resample = (y_counts[1]*50,y_counts[1])
            samp_y, samp_x = self.resample_training_dataset(y_train, X_train, sizes = target_resample)

            # train clf on resampled
            rf = RandomForestClassifier(n_jobs=-1, n_estimators= 500, oob_score=True, bootstrap=True)
            rf.fit(samp_x, np.ravel(samp_y))

            # see how well it did
            train_emitts = rf.predict(X_train)
            test_emitts  = rf.predict(X_test)

            # this is the test set
            p_score = metrics.precision_score(y_test, test_emitts)
            precision.append(p_score)
            recall_score = metrics.recall_score(y_test, test_emitts)
            recall.append(recall_score)
            f1_score = metrics.f1_score(y_test, test_emitts)
            f1.append(f1_score)


            # this is the oob on the resampled
            oob_preds = np.round(rf.oob_decision_function_[:,1]) # round the 1 column for 1 issezi. 0 is baseline
            oob_p_score = metrics.precision_score(np.ravel(samp_y), oob_preds)
            oob_precision.append(oob_p_score)
            oob_recall_score = metrics.recall_score(np.ravel(samp_y), oob_preds)
            oob_recall.append(oob_recall_score)
            oob_f1_score = metrics.f1_score(np.ravel(samp_y), oob_preds)
            oob_f1.append(oob_f1_score)


            binary_states = np.where(y_test==0,0,1) # this is for when multiple labels
            emission_matrix = hmm.get_state_emission_probs(test_emitts, binary_states)
            emission_matrices_list.append(emission_matrix)
            self.printProgress(fold,nfolds, prefix = 'Getting hmm emission probs:', suffix = 'Complete', barLength = 30)
            fold += 1

        logging.info('Precision: ' + str(precision))
        logging.info('Recall: ' + str(recall))
        logging.info('F1: ' + str(f1))

        logging.info('OOB is on the resampled data:')
        logging.info('oob_Precision: ' + str(oob_precision))
        logging.info('oob_Recall: ' + str(oob_recall))
        logging.info('oob_F1: ' + str(oob_f1))

        #print( 'Mean precision'+ str(np.mean(precision)) )
        #print( 'Mean recall: '+ str(np.mean(recall)) )
        #print( 'Mean f1: '+str(np.mean(f1)) )

        ems = np.stack(emission_matrices_list, axis = 2)
        mean_emitt_matrix = np.mean(ems, axis = 2)
        return mean_emitt_matrix

    def estimate_clf_error(self, nfolds = 3):
        #self.printProgress(0,nfolds, prefix = 'Cross validation:', suffix = 'Complete', barLength = 50)
        #fold = 1
        print ('Running '+str(nfolds)+'-fold cross validation to estimate classifier performance:')
        skf = cv.StratifiedKFold(np.ravel(self.labels), nfolds, random_state= 7)
        precision, recall, f1 = [],[],[]
        map_precision, map_recall, map_f1 = [],[],[]

        for train_ix,test_ix in skf:
            X_train = self.features[train_ix,:]
            y_train = self.labels[train_ix]
            X_test  = self.features[test_ix,:]
            y_test  = self.labels[test_ix]

            y_counts = pd.Series(y_train).value_counts().values
            target_resample = (y_counts[1]*50,y_counts[1]) #ie. 50* seizures n
            samp_y, samp_x = self.resample_training_dataset(y_train, X_train, sizes = target_resample)

            print('running balanced, undersampling only')
            rf = RandomForestClassifier(n_jobs=-1, n_estimators= 500,
                                        oob_score=True, bootstrap=True,
                                        class_weight= 'balanced')
            rf.fit(samp_x, np.ravel(samp_y))
            #rf.fit(X_train, np.ravel(y_train))

            # now you need the hmm params! nest this shit up
            train_emission_probs = self.get_cross_validation_emission_probs(X_train, y_train,nfolds = 4)
            train_transition_probs = hmm.get_state_transition_probs(y_train)
            train_hm_model = hmm.make_hmm_model(train_emission_probs,train_transition_probs)

            # predict the y_test with the rf trained on the resampled data:
            test_emissions = rf.predict(X_test)
            #viterbi_decoded = train_hm_model.predict(test_emissions, algorithm='viterbi')[1:]
            logp, path = train_hm_model.viterbi(test_emissions)
            viterbi_decoded = np.array([int(state.name) for idx, state in path[1:]])

            p_score = metrics.precision_score(y_test, viterbi_decoded)
            precision.append(p_score)
            recall_score = metrics.recall_score(y_test, viterbi_decoded)
            recall.append(recall_score)
            f1_score = metrics.f1_score(y_test, viterbi_decoded)
            f1.append(f1_score)
            print ('vit Precision: '+ str(p_score))
            print ('vit Recall: '+ str(recall_score))
            print ('vit F1: '+ str(f1_score))

            try:
                # run forward backward as well
                fb = train_hm_model.predict(test_emissions, algorithm='map')
                p_score = metrics.precision_score(y_test, fb)
                map_precision.append(p_score)
                recall_score = metrics.recall_score(y_test, fb)
                map_recall.append(recall_score)
                f1_score = metrics.f1_score(y_test, fb)
                map_f1.append(f1_score)
                #print ('fb Precision: '+ str(p_score))
                #print ('fb Recall: '+ str(recall_score))
                #print ('fb F1: '+ str(f1_score))
                fb = True
            except:
                fb = False
                pass


        clf_precision = np.mean(precision)
        clf_recall = np.mean(recall)
        clf_f1 = np.mean(f1)

        print ('Mean vit precision: '+str(clf_precision))
        print ('Mean vit recall:    '+ str(clf_recall))
        print ('Mean vit f1:        '+ str(clf_f1))

        if fb:
            map_clf_precision = np.mean(map_precision)
            map_clf_recall = np.mean(map_recall)
            map_clf_f1 = np.mean(map_f1)

            print ('Mean fb precision: '+str(map_clf_precision))
            print ('Mean fb recall:    '+ str(map_clf_recall))
            print ('Mean fb f1:        '+ str(map_clf_f1))

    def resample_training_dataset(self, labels, feature_array, sizes = (5000,500)):
        """
        Inputs:
            - labels
            - features
            - sizes: tuple, for each class (0,1,etc)m the number of training chunks you want.
            i.e for 500 seizures, 5000 baseline, sizes = (5000, 500), as 0 is baseline, 1 is Seizure
        Takes labels and features an

        WARNING: Up-sampling target class prevents random forest oob from being accurate.
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

            elif class_features.shape[0] > sizes[i]: # need to undersample
                boot_array  = resample(class_features, n_samples =  sizes[i],random_state = 7, replace = False)
                boot_labels = resample(class_labels,   n_samples =  sizes[i],random_state = 7, replace = False)

            elif class_features.shape[0] == sizes[i]:
                logging.debug('label '+str(label)+ ' had exact n as sample, doing nothing!')
                boot_array  = class_features
                boot_labels = class_labels
            else:
                print(class_features.shape[0], sizes[i])
                print ('fuckup')
            resampled_features.append(boot_array)
            resampled_labels.append(boot_labels)
        # stack both up...
        resampled_labels = np.vstack(resampled_labels)
        resampled_features = np.vstack(resampled_features)

        logging.debug('Original label counts: '+str(pd.Series(labels[:,0]).value_counts()))
        logging.debug('Resampled label counts: '+str(pd.Series(resampled_labels[:,0]).value_counts()))

        return resampled_labels, resampled_features


    def tune_hyperparameters(self):
        pass
        # just a placeholderfor now - code is in ipython notebook
        # we want to tune class weights?
        # also depth, leaves and n features?


    def predict_dir(self, prediction_dir, excel_sheet = 'clf_predictions.csv'):
        files_to_predict = os.listdir(prediction_dir)
        print('Looking through '+ str(len(files_to_predict)) + ' file for seizures:')
        for fname in files_to_predict:
            fpath = os.path.join(prediction_dir,fname)

            try:
                with h5py.File(fpath, 'r+') as f:
                    group = f[list(f.keys())[0]]
                    tid = group[list(group.keys())[0]]
                    pred_features = tid['features'][:]

                logging.info(str(fname) + ' now predicting!: ')
                pred_features = self.cleaner.transform(pred_features)
                pred_y_emitts = self.clf.predict(pred_features)
                logp, path = self.hm_model.viterbi(pred_y_emitts)
                vit_decoded_y = np.array([int(state.name) for idx, state in path[1:]])

                if sum(vit_decoded_y):
                    name_array = np.array([fname for i in range(vit_decoded_y.shape[0])])
                    #print (name_array.shape)
                    pred_sheet = self.make_excel_spreadsheet([name_array,vit_decoded_y])
                    #print(pred_sheet.head())
                    if not os.path.exists(excel_sheet):
                        pred_sheet.to_csv(excel_sheet,index = False)
                    else:
                        with open(excel_sheet, 'a') as f:
                            pred_sheet.to_csv(f, header=False, index = False)


            except KeyError:
                logging.error(str(fname) + ' did not contain any features! Skipping')

            else:
                pass
                #print ('no seizures')

        print('Re - ordering spreadsheet by date')
        self.reorder_prediction_csv(excel_sheet)
        print ('Done')

    def reorder_prediction_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        datetimes = []
        for string in df.Filename:
            ymd = string.split('_')[1]
            h = string.split('_')[2]
            m = string.split('_')[3]
            x = ymd+h+m
            datetimes.append(pd.to_datetime(x, format = '%Y-%m-%d%H%M'))
        df['datetime'] = pd.Series(datetimes)
        reordered_df = df.sort_values(by=['datetime', 'Start']).drop('datetime', axis = 1)
        reordered_df.to_csv(csv_path)

    def make_excel_spreadsheet(self, to_stack = [], columns_list = ['Name', 'Pred'], verbose = False):
        '''
        to_stack list:
            - first needs to be the name array
            - second needs to be the predictions
        '''
        for i,array in enumerate(to_stack):
            if len(array.shape) == 1:
                to_stack[i] = array[:,None]
        data = np.hstack([to_stack[0],to_stack[1].astype(int)])
        df = pd.DataFrame(data, columns = columns_list)

        # Make time index...
        df['f_index'] = df.groupby(by = columns_list[0]).cumcount()
        # assuming 1 hour per filename, first assert all files have same
        # number of
        x = df[str(columns_list[0])].value_counts()
        n_chunks = x.max()
        try:
            assert x.isin([n_chunks]).all()
        except:
            if verbose:
                print('Warning: Files not all the same length \n'+str(x))
            else:
                print('Warning: Files not all the same length, run again with verbose flag True for more detail')
        sec_per_chunk = 3600/n_chunks
        df['start_time'] = df['f_index']*sec_per_chunk
        df['end_time'] = (df['f_index']+1)*sec_per_chunk

        # okay, now find the seizures!
        seizure_indexes = df.groupby(columns_list[1]).indices['1']
        seizures_idx_tups = []
        start = None
        for i,t  in enumerate(seizure_indexes):
            if start is None:
                start = t
            try:
                if seizure_indexes[i+1] - t != 1:
                    end = t
                    seizures_idx_tups.append((start,end))
                    start = None
            except IndexError:
                end = t
                seizures_idx_tups.append((start,end))
                start = None

        # make the Dataframe for predicted seizures
        df_rows = []
        for tup in seizures_idx_tups:
            name = df.ix[tup[0],columns_list[0]]
            start_time = df.ix[tup[0],'start_time']
            end_time = df.ix[tup[1],'end_time']
            row = pd.Series([name,start_time,end_time],
                            index = ['Filename','Start', 'End'])
            df_rows.append(row)
        excel_sheet = pd.DataFrame(df_rows)
        return excel_sheet

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