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
from sklearn.calibration import CalibratedClassifierCV


from .h5loader import H5File
from . import utils
from . import hmm_pyecog

try:
    from line_profiler import LineProfiler
    # decorator needed when profiling
    def lprofile():
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)

                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner
except:
    print('failed to load lineprofile')
    pass

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
        self.emission_probs = None
        self.transition_probs = None

        with h5py.File(library_path, 'r+') as f:
            self.keys = list(f.keys())
            self.keys = np.random.permutation(self.keys)
            self.tids = [f[key].attrs['tid'] for key in self.keys]
            self.fname_array = np.vstack([np.vstack([fname for i in range(f[fname+'/features'].shape[0])]) for fname in self.keys])
            self.features = np.vstack( f[name+'/features'] for name in self.keys)
            self.labels   = np.hstack( f[name+'/labels'] for name in self.keys)
            self.feature_names = f[self.keys[0]].attrs['feature_col_names'].astype(str)

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
                    n_emission_prob_cvfolds = 3,
                    pyecog_hmm = True,
                    calc_emissions = True,
                    rf_weights = None,
                    calibrate = False):
        '''

        Args:
            downsample_bl_by_x:
            upsample_seizure_by_x:
            ntrees:
            n_cores:
            n_emission_prob_cvfolds:
            pyecog_hmm: NOT NEEDED ANYMORE!
            calc_emissions:
            rf_weights:
            calibrate:

        Returns:

        '''
        self.ntrees = ntrees
        self.downsample_bl_factor    = downsample_bl_by_x
        self.upsample_seizure_factor = upsample_seizure_by_x

        if upsample_seizure_by_x != 1:
            print('Warning: you are upsampling minority class - oob error cannot be trusted' )
        
        counts = pd.Series(np.ravel(self.labels[:])).value_counts().values
        target_resample = (int(counts[0]/self.downsample_bl_factor),counts[1]*self.upsample_seizure_factor)

        logging.info('Training classifier: ')
        logging.info('Resampling training - upsampling seizure seconds, downsampling baseline periods...')
        res_y, res_x = self.resample_training_dataset(self.labels, self.features,
                                                      sizes = target_resample)

        logging.info('Training Random Forest on resampled data')
        print("Training Random Forest on all data:")
        print('Resampling '+str(counts)+ ' to '+str(target_resample)+ ' with upsample seizures factor: ' +\
              str(self.upsample_seizure_factor) + ', and downsample bl factor: '+ str(self.downsample_bl_factor))
        print(rf_weights)
        self.rf =  RandomForestClassifier(n_jobs=n_cores, n_estimators= ntrees, oob_score=True, bootstrap=True, class_weight=rf_weights)
        if calibrate:
            print('running calibration')
            self.platt_rf = CalibratedClassifierCV(self.rf, method = 'sigmoid', cv=3)
            self.rf = self.platt_rf.fit(res_x,np.ravel(res_y)) # sample_weight : array-like, shape = [n_samples] or None
        else:
            self.rf.fit(res_x, np.ravel(res_y))

        '''
        print ('********* Out-of-bag results on resampled data *******')
        self.oob_preds = np.round(self.rf.oob_decision_function_[:,1])
        print('ROC_AUC score: '+str(metrics.roc_auc_score(np.ravel(res_y), self.rf.oob_decision_function_[:,1])))
        print('Recall: '+str(metrics.recall_score(np.ravel(res_y), self.oob_preds)))
        print('F1: '+str(metrics.f1_score(np.ravel(res_y), self.oob_preds)))
        print(metrics.classification_report(np.ravel(res_y),self.oob_preds))

        print ('Printing feature importances for classifier:')
        self.feature_weightings = sorted(zip(self.rf.feature_importances_, self.feature_names),reverse = True)
        for imp_name_tup in self.feature_weightings:
            print(str(imp_name_tup[1])+' : '+str(np.round(imp_name_tup[0],4)))
        '''
        if pyecog_hmm:
            logging.info(' Now getting HMM parameters using pyecog hmm class')
            print('Now getting HMM parameters using pyecog hmm class')
            self.make_pyecog_hmm_model(n_emission_prob_cvfolds, calc_emissions = calc_emissions) # initially keep for the non prob version of self rolled f/b stuff

    def make_pyecog_hmm_model(self, n_emission_prob_folds = 3, calc_emissions = True):
        # how to get these bad boys...
        if calc_emissions:
            self.emission_probs = self.get_cross_validation_emission_probs(self.features,
                                                                           self.labels,
                                                                           nfolds = n_emission_prob_folds)
            print('Emission probs')
            print (self.emission_probs)
        self.transition_probs = hmm_pyecog.get_state_transition_probs(self.labels)
        print ('Transition probs')
        print (self.transition_probs)
        self.hm_model =  hmm_pyecog.HMM(self.transition_probs)
        
    def get_cross_validation_emission_probs(self, X, y, nfolds = 3):
        '''
        - X has been imputed and cleaned etc...
        - Also, we are using default RF, should nest some
        Returns:
            - HMM model emission probs.
        '''
        logging.info('Starting stratified cross validation with '+ str(nfolds)+ ' folds!' )
        print('Getting HMM Emission probabilites:')
        if len(y.shape) != 1: # ravel labels if they need it
            y = np.ravel(y)

        #self.printProgress(0,nfolds, prefix = 'Getting hmm emission probs:', suffix = 'Complete', barLength = 30)

        skf = cv.StratifiedKFold(y, nfolds, random_state= 7)
        fold = 1
        emission_matrices_list = [] # to hold the result from the k fold
        for train_ix,test_ix in skf:

            # index for the k fold
            X_train = X[train_ix,:]
            y_train = y[train_ix]
            X_test  = X[test_ix,:]
            y_test  = y[test_ix]

            # work out desired resampling - this needs to be inheritied
            y_counts = pd.Series(y_train).value_counts().values

            # use the correct resampling with the same ratio as found in the clf main
            target_resample = (int(y_counts[0]/self.downsample_bl_factor),y_counts[1]*self.upsample_seizure_factor)
            print('CV fold:'+str(fold)+' . Resampling '+str(y_counts)+ ' to '+str(target_resample)+ \
                  ' with upsample seizures factor: ' + str(self.upsample_seizure_factor) + ',and downsample bl factor: '+ str(self.downsample_bl_factor))
            samp_y, samp_x = self.resample_training_dataset(self.labels, self.features, sizes = target_resample)

            # train clf on resampled
            rf = RandomForestClassifier(n_jobs=-1, n_estimators= self.ntrees, oob_score=False, bootstrap=True)
            rf.fit(samp_x, np.ravel(samp_y))

            # now get the emissions
            test_emitts  = rf.predict(X_test)

            # work out emission probabilities from the difference between annotations and emissions
            binary_states = np.where(y_test==0,0,1) # this is for when multiple labels
            emission_matrix = hmm_pyecog.get_state_emission_probs(test_emitts, binary_states)
            print('CV fold: '+str(fold)+ ' emission matrix:')
            print(emission_matrix)
            emission_matrices_list.append(emission_matrix)
            #self.printProgress(fold,nfolds, prefix = 'Getting hmm emission probs:', suffix = 'Complete', barLength = 30)
            fold += 1

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
            train_transition_probs = hmm_pyecog.get_state_transition_probs(y_train)

            train_hm_model =  hmm_pyecog.HMM(train_transition_probs)


            # predict the y_test with the rf trained on the resampled data:
            test_emissions = rf.predict(X_test)
            #viterbi_decoded = train_hm_model.predict(test_emissions, algorithm='viterbi')[1:]
            posterior = self.hm_model.forward_backward(test_emissions, phi_mat = train_emission_probs)
            decoded_y = posterior[1,:]>self.posterior_thresh
            viterbi_decoded = decoded_y

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


    def run_clf_and_hmm_on_features_pyecog(self, pred_features, use_probs = False):
        """ this method relies on rolled own version...

        time is 75% on pred_y_emits and 25% on forward backward (for else)

        """
        pred_features = self.cleaner.transform(pred_features)

        if use_probs: # x is a 2d vector of p(zt|xt)
            pred_y_probs = self.rf.predict_proba(pred_features).T # expecting cols to be timepoints
            posterior = self.hm_model.forward_backward(pred_y_probs, phi_mat=None)
        else: # x is assumed to be a 1d vector of emissions and phi mat is cross validation?
            pred_y_emitts = self.rf.predict(pred_features)
            #print('running clf on hmm features pyecog')
            posterior = self.hm_model.forward_backward(pred_y_emitts, phi_mat = self.emission_probs)
        decoded_y = posterior[1,:]>self.posterior_thresh
        return decoded_y

    #@lprofile()
    def predict_dir(self, prediction_dir,
                    output_csv_filename ='clf_predictions.csv',
                    overwrite_previous_predicitions = True,
                    pyecog_hmm = True,
                    use_probs = False,
                    posterior_thresh = 0.5,
                    called_from_gui = False):
        '''
        Args:
            prediction_dir:
            output_csv_filename:
            overwrite_previous_predicitions:
            pyecog_hmm:  NOT NEEDED ANYMORE!
            use_probs:
            posterior_thresh:
            called_from_gui:

        Returns:
        '''
        if not output_csv_filename.endswith('.csv'):
            output_csv_filename = output_csv_filename + '.csv'
        self.posterior_thresh = posterior_thresh
        files_to_predict = [os.path.join(prediction_dir,f) for f in os.listdir(prediction_dir) if not f.startswith('.') if f.endswith('.h5') ]
        l = len(files_to_predict)-1

        print('Looking through '+ str(len(files_to_predict)) + ' files for seizures:')

        if overwrite_previous_predicitions:
            try:
                os.remove(output_csv_filename)
            except:
                pass


        self.printProgress(0,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        skip_n =  0
        pred_count = 0
        all_predictions_df = None

        for i,fpath in enumerate(files_to_predict):
            full_fname = str(os.path.split(fpath)[1]) # this has more than one tid
            try:
                with h5py.File(fpath, 'r+') as f:
                    group = f[list(f.keys())[0]]
                    for tid_no in list(group.keys()):
                        tid_no = str(tid_no)
                        tid = group[tid_no]
                        pred_features = tid['features'][:]
                        logging.info(full_fname + ' now predicting tid '+tid_no)
                        if use_probs: # we are going to change this towards the probabilities using forward backward...
                            posterior_y = self.run_clf_and_hmm_on_features_pyecog(pred_features, use_probs = True)
                        else: # use the CV emission probs
                            posterior_y = self.run_clf_and_hmm_on_features_pyecog(pred_features)

                        if sum(posterior_y):
                            fname = full_fname.split('[')[0]+'['+tid_no+'].h5'
                            name_array = np.array([fname for i in range(posterior_y.shape[0])])
                            prediction_df = self.make_prediction_dataframe_rows_from_chunk_labels(tid_no, to_stack = [name_array, posterior_y], )
                            pred_count += prediction_df.shape[0]
                            if all_predictions_df is None:
                                all_predictions_df = prediction_df
                            else:
                                all_predictions_df = all_predictions_df.append(prediction_df, ignore_index=True, )

                            if all_predictions_df.shape[0] > 50:
                                if not os.path.exists(output_csv_filename):
                                    all_predictions_df.to_csv(output_csv_filename, index = False)
                                else:
                                    with open(output_csv_filename, 'a') as f2:
                                        all_predictions_df.to_csv(f2, header=False, index = False)
                                all_predictions_df = None
                        else:
                            pass
                            # no seizures detected for that posterior_y
            except KeyError:
                logging.error(str(full_fname) + ' did not contain any features! Skipping')
                skip_n += 1
            self.print_progress_preds(i,l, pred_count,prefix = 'Progress:', suffix = 'Seizure predictions:', barLength = 50)
        # save whatever is left of the predicitions (will be < 50 as)
        if not os.path.exists(output_csv_filename):
            all_predictions_df.to_csv(output_csv_filename, index = False)
        else:
            with open(output_csv_filename, 'a') as f2:
                all_predictions_df.to_csv(f2, header=False, index = False)
        try:
            print('Re - ordering spreadsheet by date')
            self.reorder_prediction_csv(output_csv_filename)
            print ('Done')
        except:
            print('unable to re-order spreadsheet by date')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print (traceback.print_exception(exc_type, exc_value, exc_traceback))

        if skip_n != 0:
            print('WARNING: There were files '+str(skip_n)+' without features that were skipped ')
            time.sleep(5)
        return skip_n

    def reorder_prediction_csv(self, csv_path):
        df = pd.read_csv(csv_path, parse_dates=['real_start'])
        reordered_df = df.sort_values(by='real_start')
        reordered_df.to_csv(csv_path)

    def make_prediction_dataframe_rows_from_chunk_labels(self, tid_no, to_stack = [], columns_list = ['Name', 'Pred'], verbose = False):
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
        sec_per_chunk = 3600/n_chunks # assumes  hour here
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
            duration = end_time-start_time
            real_start = utils.get_time_from_seconds_and_filepath(name, float(start_time), split_on_underscore = True).round('s')
            real_end   = utils.get_time_from_seconds_and_filepath(name,float(end_time), split_on_underscore = True ).round('s')
            row = pd.Series([name,start_time,end_time,duration, tid_no, real_start, real_end],
                            index = ['filename','start', 'end','duration', 'transmitter','real_start','real_end'])
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
        bar             = '*' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def print_progress_preds (self, iteration, total, pred_count, prefix = '', suffix = '', decimals = 2, barLength = 100):
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

        bar             = '*' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %.2f%s %s %s' % (prefix, bar, percents, '%', suffix, pred_count)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write('\n')
            sys.stdout.flush()

