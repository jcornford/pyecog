import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics as sk_metrics


def get_predictions_cross_val(X, y, model,
                              n_cvfolds=3,
                              random_state=7,
                              print_scores=False,
                              cc_lr=False):
    '''
    returns cross_val_predictions.values, cross_val_probabilities.values '''

    skfold_cv = StratifiedKFold(n_cvfolds, random_state=7)
    cross_val_predictions = pd.DataFrame(np.zeros(shape=(y.shape[0], n_cvfolds)))  # do the df stack..
    cross_val_probabilities = pd.DataFrame(np.zeros(shape=(y.shape[0], np.unique(y).shape[0])))
    metrics = [sk_metrics.f1_score]
    model_cv_performance = {m.__name__: [] for m in metrics}
    model_tr_performance = {m.__name__: [] for m in metrics}

    fold_i = 0
    for t_index, v_index in skfold_cv.split(X, y):
        x_val = X[v_index, :]
        x_train = X[t_index, :]
        y_val = y[v_index]
        y_train = y[t_index]

        model.fit(x_train, y_train)
        if cc_lr:
            print('running case control sampling',)
            model.intercept_ = cc_correct_intercept(model, 0.002993, 0.028)
            print(model.intercept_)
        # get predictions
        val_preds = model.predict(x_val)
        val_probs = model.predict_proba(x_val)
        cross_val_predictions.iloc[v_index, fold_i] = val_preds
        cross_val_probabilities.iloc[v_index] = val_probs

        # validation scores
        f1score = sk_metrics.f1_score(y_val, val_preds, pos_label=1)
        model_cv_performance['f1_score'].append(f1score)

        # training scores
        train_preds = model.predict(x_train)
        f1score = sk_metrics.f1_score(y_train, train_preds, pos_label=1)
        model_tr_performance['f1_score'].append(f1score)

        fold_i += 1

    for key in model_cv_performance.keys():
        model_cv_performance[key] = np.mean(model_cv_performance[key])
        model_tr_performance[key] = np.mean(model_tr_performance[key])
    if print_scores:
        print('Ran CV on library data for HMM state emissions:')
        print('Training:', model_tr_performance)
        print('Testing: ', model_cv_performance)
    cross_val_predictions = cross_val_predictions.sum(axis=1)
    cross_val_predictions[cross_val_predictions > 0] = 1
    return cross_val_predictions.values, cross_val_probabilities.values

from sklearn.utils import resample
def resample_training_dataset(feature_array,labels, sizes):
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
    return resampled_features,resampled_labels[:,0]

import os
import logging
import traceback
import time
import h5py
import sys
from . import utils
from . import h5loader
def predict_dir(prediction_dir,
                output_csv_filename,
                classfier_object,
                posterior_thresh,
                overwrite_previous_predicitions = True,
                gui_object = False):
    '''
    Function to handle prediction of directory
    '''
    if not output_csv_filename.endswith('.csv'):
        output_csv_filename = output_csv_filename + '.csv'
    files_to_predict = [os.path.join(prediction_dir,f) for f in os.listdir(prediction_dir) if not f.startswith('.') if f.endswith('.h5') ]
    l = len(files_to_predict)

    print('Looking through '+ str(len(files_to_predict)) + ' files for seizures:')
    print_progress_preds(0,l,pred_count=0, prefix = 'Progress:', suffix = 'Complete', barLength = 50)

    if gui_object:
        gui_object.update_label_above2.emit('Looking through '+ str(len(files_to_predict)) + ' files for seizures:')
        gui_object.setMaximum_progressbar.emit(str(l))
        gui_object.update_progress_label.emit('Progress: ' + str(0) + ' / ' + str(l))

    if overwrite_previous_predicitions:
        try:
            os.remove(output_csv_filename)
        except:
            pass

    skip_n =  0
    pred_count = 0
    all_predictions_df = None

    for i,fpath in enumerate(files_to_predict, 1):
        full_fname = str(os.path.split(fpath)[1]) # this has more than one tid

        print_progress_preds(i, l, pred_count, prefix='Progress:', suffix='Seizure predictions:', barLength=50)
        if gui_object:
            gui_object.update_progress_label.emit('Progress: ' + str(i) + ' / ' + str(l))
            gui_object.SetProgressBar.emit(str(i))

        try:
            h5file = h5loader.H5File(fpath)
            for tid_no in h5file.attributes['t_ids']:
                tid_no = str(tid_no)
                try:
                    col_names = h5file[int(tid_no)]['feature_col_names']
                    col_names = [b.decode("utf-8") for b in col_names]
                except:
                    continue # to next tid
                pred_features = h5file[int(tid_no)]['features']
                pred_features = classfier_object.pyecog_scaler.transform_features(pred_features, col_names)
                logging.info(full_fname + ' now predicting tid ' + str(tid_no))

                posterior_y = classfier_object.predict(pred_features, posterior_thresh)

                if sum(posterior_y):
                    fname = full_fname.split('[')[0]+'['+tid_no+'].h5'
                    name_array = np.array([fname for i in range(posterior_y.shape[0])])
                    prediction_df = make_prediction_dataframe_rows_from_chunk_labels(tid_no,
                                                                                     to_stack = [name_array, posterior_y])
                    pred_count += prediction_df.shape[0]
                    if all_predictions_df is None:
                        all_predictions_df = prediction_df
                    else:
                        all_predictions_df = all_predictions_df.append(prediction_df, ignore_index=True)

                    if all_predictions_df.shape[0] > 50:
                        if not os.path.exists(output_csv_filename):
                            all_predictions_df.to_csv(output_csv_filename, index=False)
                        else:
                            with open(output_csv_filename, 'a') as f2:
                                all_predictions_df.to_csv(f2, header=False, index=False)
                        all_predictions_df = None
                else:
                    pass
                    # no seizures detected for that posterior_y
        except KeyError:
            # gui should throw error
            print('KeyError:did not contain any features! Skipping')
            logging.error(str(full_fname) + 'did not contain any features! Skipping')
            skip_n += 1

    # save whatever is left of the predicitions (will be < 50 rows)
    if all_predictions_df is not None:
        if not os.path.exists(output_csv_filename):
            all_predictions_df.to_csv(output_csv_filename, index = False)
        else:
            with open(output_csv_filename, 'a') as f2:
                all_predictions_df.to_csv(f2, header=False, index = False)
    try:
        print('Re - ordering spreadsheet by date')
        reorder_prediction_csv(output_csv_filename)
        print ('Done')
    except:
        print('Unable to re-order prediction spreadsheet by date - maybe you had no seizures?')
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print (traceback.print_exception(exc_type, exc_value, exc_traceback))

    if skip_n != 0:
        print('WARNING: There were files '+str(skip_n)+' without features that were skipped ')
        time.sleep(5)
        if gui_object:
            gui_object.throw_error('WARNING: There were files '+str(skip_n)+' without features that were skipped ')

    return skip_n

def reorder_prediction_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['real_start'])
    reordered_df = df.sort_values(by='real_start')
    reordered_df.to_csv(csv_path)

def make_prediction_dataframe_rows_from_chunk_labels(tid_no, to_stack = [], columns_list = ['Name', 'Pred'], verbose = False):
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

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
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

def print_progress_preds (iteration, total, pred_count, prefix = '', suffix = '', decimals = 2, barLength = 100):
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



def get_feature_imps(clf_obj, X_df_columns):
    feature_imps = pd.Series(data=clf_obj.feature_importances_, index=X_df_columns).sort_values(ascending=False)
    return feature_imps


def get_preds_from_rf_oob(rf):
    # todo docstring
    dummy_preds = np.zeros(shape=rf.oob_decision_function_.shape)
    dummy_preds[np.arange(rf.oob_decision_function_.shape[0]), rf.oob_decision_function_.argmax(axis=1)] = 1
    dummy_preds_df = pd.DataFrame(dummy_preds, columns=rf.classes_).astype(int)

    yhat = dummy_preds_df.idxmax(axis=1)
    return yhat


def labelled_confusion_matrix(y, ypred, clf):
    # todo docstring
    row_labels = ['True ' + label for label in clf.classes_]
    col_labels = ['Predicted ' + label for label in clf.classes_]

    cm = sk_metrics.confusion_matrix(y, ypred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.round(cm_norm, decimals=3)
    cm_norm_df = pd.DataFrame(cm_norm, index=row_labels, columns=col_labels)

    return cm_df, cm_norm_df


def get_clf_performance(X, y, model, n_cvfolds=3, random_state=7):
    metrics = [sk_metrics.f1_score, sk_metrics.log_loss]
    model_cv_performance = {m.__name__: [] for m in metrics}
    model_tr_performance = {m.__name__: [] for m in metrics}

    skfold_cv = StratifiedKFold(n_cvfolds, random_state=7)
    for t_index, v_index in skfold_cv.split(X, y):
        x_val = X[v_index, :]
        x_train = X[t_index, :]
        y_val = y[v_index]
        y_train = y[t_index]

        model.fit(x_train, y_train)

        # get predictions
        val_preds = model.predict(x_val)
        val_probs = model.predict_proba(x_val)
        train_preds = model.predict(x_train)
        train_probs = model.predict_proba(x_train)

        # validation scores
        f1score = sk_metrics.f1_score(y_val, val_preds, pos_label=1)
        logloss = sk_metrics.log_loss(y_val, val_probs[:, 1])  # print(model.classes_) if uncertain
        model_cv_performance['log_loss'].append(logloss)
        model_cv_performance['f1_score'].append(f1score)

        # training scores
        f1score = sk_metrics.f1_score(y_train, train_preds, pos_label=1)
        logloss = sk_metrics.log_loss(y_train, train_probs[:, 1])  # print(model.classes_) if uncertain
        model_tr_performance['log_loss'].append(logloss)
        model_tr_performance['f1_score'].append(f1score)

    # insert confusion matrix adding...

    for key in model_cv_performance.keys():
        model_cv_performance[key] = np.mean(model_cv_performance[key])
        model_tr_performance[key] = np.mean(model_tr_performance[key])
    print(model.classes_)
    print('Training:', model_tr_performance)
    print('Testing: ', model_cv_performance)


def make_clf_report(f_string,anno_string, unfair_df = False ):
    ''' f sting is the predictions, anno'''
    print("Classification report for predictions "+f_string)
    try:
        pred_df = pd.read_csv(f_string).iloc[:,1:]
    except:
        pred_df = pd.read_excel(f_string).iloc[:,1:]
    pred_df = add_mcode_tid_col(pred_df)

    try:
        anno_df = pd.read_csv(anno_string).iloc[:,1:]
    except:
        anno_df = pd.read_excel(anno_string).iloc[:,1:]
    anno_df  = add_mcode_tid_col(anno_df)
    if unfair_df:
        anno_unfair_df = add_mcode_tid_col(unfair_df)
        unfair_preds, fair_preds = compare_dfs(pred_df, anno_unfair_df)

    else:
        fair_preds = pred_df
    tp_df, fp_df = compare_dfs(fair_preds, anno_df)

    #print('TP:',tp_df.shape[0],anno_df.shape,'FP:',fp_df.shape[0], fair_preds.shape[0], tp_df.shape[0]+fp_df.shape[0])
    try:
        s_annos = anno_df['seizure duration'] # instead of anno_df.seizure duration.sum()
    except:
        s_annos = anno_df['duration'] # instead of anno_df.seizure duration.sum()

    print(tp_df.overlap.sum(),s_annos.sum(), (tp_df.overlap.sum()/s_annos.sum())*100,'%')

    anno_in_preds, anno_not_in_preds = compare_dfs(anno_df, tp_df) # anno_in_tp
    #print(anno_in_preds.shape, anno_df.shape)
    TPs = anno_in_preds.shape[0]
    FPs = fp_df.shape[0]
    total_Ps  = anno_df.shape[0]
    FNs = anno_not_in_preds.shape[0]
    precision = TPs/(TPs + FPs)
    recall    = TPs/(TPs+FNs)
    f1 = 2*((precision * recall) / (precision + recall))

    # Make table
    headers = ["TP","FP","FN ","precision", "recall", "f1-score",]
    fmt = '%% %ds' % 6  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    for i, label in enumerate(['ictal']):
        values = ['ictal']
        for v in (TPs,FPs,FNs, precision, recall, f1):
            values += ["{0:0.{1}f}".format(v, 3)]

        report += fmt % tuple(values)

    report += '\n'
    print(report)


def add_mcode_tid_col(df):
    df['mcode_tid'] = df.filename.str.slice(0,11)+'_'+df.transmitter.astype(str)
    return df

def check_overlap(series1,series2):
    ''' series should both have start and end attrs'''
    start_a, end_a = float(series1.start), float(series1.end)
    start_b, end_b = float(series2.start), float(series2.end)
    overlap_bool = (start_a <= end_b) and (end_a>=start_b)
    #print(overlap_bool,': ',start_a, '-',end_a,',', start_b,'-', end_b)
    return overlap_bool

def calculate_overlap(series1,series2):
    ''' series should both have start and end attrs
    http://baodad.blogspot.co.uk/2014/06/date-range-overlap.html
    '''
    a, b = float(series1.start), float(series1.end)
    c, d = float(series2.start), float(series2.end)
    b_a = b-a
    b_c = b-c
    d_c = d-c
    d_a = d-a
    overlap = min([b_a,b_c,d_c,d_a])
    #print(overlap,a,b,c,d)
    return overlap
# not quite there - need to make sure every seziure in annot is there, mult seizures when actuall one can mean could skip..?
def compare_dfs(prediction_df, annotation_df):
    ''' You should add a overlap calculation step to this...'''
    #annotations_found_df = annotation_df[['mcode_tid', 'seizure duration']]
    preds_in_anno    = pd.DataFrame(columns = prediction_df.columns)
    preds_notin_anno = pd.DataFrame(columns = prediction_df.columns)
    for _,prediction in prediction_df.iterrows():
        # check if hours and tid or prediciton in annotation df
        overlap_bool = 0
        if prediction.mcode_tid in annotation_df.mcode_tid.unique():
            # now filter annotation df for same hour and tid annotation as the prediction,
            # this should normally just be one row, but if >1 seizures in that hour will be more
            revevant_annotations_df = annotation_df[annotation_df.mcode_tid.isin([prediction.mcode_tid])]
            t_overlap = 0 # storing the overlap time
            for _,annotation in revevant_annotations_df.iterrows():
                row_overlap =  check_overlap(prediction,annotation) # in the case that two seizures, want to add...
                overlap_bool += row_overlap
                if row_overlap:# is this robust to two seizures?
                    t_overlap += calculate_overlap(prediction,annotation)

        if overlap_bool>0:
            prediction['overlap'] = t_overlap
            preds_in_anno = preds_in_anno.append(prediction)
        else:
            preds_notin_anno = preds_notin_anno.append(prediction)
    return preds_in_anno, preds_notin_anno#, annotations_found_df
