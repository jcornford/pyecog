import sys
import os
import multiprocessing
import traceback
import numpy as np
import pandas as pd
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QTimer
import pyqtgraph as pg
import h5py
import logging
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

try:
    from context import loading_subwindow, convert_ndf_window, library_subwindow, add_pred_features_subwindow, clf_subwindow
    from context import ndf
except:
    from .context import loading_subwindow, convert_ndf_window, library_subwindow, add_pred_features_subwindow, clf_subwindow
    from .context import ndf

from ndf.h5loader import H5File
from ndf.datahandler import DataHandler, NdfFile # todo - should bot be importing ndffile?
from ndf.classifier import Classifier
'''
except:
    import loading_subwindow, convert_ndf_window, library_subwindow, add_pred_features_subwindow, clf_subwindow
    from pyecog.ndf.h5loader import H5File
    from pyecog.ndf.datahandler import DataHandler, NdfFile
    from pyecog.ndf.classifier import Classifier
'''
# todo : these classes could inherit classes that have signals and slots already made, as you kept the gui element names the same when possible.

def throw_error(error_text = None):
    msgBox = QtWidgets.QMessageBox()
    if error_text is None:
        msgBox.setText('Error caught! \n'+str(traceback.format_exc(1)))
    else:
        msgBox.setText('Error caught! \n'+str(error_text))
    msgBox.exec_()
    return 0

class ClfWindow(QtGui.QDialog,clf_subwindow.Ui_ClfManagement):
    ''' For handling the classifier...'''
    def __init__(self, parent = None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.home = '' # default folder that can get set when this is class is called from main window
        self.h5directory  = None
        self.library_path = None
        self.clf = None
        self.worker = None

        self.set_h5_folder.clicked.connect(self.get_h5_folder)
        self.set_library.clicked.connect(self.get_library)
        self.make_classifier.clicked.connect(self.make_classifier_method)
        self.save_classifier.clicked.connect(self.save_classifier_method)
        self.load_classifier.clicked.connect(self.load_classifier_method)
        self.train_clf.clicked.connect(self.train_clf_method)
        self.downsample_bl.textChanged.connect(self.updated_sampling_params)
        self.upsample_s_factor.textChanged.connect(self.updated_sampling_params)
        self.estimate_error.clicked.connect(self.not_done_yet)
        self.run_clf_on_folder.clicked.connect(self.predict_seizures)

    def not_done_yet(self):
        QtGui.QMessageBox.information(self,"Not implemented, lazy!", "Not implemented yet! Jonny has been lazy!")

    def get_library(self):
        self.library_path = QtGui.QFileDialog.getOpenFileName(self, "Pick an annotations file", self.home)[0]
        if self.library_path == '':
            print('No path selected')
            return 0
        self.library_path_display.setText(self.library_path)

    def get_label_counts(self):
        self.counts = pd.Series(np.ravel(self.clf.labels[:])).value_counts().values
        self.label_n_baseline.setText('Library has '+ str(self.counts[0]) +' BL chunks')
        self.label_n_seizures.setText('and '+ str(self.counts[1]) +' Seizure chunks. '+str(np.round((self.counts[1]/self.counts[0])*100, 2)) + '%')

        dwnsample_factor = int(self.downsample_bl.text())
        upsample_factor = int(self.upsample_s_factor.text())
        expected_resample = (int(self.counts[0]/dwnsample_factor),self.counts[1]*upsample_factor)
        self.label_resampled_numbers.setText('Expected resample: ' +str(list(expected_resample)) + '. '+
                                             str(np.round((expected_resample[1]/expected_resample[0])*100, 2)) + '%')
    def updated_sampling_params(self):
        try:
            dwnsample_factor = int(self.downsample_bl.text())
            upsample_factor = int(self.upsample_s_factor.text())
        except ValueError:
            #print('Not valid number entered')
            return 0

        expected_resample = (int(self.counts[0]/dwnsample_factor),self.counts[1]*upsample_factor)
        self.label_resampled_numbers.setText('Expected resample: ' +str(list(expected_resample)) + '. '+
                                             str(np.round((expected_resample[1]/expected_resample[0])*100, 2)) + '%')
    def make_classifier_method(self):
        if self.library_path:
            try:
                self.clf = Classifier(self.library_path)
                QtGui.QMessageBox.information(self, "Not?", "Classifier initialised successfully!")
                self.get_label_counts()
            except:
                msgBox = QtWidgets.QMessageBox()
                msgBox.setText('Error!   \n'+ str(traceback.format_exc(1)) )
                msgBox.exec_()
                return 0
        else:
            QtGui.QMessageBox.information(self, "Not?", "Please choose a valid library path")
            return 0

    def save_classifier_method(self):
        self.clf_path = QtGui.QFileDialog.getSaveFileName(self, "Choose savename", self.home)[0]
        print(self.clf_path)
        self.clf_path = self.clf_path if self.clf_path.endswith('.p') else self.clf_path+'.p'
        if self.clf:
            self.clf.save(fname = self.clf_path)
            self.update_clf_path_display()
        else:
            QtGui.QMessageBox.information(self, "Not?", "No classifier to save")
            return 0

    def load_classifier_method(self):
        temp_clf_path = QtGui.QFileDialog.getOpenFileName(self, "Select pickled classifier to load", self.home)[0]
        if temp_clf_path == '':
            print('No folder selected')
            return 0
        else:
            try:
                with open(temp_clf_path, 'rb') as f:
                    self.clf = pickle.load(f)
            except:
                QtGui.QMessageBox.information(self, "Not?", "ERROR: Classifier loading failed. ")
                with open(temp_clf_path, 'rb') as f:
                    self.clf = pickle.load(f)
                return 0

            self.clf_path = temp_clf_path
            self.update_clf_path_display()
            self.get_label_counts()

    def get_h5_folder(self):
        self.h5directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)
        if self.h5directory == '':
            print('No folder selected')
            return 0
        self.update_h5_folder_display()

    def update_clf_path_display(self):
        self.clf_path_display.setText(str(self.clf_path))
        print(pd.Series(np.ravel(self.clf.labels[:])).value_counts().values)

    def update_h5_folder_display(self):
        self.h5_folder_display.setText(str(self.h5directory))

    def train_clf_method(self):
        # boot up a thread here and re-implement the train method of the classifier... maybe with more sensible resampling stuff!
        try:
            if self.worker:
                if self.worker.isRunning():
                    QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
                return 0
        except:
            pass

        self.worker = TrainClassifierThread() # assume the previous finished thread is garbage collected...!
        self.worker.update_progress_label.connect(self.update_progress_label)
        self.worker.SetProgressBar.connect(self.update_progress)
        self.worker.setMaximum_progressbar.connect(self.set_max_bar)
        self.worker.update_label_below.connect( self.update_label_below)
        self.worker.update_label_above2.connect( self.update_label_above2)
        self.worker.finished.connect(self.end_training)



        dwnsample_factor = int(self.downsample_bl.text())
        upsample_factor = int(self.upsample_s_factor.text())
        ntrees = int(self.n_trees.text())
        ncores = self.n_cores.text()
        if ncores == 'all':
            ncores = -1
        else:
            ncores = int(ncores)

        self.worker.set_training_params(self.clf, dwnsample_factor, upsample_factor,ntrees, ncores)
        self.worker.start()

    def end_training(self):
        # todo quit threads properly
        QtGui.QMessageBox.information(self, "Not implemented, lazy!", "You probably want to save the trained classifier...!")
        self.save_classifier_method()
        self.worker.quit()
        self.worker.terminate()
        del self.worker

    def end_prediction(self):
        self.worker.quit()
        self.worker.terminate()
        del self.worker


    def update_label_above2(self, label_string):
        self.progressBar_label_above2.setText(label_string)
    def update_progress_label(self, label_string):
        self.progressBar_lable_above1.setText(label_string)
    def update_label_below(self, label_string):
        self.progressBar_label_below.setText(label_string)
    def set_max_bar(self, signal):
        self.progressBar.setMaximum(int(signal))
    def update_progress(self, signal):
        self.progressBar.setValue(int(signal))
    def missing_features(self, n_missing):
        throw_error('There were '+str(n_missing)+' files with missing features!')

    def predict_seizures(self):
        try:
            if self.worker:
                if self.worker.isRunning():
                    QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
                return 0
        except:
            pass
        try:

            if self.h5directory  == None:
                QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Please choose a h5 folder!")
                self.get_h5_folder()
            if self.clf == None:
                QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Please choose a trained, pickled classifier!")
                self.load_classifier_method()



            self.worker = PredictSeizuresThread() # assume the previous finished thread is garbage collected...!
            self.worker.update_progress_label.connect(self.update_progress_label)
            self.worker.SetProgressBar.connect(self.update_progress)
            self.worker.setMaximum_progressbar.connect(self.set_max_bar)
            self.worker.update_label_below.connect( self.update_label_below)
            self.worker.update_label_above2.connect( self.update_label_above2)
            self.worker.finished.connect(self.end_prediction)
            self.worker.missing_features.connect(self.missing_features)

            h5_folder = self.h5_folder_display.text()
            excel_sheet = QtGui.QFileDialog.getSaveFileName(self,  "make output csv file", self.home)[0]

            self.worker.set_params(self.clf, h5_folder, excel_sheet)
            self.worker.start()

        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText('Error!   \n'+ str(traceback.format_exc(1)) )
            msgBox.exec_()


class PredictSeizuresThread(QThread):
    finished = pyqtSignal()
    update_progress_label = pyqtSignal(str)
    SetProgressBar= pyqtSignal(str)
    setMaximum_progressbar= pyqtSignal(str)
    update_label_below = pyqtSignal(str)
    update_label_above2 = pyqtSignal(str)
    missing_features = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
    def set_params(self, clf, prediction_dir, excel_sheet):
        self.clf = clf
        self.prediction_dir = prediction_dir
        self.excel_output = excel_sheet.split('.')[0]+'.csv'
        if os.path.exists(self.excel_output):
            os.remove(self.excel_output)

    def run(self):
        self.update_label_below.emit('Saving predictions: '+ self.excel_output)
        self.update_progress_label.emit('We are now rolling - look in your terminal for progress')

        output = self.clf.predict_dir(self.prediction_dir,
                             self.excel_output,
                             called_from_gui = True)
        if output != 0:
            self.missing_features.emit(str(output))
        self.update_progress_label.emit('Done')
        self.finished.emit()
        self.exit()

class TrainClassifierThread(QThread):
    # add in signals here
    finished = pyqtSignal()
    update_progress_label = pyqtSignal(str)
    SetProgressBar= pyqtSignal(str)
    setMaximum_progressbar= pyqtSignal(str)
    update_label_below = pyqtSignal(str)
    update_label_above2 = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)

    def set_training_params(self, clf, downsample_bl_by_x, upsample_seizure_by_x, ntrees, n_cores):
        # you sort out the re-sampling and n trees here
        # or do you want to use the class importances?
        self.clf = clf
        self.n_cores = n_cores
        self.ntrees = ntrees
        self.downsample_bl_factor    = downsample_bl_by_x
        self.upsample_seizure_factor = upsample_seizure_by_x

        counts = pd.Series(np.ravel(self.clf.labels[:])).value_counts().values
        target_resample = (int(counts[0]/self.downsample_bl_factor),counts[1]*self.upsample_seizure_factor)
        self.update_label_above2.emit('Resampling [BL S] from '+str(counts)+' to ' + str(list(target_resample)))
        self.res_y, self.res_x = self.clf.resample_training_dataset(self.clf.labels, self.clf.features,
                                                      sizes = target_resample)

    def run(self):

        '''
        # would be very nice to emit this back
        #self.feature_weightings = sorted(zip(self.clf.rf.feature_importances_, self.clf.feature_names),reverse = True)
        '''
        self.update_progress_label.emit('Training Random Forest...')
        self.clf.train(self.downsample_bl_factor,
                       self.upsample_seizure_factor,
                       self.ntrees,
                       self.n_cores,
                       n_emission_prob_cvfolds = 3,
                       pyecog_hmm = True,
                       calc_emissions = True,
                       rf_weights = None,
                       calibrate = False)
        self.finished.emit()
        self.exit()

class AddPredictionFeaturesWindow(QtGui.QDialog, add_pred_features_subwindow.Ui_make_features):

    ''' Add predictions to h5 folder '''
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.set_h5_folder.clicked.connect(self.get_h5_folder)
        self.extract_features_button.clicked.connect(self.run_pred_feature_extraction)
        self.run_peakdet_checkBox.stateChanged.connect(self.use_peaks_changed)

        self.home = '' # default folder that can get set when this is class is called from main window
        self.h5directory  = None

        self.extraction_thread = ExtractPredictionFeaturesThread()
        self.extraction_thread.set_progress_bar.connect(self.update_progress)
        self.extraction_thread.set_max_progress.connect( self.set_max_bar)
        self.extraction_thread.update_hidden_label.connect( self.update_hidden)
        self.extraction_thread.update_progress_label.connect( self.update_progress_label)

    def get_h5_folder(self):
        self.h5directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)
        if self.h5directory == '':
            print('No foldrr selected')
            return 0
        self.update_h5_folder_display()

    def update_h5_folder_display(self):
        self.h5_display.setText(str(self.h5directory))

    def use_peaks_changed(self):
        self.use_peaks_bool = self.run_peakdet_checkBox.isChecked()

    def update_hidden(self, label_string):
        self.hidden_label.setText(label_string)
    def update_progress_label(self, label_string):
        self.progress_bar_label.setText(label_string)
    def set_max_bar(self, signal):
        self.progressBar.setMaximum(int(signal))
    def update_progress(self, signal):
        self.progressBar.setValue(int(signal))

    def run_pred_feature_extraction(self):
        # grab the settings...

        if self.extraction_thread.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0

        chunk_len   = int(self.chunk_len_box.text())
        ncores = self.cores_to_use.text()
        use_peaks_bool = self.run_peakdet_checkBox.isChecked()

        if ncores == 'all':
            ncores = -1
        else:
            ncores = int(ncores)

        try:
            logfilepath = logging.getLoggerClass().root.handlers[0].baseFilename
            self.logpath_dsplay.setText(str(logfilepath))
        except:
            print('couldnt get logpath')



        self.extraction_thread.set_params_for_extraction(h5_folder=self.h5directory,
                                            timewindow = chunk_len,
                                            run_peakdet_flag = use_peaks_bool,
                                            n_cores=ncores)
        self.extraction_thread.start()



class ExtractPredictionFeaturesThread(QThread):
    # todo this re implements datahandler - delete this!
    set_max_progress = pyqtSignal(str)
    update_hidden_label = pyqtSignal(str)
    set_progress_bar =  pyqtSignal(str)
    update_progress_label = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        self.handler = DataHandler()

    def set_params_for_extraction(self, h5_folder,
                                        timewindow,
                                        run_peakdet_flag,
                                        n_cores = -1):

        self.handler.parrallel_flag_pred = True
        self.handler.run_pkdet = run_peakdet_flag
        self.handler.twindow = timewindow
        self.files_to_add_features = [f for f in self.handler.fullpath_listdir(h5_folder) if f.endswith('.h5')]
        if n_cores == -1:
            n_cores = multiprocessing.cpu_count()
        self.n_cores = n_cores

        #l = len(self.files_to_add_features)

        self.set_max_progress.emit(str(len(self.files_to_add_features)))
        self.update_hidden_label.emit(str(len(self.files_to_add_features))+' Files to extract features from')

    def run(self):
        pool = multiprocessing.Pool(self.n_cores)

        self.set_progress_bar.emit(str(0))
        self.update_progress_label.emit('Progress: ' +str(0)+ ' / '+ str(len(self.files_to_add_features)))

        for i, _ in enumerate(pool.imap(self.handler.add_predicition_features_to_h5_file, self.files_to_add_features), 1):
            self.set_progress_bar.emit(str(i))
            self.update_progress_label.emit('Progress: ' +str(i)+ ' / '+ str(len(self.files_to_add_features)))

        pool.close()
        pool.join()

        self.update_progress_label.emit('Progress: Done')
        self.set_progress_bar.emit(str(0))
        self.handler.reset_date_modified_time(self.files_to_add_features)
        self.handler.parrallel_flag_pred = False # really not sure this is needed- just a hangover from the datahandler code?


class LibraryWindow(QtGui.QDialog, library_subwindow.Ui_LibraryManagement):
    ''' this is for the predictions, csv and h5 folder needed '''
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.select_annotations.clicked.connect(self.select_annotations_method)
        self.select_h5_folder.clicked.connect(self.select_h5_folder_method)
        self.new_library.clicked.connect(self.make_new_library)
        self.add_to_library.clicked.connect(self.append_to_library)
        self.set_library.clicked.connect(self.select_library)
        self.clear_library.clicked.connect(self.clear_library_path)
        self.chunk_length.textChanged.connect(self.chunk_len_changed)
        self.add_labels.clicked.connect(self.calculate_labels_for_library)
        self.add_features.clicked.connect(self.calculate_features_for_library)
        self.use_peaks.clicked.connect(self.use_peaks_changed)
        self.use_peaks.stateChanged.connect(self.use_peaks_changed)
        self.overwrite_box.stateChanged.connect(self.overwrite_box_changed)
        self.fs_box.textChanged.connect(self.fs_box_changed)

        self.library_path = None
        self.annotation_path = None
        self.h5_folder_path = None
        self.home = None # inherit default folder
        self.chosen_chunk_length = int(self.chunk_length.text())
        self.annotation_df = None
        self.overwrite_box.setChecked(True)
        self.overwrite_bool = self.overwrite_box.isChecked()
        self.use_peaks_bool = self.use_peaks.isChecked()
        self.fs = int(self.fs_box.text())

        #self.progressBar etc...
        #self.progressBar_label_above2
        self.spawn_worker()

    def spawn_worker(self):
            self.worker = LibraryWorkerThread()
            self.worker.finished.connect(self.worker_finished)
            self.worker.update_progress_label.connect(self.update_progress_label)
            self.worker.SetProgressBar.connect(self.update_progress)
            self.worker.setMaximum_progressbar.connect( self.set_max_bar)
            self.worker.update_label_below.connect( self.update_label_below)
            self.worker.update_label_above2.connect( self.update_label_above2)


    def worker_finished(self):
        print('worker finished! method called - terminating? - needlessly?')
        if self.worker:
            print(self.worker)
        self.spawn_worker()

    def select_annotations_method(self):
        self.annotation_path = QtGui.QFileDialog.getOpenFileName(self, "Pick an annotations file", self.home)[0]
        self.annotations_display.setText(self.annotation_path)
        if self.annotation_path.endswith('.xlsx'):
            self.annotation_df = pd.read_excel(self.annotation_path)
        elif self.annotation_path.endswith('.csv'):
            self.annotation_df = pd.read_csv(self.annotation_path)
        else:
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Please use .csv or .xlsx files!")
        try:
            self.annotation_df.columns = [label.lower() for label in self.annotation_df.columns]
            self.annotation_df.columns  = [label.strip(' ') for label in self.annotation_df.columns]
            assert 'filename' in self.annotation_df
            assert 'start' in self.annotation_df
            assert 'end'   in self.annotation_df
            assert 'transmitter' in self.annotation_df
        except:
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Please make sure annotations file contains at least 'Filename', 'Start', 'End', 'Transmitter'!")

        # here check fileheadings are all good

    def select_h5_folder_method(self):
        self.h5_folder_path = QtGui.QFileDialog.getExistingDirectory(self, "Pick h5 folder", self.home)
        self.h5_folder_display.setText(self.h5_folder_path)

    def chunk_len_changed(self):
        try:
            self.chosen_chunk_length = int(self.chunk_length.text())
            print(self.chosen_chunk_length)
        except ValueError:
            print('Not valid number entered')


    def fs_box_changed(self):
        try:
            self.fs = int(self.fs_box.text())
        except:
            pass

    def use_peaks_changed(self):
        self.use_peaks_bool = self.use_peaks.isChecked()

    def overwrite_box_changed(self):
        self.overwrite_bool = self.overwrite_box.isChecked()

    def select_library(self, default_lib = None):
        if default_lib:
            self.library_path = QtGui.QFileDialog.getOpenFileName(self,  "Choose Library file", default_lib)[0]
        else:
            self.library_path = QtGui.QFileDialog.getOpenFileName(self,  "Choose Library file", self.home)[0]

        self.update_library_path_display()

    def clear_library_path(self):
        self.library_path = ''
        self.update_library_path_display()

    def update_library_path_display(self):
        self.library_path_display.setText(self.library_path)

    def update_label_above2(self, label_string):
        self.progressBar_label_above2.setText(label_string)
    def update_progress_label(self, label_string):
        self.progressBar_lable_above1.setText(label_string)
    def update_label_below(self, label_string):
        self.progressBar_label_below.setText(label_string)
    def set_max_bar(self, signal):
        self.progressBar.setMaximum(int(signal))
    def update_progress(self, signal):
        self.progressBar.setValue(int(signal))

    def make_new_library(self):
        self.library_path = QtGui.QFileDialog.getSaveFileName(self,  "Make new Library file", self.home)[0]
        if self.library_path == '':
            print('No path entered... aborting')
            return 0

        if not self.library_path.endswith('.h5'):
            try:
                self.library_path = self.library_path.split('.')[0]+'.h5'
            except:
                self.library_path = self.library_path +'.h5'
        print(self.library_path)


        self.update_library_path_display()

        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        else:
            self.worker.set_library_attributes(self.library_path,
                                     self.annotation_df,
                                     self.h5_folder_path,
                                     self.chosen_chunk_length,
                                     self.overwrite_bool,
                                     self.fs)
            self.worker.new_library_mode()
            self.worker.start()
            self.worker.wait()

            print('Worker finished')
            self.emit_finished_message()

    def emit_finished_message(self):
        # as you currently have the prgoressbar frozen
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText('Process finished')
        msgBox.exec_()

    def append_to_library(self):
        if self.library_path:
            self.select_library(default_lib=self.library_path)
        else:
            self.select_library()

        if self.library_path == '':
            print ('No library path chosen')
            return 0

        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        else:
            self.worker.set_library_attributes(self.library_path,
                                     self.annotation_df,
                                     self.h5_folder_path,
                                     self.chosen_chunk_length,
                                     self.overwrite_bool,
                                     self.fs)

            self.worker.append_to_library_mode()
            self.worker.start()
            #self.worker.wait()
            print('Worker finished')
            self.emit_finished_message()


    def calculate_features_for_library(self):
        if self.library_path is None:
            self.select_library()
            self.update_library_path_display()

        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        else:
            self.worker.add_features_mode()
            self.worker.set_library_attributes_for_feats(self.library_path, self.chosen_chunk_length, self.overwrite_bool, self.use_peaks_bool)
            self.worker.start()
            print('Worker finished')
            self.emit_finished_message()


    def calculate_labels_for_library(self):
        if self.library_path is None:
            self.select_library()
            self.update_library_path_display()

        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        else:
            self.worker.add_labels_mode()
            self.worker.set_library_attributes_for_feats(self.library_path, self.chosen_chunk_length, self.overwrite_bool, self.use_peaks_bool)
            self.worker.start()
            print('Worker finished')
            self.emit_finished_message()

class LibraryWorkerThread(QThread):

    finished = pyqtSignal(str)
    update_progress_label = pyqtSignal(str)
    SetProgressBar= pyqtSignal(str)
    setMaximum_progressbar= pyqtSignal(str)
    update_label_below =pyqtSignal(str)
    update_label_above2 = pyqtSignal(str)


    def __init__(self):
        QThread.__init__(self)
        self.handler = DataHandler()

    def __del__(self):
        #self.wait()
        self.exit()

    def set_library_attributes(self, l_path, a_df, h5_path, timewindow, overwrite_bool, fs):
        self.library_path = l_path
        self.h5_path = h5_path
        self.annotations_df = a_df
        self.t_len = timewindow
        self.overwrite_bool = overwrite_bool
        self.fs = fs

    def set_library_attributes_for_feats(self, l_path, timewindow, overwrite_bool, peaks_bool):
        self.library_path = l_path
        self.t_len = timewindow
        self.overwrite_bool = overwrite_bool
        self.run_peaks_bool = peaks_bool

    def add_labels_mode(self):
        self.labels_or_features = True
        self.add_features = False

    def add_features_mode(self):
        self.labels_or_features = True
        self.add_features = True

    def new_library_mode(self):
        self.appending_to_library = False
        self.labels_or_features = False

    def append_to_library_mode(self):
        self.appending_to_library = True
        self.labels_or_features = False

    def run(self):

        # this not how to do it
        if self.labels_or_features == False:
            if self.appending_to_library:
                self.update_progress_label.emit('Progress Bar is Frozen - no biggy')
                output = self.handler.append_to_seizure_library(df = self.annotations_df,
                                                       file_dir=self.h5_path,
                                                       seizure_library_path=self.library_path,
                                                       overwrite=self.overwrite_bool,
                                                       timewindow=self.t_len, fs=self.fs)
            else:
                #self.emit(pyqtSignal("update_progress_label(QString)"),'Progress Bar is Frozen - no biggy')
                self.update_progress_label.emit('Progress Bar is Frozen - no biggy')
                output = self.handler.make_seizure_library(df = self.annotations_df,
                                                       file_dir=self.h5_path,
                                                       seizure_library_name=self.library_path,
                                                       overwrite=self.overwrite_bool,
                                                       timewindow=self.t_len, fs=self.fs)
                if output == 0:
                    throw_error(' An error occurred, check terminal window ')

        elif self.labels_or_features == True:
            if not self.add_features:
                #self.emit(pyqtSignal("update_progress_label(QString)"),'Progress Bar is Frozen - no biggy')
                self.update_progress_label.emit('Progress Bar is Frozen - no biggy')
                self.handler.add_labels_to_seizure_library(self.library_path,self.overwrite_bool,self.t_len)
                print('labels done, chunked: '+ str(self.t_len))

            elif self.add_features:
                print('lets do the features')
                #self.emit(pyqtSignal("update_progress_label(QString)"),'Progress Bar is Frozen - no biggy')
                self.update_progress_label.emit('Progress Bar is Frozen - no biggy')
                self.handler.add_features_seizure_library(self.library_path,self.overwrite_bool,self.run_peaks_bool, self.t_len)
                print('features done, chunked: '+ str(self.t_len))
        self.exit() # does this emit the finished signal? dont think so




class ConvertingNDFsWindow(QtGui.QDialog, convert_ndf_window.Ui_convert_ndf_to_h5):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.select_ndf_folder.clicked.connect(self.get_ndf_folder)
        self.set_h5_folder.clicked.connect(self.get_h5_folder)
        self.convert_button.clicked.connect(self.convert_folder)

        #self.transmitter_ids
        self.home = '' # default folder to be inherited
        self.progressBar.setValue(0)
        self.cores_to_use.setText(str(1))
        self.transmitter_ids.setText('all')
        self.converting_thread = None

    def get_h5_folder(self):
        self.h5directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)
        self.h5_display.setText(str(self.h5directory))

    def get_ndf_folder(self):
        self.ndf_folder = QtGui.QFileDialog.getExistingDirectory(self, 'Select a ndf folder to convert', self.home)
        self.ndf_display.setText(str(self.ndf_folder))

    def convert_folder(self):
        try:
            if self.converting_thread.isRunning():
                QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
                return 0
        except:
            pass

        tids = self.transmitter_ids.text().strip("'")
        fs   = int(self.fs_box.text())
        ncores = self.cores_to_use.text()
        glitch_detection = self.checkbox_ndf_glitch_removal.isChecked()
        if ncores == 'all':
            ncores = -1
        else:
            ncores = int(ncores)

        if tids != 'all':
            if type(eval(tids)) != list:
                tids = eval('['+tids+']')
            else:
                tids = eval(tids)
            tids = sorted(tids)


        self.converting_thread = ConvertNdfThread()
        self.converting_thread.set_progress_bar.connect(self.update_progress)
        self.converting_thread.set_max_progress.connect( self.set_max_bar)
        self.converting_thread.update_hidden_label.connect( self.update_hidden)
        self.converting_thread.update_progress_label.connect( self.update_progress_label)

        try:
            logfilepath = logging.getLoggerClass().root.handlers[0].baseFilename
            self.logpath_dsplay.setText(str(logfilepath))
        except:
            print('couldnt get logpath')
            #logfilepath = logging.getLoggerClass().root.handlers[0].baseFilename
        try:

            self.converting_thread.convert_ndf_directory_to_h5(ndf_dir=self.ndf_folder,
                                                save_dir=self.h5directory,
                                                tids=tids,
                                                n_cores=ncores,
                                                fs=fs,
                                                glitch_detection_flag = glitch_detection)
            self.converting_thread.start()
        except:
            QtGui.QMessageBox.information(self," ", "Error!")


    def update_hidden(self, label_string):
        self.hidden_label.setText(label_string)
    def update_progress_label(self, label_string):
        self.progress_bar_label.setText(label_string)
    def set_max_bar(self, signal):
        self.progressBar.setMaximum(int(signal))
    def update_progress(self, signal):
        self.progressBar.setValue(int(signal))

class ConvertNdfThread(QThread):

    set_max_progress = pyqtSignal(str)
    update_hidden_label = pyqtSignal(str)
    set_progress_bar =  pyqtSignal(str)
    update_progress_label = pyqtSignal(str)
    def __init__(self):
        QThread.__init__(self)
        self.handler = DataHandler()


    def convert_ndf_directory_to_h5(self,
                                    ndf_dir,
                                    tids = 'all',
                                    save_dir  = 'same_level',
                                    n_cores = -1,
                                    fs = 'auto',
                                    glitch_detection_flag = True):
        """
        Copy from datahandler, this should be a thread?
        """
        self.handler.glitch_detection_flag_for_parallel_conversion = glitch_detection_flag
        self.handler.fs_for_parallel_conversion = fs
        print(self.handler.fs_for_parallel_conversion)
        self.files = [f for f in self.handler.fullpath_listdir(ndf_dir) if f.endswith('.ndf')]

        # tids
        if not tids == 'all':
            if not hasattr(tids, '__iter__'):
                tids = [tids]

        self.handler.tids_for_parallel_conversion = tids

        # set n_cores
        if n_cores == -1:
            n_cores = multiprocessing.cpu_count()
        self.n_cores = n_cores
        # Make save directory
        if save_dir  == 'same_level':
            save_dir = ndf_dir+'_converted_h5s'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.handler.savedir_for_parallel_conversion = save_dir

        self.set_max_progress.emit(str(len(self.files)))
        self.update_hidden_label.emit(str(len(self.files))+' Files for conversion. Transmitters: '+ str(self.handler.tids_for_parallel_conversion))

    def run(self):
        pool = multiprocessing.Pool(self.n_cores)
        self.set_progress_bar.emit(str(0))
        self.update_progress_label.emit('Progress: ' +str(0)+ ' / '+ str(len(self.files)))
        for i, _ in enumerate(pool.imap(self.handler.convert_ndf, self.files), 1):
            self.set_progress_bar.emit(str(i))
            self.update_progress_label.emit('Progress: ' +str(i)+ ' / '+ str(len(self.files)))
        pool.close()
        pool.join()
        self.update_progress_label.emit('Progress: Done')
        #self.set_progress_bar.emit(str())
        self.handler.reset_date_modified_time(self.files)

class LoadingSubwindow(QtGui.QDialog, loading_subwindow.Ui_Dialog):
    ''' this is for checking out predictions on main gui, csv and h5 folder needed '''
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.set_prediction_file.clicked.connect(self.get_pred_filename)
        self.set_h5_folder.clicked.connect(self.get_h5_folder)

        self.home = '' # default folder to be inherited
        self.predictions_fname = None
        self.h5directory       = None

    def get_h5_folder(self):
        self.h5directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)
        print(self.h5directory)
        print(type(self.h5directory))
        self.update_h5_folder_display()

    def update_h5_folder_display(self):
        self.h5_display.setText(str(self.h5directory))

    def update_predictionfile_display(self):
        self.prediction_display.setText(str(self.predictions_fname))

    def get_pred_filename(self):
        self.predictions_fname = QtGui.QFileDialog.getOpenFileName(self, 'Select a predicitons file', self.home)[0]
        self.update_predictionfile_display()