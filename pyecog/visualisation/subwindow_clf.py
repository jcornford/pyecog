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


from sklearn import metrics

try:
    from context import loading_subwindow, convert_ndf_window, library_subwindow, add_pred_features_subwindow, clf_subwindow
    from context import ndf
except:
    from .context import loading_subwindow, convert_ndf_window, library_subwindow, add_pred_features_subwindow, clf_subwindow
    from .context import ndf

from ndf.h5loader import H5File
from ndf.datahandler import DataHandler, NdfFile # todo - should bot be importing ndffile?
from ndf.classifier import Classifier, ClassificationAlgorithm
from ndf.hmm_pyecog import HMMBayes,HMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

        self.descrim_algo_button_group = QtGui.QButtonGroup(self)
        self.descrim_algo_button_group.addButton(self.lg_box)
        self.descrim_algo_button_group.addButton(self.rf_box)

    @staticmethod
    def throw_error(error_text=None):
        msgBox = QtWidgets.QMessageBox()
        if error_text is None:
            msgBox.setText('Error caught! \n' + str(traceback.format_exc(1)))
        else:
            msgBox.setText('Error caught! \n' + str(error_text))
        msgBox.exec_()
        return 0

    def not_done_yet(self):
        QtGui.QMessageBox.information(self,"Not implemented, lazy!", "Not implemented yet! Jonny has been lazy!")

    def get_library(self):
        self.library_path = QtGui.QFileDialog.getOpenFileName(self, "Pick an annotations file", self.home)[0]
        if self.library_path == '':
            print('No path selected')
            return 0
        self.library_path_display.setText(self.library_path)

    def get_label_counts(self):
        self.counts = self.clf.label_value_counts
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
                if self.lg_box.isChecked():
                    descrim_algo = LogisticRegression()
                elif self.rf_box.isChecked():
                    ntrees = int(self.n_trees.text())
                    ncores = self.n_cores.text()
                    descrim_algo = RandomForestClassifier(n_estimators=ntrees,
                                                          random_state=7, n_jobs=ncores)
                if self.probabilistic_hmm_box.isChecked():
                    hmm_algo = HMMBayes()
                else:
                    hmm_algo = HMM()
                self.clf.algo = ClassificationAlgorithm(descrim_algo,hmm_algo)
                self.clf.preprocess_features()
                print(self.clf.algo.descriminative_model)
                print(self.clf.algo.hmm)
                self.get_label_counts()
                QtGui.QMessageBox.information(self, "Info:", "Classifier initialised successfully!")
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
        ntrees = int(self.n_trees.text()) # these arent required here
        ncores = self.n_cores.text()
        if ncores == 'all':
            ncores = -1
        else:
            ncores = int(ncores)

        self.worker.set_training_params(self.clf,
                                        dwnsample_factor,
                                        upsample_factor,
                                        ntrees, ncores)
        self.worker.start()

    def end_training(self):
        # todo quit threads properly
        QtGui.QMessageBox.information(self, "INFO",
                                      "You probably want to save the trained classifier...!")
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

        output = self.clf.predict_directory(self.prediction_dir,
                             self.excel_output,
                             gui_object = self)
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

    def set_training_params(self, clf, downsample_bl_by_x,
                            upsample_seizure_by_x,
                            ntrees, n_cores):
        # you sort out the re-sampling and n trees here
        # or do you want to use the class importances?
        self.clf = clf
        self.n_cores = n_cores
        self.ntrees = ntrees
        self.downsample_bl_factor    = downsample_bl_by_x
        self.upsample_seizure_factor = upsample_seizure_by_x

    def run(self):

        '''
        # would be very nice to emit this back
        # NOTE doesnt use the n cores or n trees here anymore!
        #self.feature_weightings = sorted(zip(self.clf.rf.feature_importances_, self.clf.feature_names),reverse = True)
        '''
        try:
            self.update_progress_label.emit('Training classifier...')
            self.clf.train(self.downsample_bl_factor,
                           self.upsample_seizure_factor)
            self.finished.emit()
            self.exit()
        except:
            throw_error()