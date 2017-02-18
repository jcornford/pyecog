import sys
import os
import multiprocessing
import numpy as np
import pandas as pd
from PyQt5 import QtGui#,# uic
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QTimer
import pyqtgraph as pg
import h5py
import logging
from pyecog.visualisation import loading_subwindow, convert_ndf_window, library_subwindow
from pyecog.ndf.h5loader import H5File
from pyecog.ndf.datahandler import DataHandler, NdfFile

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
            #self.connect(self.worker, pyqtSignal("update_label_above2(QString)"), self.update_label_above2)
            #self.connect(self.worker, pyqtSignal("update_label_below(QString)"), self.update_label_below)
            #self.connect(self.worker, pyqtSignal("setMaximum_progressbar(QString)"), self.set_max_bar)
            #self.connect(self.worker, pyqtSignal("SetProgressBar(QString)"), self.update_progress)
            #self.connect(self.worker, pyqtSignal("update_progress_label(QString)"), self.update_progress_label)
            #self.connect(self.worker, pyqtSignal("finished()"), self.worker_finished)

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
        self.fs = int(self.fs_box.text())

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
        self.update_library_path_display()


        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        elif self.worker.isFinished():
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
        else:
            print('else got called')
            self.worker.set_library_attributes(self.library_path,
                                     self.annotation_df,
                                     self.h5_folder_path,
                                     self.chosen_chunk_length,
                                     self.overwrite_bool,
                                     self.fs)

            self.worker.new_library_mode()
            self.worker.start()
            self.worker.wait()

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
        #todo catch not having the annotations and h5 files
        elif self.worker.isFinished():
            self.worker.set_library_attributes(self.library_path,
                                     self.annotation_df,
                                     self.h5_folder_path,
                                     self.chosen_chunk_length,
                                     self.overwrite_bool,
                                     self.fs)

            self.worker.append_to_library_mode()
            self.worker.start()
            self.worker.wait()
            print('Worker finished')
        else: # why do you have this - for running first time off??
            print('else got called in append to library')
            self.worker.set_library_attributes(self.library_path,
                                     self.annotation_df,
                                     self.h5_folder_path,
                                     self.chosen_chunk_length,
                                     self.overwrite_bool,
                                     self.fs)

            self.worker.append_to_library_mode()
            self.worker.start()
            self.worker.wait()

    def calculate_features_for_library(self):
        if self.library_path is None:
            self.select_library()
            self.update_library_path_display()

        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        elif self.worker.isFinished():
            self.worker.add_features_mode()
            self.worker.set_library_attributes_for_feats(self.library_path, self.chosen_chunk_length, self.overwrite_bool, self.use_peaks_bool)
            self.worker.start()
            print('Worker finished')
        else:
            print('else got called in labels to library')
            self.worker.add_features_mode()
            self.worker.set_library_attributes_for_feats(self.library_path, self.chosen_chunk_length, self.overwrite_bool, self.use_peaks_bool)
            self.worker.start()


    def calculate_labels_for_library(self):
        if self.library_path is None:
            self.select_library()
            self.update_library_path_display()

        if self.worker.isRunning():
            QtGui.QMessageBox.information(self, "Not implemented, lazy!", "Worker thread still running, please wait for previous orders to be finished!")
            return 0
        elif self.worker.isFinished():
            self.worker.add_labels_mode()
            self.worker.set_library_attributes_for_feats(self.library_path, self.chosen_chunk_length, self.overwrite_bool, self.use_peaks_bool)
            self.worker.start()
            print('Worker finished')

        else:
            print('else got called in labels to library')
            self.worker.add_labels_mode()
            self.worker.set_library_attributes_for_feats(self.library_path, self.chosen_chunk_length, self.overwrite_bool, self.use_peaks_bool)
            self.worker.start()

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
        self.wait()

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

        # this is gonna be a bit of a hack...
        if self.labels_or_features == False:
            if self.appending_to_library:
                self.update_progress_label.emit('Progress Bar is Frozen - no biggy')
                self.handler.append_to_seizure_library(df = self.annotations_df,
                                                       file_dir=self.h5_path,
                                                       seizure_library_path=self.library_path,
                                                       overwrite=self.overwrite_bool,
                                                       timewindow=self.t_len, fs=self.fs)
            else:
                #self.emit(pyqtSignal("update_progress_label(QString)"),'Progress Bar is Frozen - no biggy')
                self.update_progress_label.emit('Progress Bar is Frozen - no biggy')
                self.handler.make_seizure_library(df = self.annotations_df,
                                                       file_dir=self.h5_path,
                                                       seizure_library_name=self.library_path,
                                                       overwrite=self.overwrite_bool,
                                                       timewindow=self.t_len, fs=self.fs)
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
        self.exit()




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
        self.transmitter_ids.setText(str(8))
        self.h5directory = '/Volumes/G-DRIVE with Thunderbolt/test_h5'
        self.ndf_folder  = '/Volumes/G-DRIVE with Thunderbolt/test_ndfs'
        self.h5_display.setText(str(self.h5directory))
        self.ndf_display.setText(str(self.ndf_folder))


    def get_h5_folder(self):
        self.h5directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)
        self.h5_display.setText(str(self.h5directory))

    def get_ndf_folder(self):
        self.ndf_folder = QtGui.QFileDialog.getExistingDirectory(self, 'Select a ndf folder to convert', self.home)
        self.ndf_display.setText(str(self.ndf_folder))

    def convert_folder(self):
        tids = self.transmitter_ids.text()
        fs   = int(self.fs_box.text())
        ncores = self.cores_to_use.text()
        if ncores == 'all':
            ncores = -1
        else:
            ncores = int(ncores)

        if tids != 'all':
            tids = eval('['+tids+']')

        self.converting_thread = ConvertNdfThread()
        self.converting_thread.set_progress_bar.connect(self.update_progress)
        self.converting_thread.set_max_progress.connect( self.set_max_bar)
        self.converting_thread.update_hidden_label.connect( self.update_hidden)
        self.converting_thread.update_progress_label.connect( self.update_progress_label)
        #self.connect(self.converting_thread, pyqtSignal("update_hidden_label(QString)"), self.update_hidden)
        #self.connect(self.converting_thread, pyqtSignal("setMaximum_progressbar(QString)"), self.set_max_bar)
        #self.connect(self.converting_thread, pyqtSignal("SetProgressBar(QString)"), self.update_progress)
        #self.connect(self.converting_thread, pyqtSignal("update_progress_label(QString)"), self.update_progress_label)
        try:
            logfilepath = logging.getLoggerClass().root.handlers[0].baseFilename
            self.logpath_dsplay.setText(str(logfilepath))
        except:
            print('couldnt get logpath')
            #logfilepath = logging.getLoggerClass().root.handlers[0].baseFilename
        try:
            # pass
            # you've made a log file, let them know where it is!

            self.converting_thread.convert_ndf_directory_to_h5(ndf_dir=self.ndf_folder,
                                                save_dir=self.h5directory,
                                                tids=tids,
                                                n_cores=ncores,
                                                fs=fs)
            self.converting_thread.start()
        except:
            QtGui.QMessageBox.information(self,"Not implemented, lazy!", "Error!: /n Stop fucking around?! Currently set to re-run, so check errors in terminal")
            self.converting_thread.convert_ndf_directory_to_h5(ndf_dir=self.ndf_folder,
                                                save_dir=self.h5directory,
                                                tids=tids,
                                                n_cores=ncores,
                                                fs=fs)
            self.converting_thread.start()

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
        #self.set_max_progress = pyqtSignal(str)
        #self.update_hidden_label = pyqtSignal(str)
        #self.set_progress_bar =  pyqtSignal(str)
        #self.update_progress_label = pyqtSignal(str)

    def convert_ndf_directory_to_h5(self,
                                    ndf_dir,
                                    tids = 'all',
                                    save_dir  = 'same_level',
                                    n_cores = -1,
                                    fs = 'auto'):
        """
        Copy from datahandler, this should be a thread?
        """

        self.handler.fs_for_parallel_conversion = fs
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


        l = len(self.files)
        #self.emit(pyqtSignal("update_hidden_label(QString)"), str(len(self.files))+' Files for conversion. Transmitters: '+ str(self.handler.tids_for_parallel_conversion))
        #self.emit(pyqtSignal("setMaximum_progressbar(QString)"),str(len(self.files)))
        self.set_max_progress.emit(str(len(self.files)))
        self.update_hidden_label.emit(str(len(self.files))+' Files for conversion. Transmitters: '+ str(self.handler.tids_for_parallel_conversion))
    def run(self):
        pool = multiprocessing.Pool(self.n_cores)

        for i, _ in enumerate(pool.imap(self.handler.convert_ndf, self.files), 1):
            self.set_progress_bar.emit(str(i))
            self.update_progress_label.emit('Progress: ' +str(i)+ ' / '+ str(len(self.files)))
            #self.emit(pyqtSignal("SetProgressBar(QString)"),str(i))
            #self.emit(pyqtSignal("update_progress_label(QString)"),'Progress: ' +str(i)+ ' / '+ str(len(self.files)))
        pool.close()
        pool.join()
        #self.emit(pyqtSignal("update_progress_label(QString)"),'Progress: Done')
        self.update_progress_label.emit('Progress: Done')

class LoadingSubwindow(QtGui.QDialog, loading_subwindow.Ui_Dialog):
    ''' this is for the predictions, csv and h5 folder needed '''
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