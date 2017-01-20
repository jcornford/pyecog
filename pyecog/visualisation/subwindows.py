import sys
import os
import multiprocessing
import numpy as np
import pandas as pd
from PyQt4 import QtGui#,# uic
from PyQt4.QtCore import QThread, SIGNAL, Qt, QRect, QTimer
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

        self.library_path = None
        self.annotation_path = None
        self.h5_folder_path = None
        self.home = None # inherit default folder
        self.chosen_chunk_length = self.chunk_length.text()
        self.annotation_df = None
        self.overwrite_bool = self.overwrite_box.isChecked()
        self.use_peaks_bool = self.use_peaks.isChecked()

        #self.progressBar etc...
        #self.progressBar_label_above2

    def select_annotations_method(self):
        self.annotation_path = QtGui.QFileDialog.getOpenFileName(self, "Pick an annotations file", self.home)
        self.annotations_display.setText(self.annotation_path)
        # here check fileheadings are all good

    def select_h5_folder_method(self):
        self.h5_folder_path = QtGui.QFileDialog.getExistingDirectory(self, "Pick h5 folder", self.home)
        self.h5_folder_display.setText(self.h5_folder_path)

    def make_new_library(self):
        pass

    def append_to_library(self):
        pass

    def calculate_features_for_library(self):
        pass

    def calculate_labels_for_library(self):
        pass

    def chunk_len_changed(self):
        self.chosen_chunk_length = self.chunk_length.text()
        print(self.chosen_chunk_length)

    def use_peaks_changed(self):
        self.use_peaks_bool = self.use_peaks.isChecked()

    def overwrite_box_changed(self):
        self.overwrite_bool = self.overwrite_box.isChecked()

    def select_library(self):
        self.library_path = QtGui.QFileDialog.getOpenFileName(self, "Pick a Library file", self.home)
        self.update_library_path_display()

    def clear_library_path(self):
        self.library_path = ''
        self.update_library_path_display()

    def update_library_path_display(self):
        self.library_path_display.setText(self.library_path)


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
        self.connect(self.converting_thread, SIGNAL("update_hidden_label(QString)"), self.update_hidden)
        self.connect(self.converting_thread, SIGNAL("setMaximum_progressbar(QString)"), self.set_max_bar)
        self.connect(self.converting_thread, SIGNAL("SetProgressBar(QString)"), self.update_progress)
        self.connect(self.converting_thread, SIGNAL("update_progress_label(QString)"), self.update_progress_label)
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
    def __init__(self):
        QThread.__init__(self)
        self.handler = DataHandler()
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
        self.emit(SIGNAL("update_hidden_label(QString)"), str(len(self.files))+' Files for conversion. Transmitters: '+ str(self.handler.tids_for_parallel_conversion))
        self.emit(SIGNAL("setMaximum_progressbar(QString)"),str(len(self.files)))
    def run(self):
        pool = multiprocessing.Pool(self.n_cores)

        for i, _ in enumerate(pool.imap(self.handler.convert_ndf, self.files), 1):
            self.emit(SIGNAL("SetProgressBar(QString)"),str(i))
            self.emit(SIGNAL("update_progress_label(QString)"),'Progress: ' +str(i)+ ' / '+ str(len(self.files)))
        pool.close()
        pool.join()
        self.emit(SIGNAL("update_progress_label(QString)"),'Progress: Done')

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
        self.update_h5_folder_display()

    def update_h5_folder_display(self):
        self.h5_display.setText(str(self.h5directory))

    def update_predictionfile_display(self):
        self.prediction_display.setText(str(self.predictions_fname))

    def get_pred_filename(self):
        self.predictions_fname = QtGui.QFileDialog.getOpenFileName(self, 'Select a predicitons file', self.home)
        self.update_predictionfile_display()