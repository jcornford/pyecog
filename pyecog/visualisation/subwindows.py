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
from pyecog.visualisation import check_preds_design, loading_subwindow, convert_ndf_window
from pyecog.ndf.h5loader import H5File
from pyecog.ndf.datahandler import DataHandler, NdfFile


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
            self.time_elapsed.setText(str(logfilepath))
        except:
            print('couldnt get logpath')
            logfilepath = logging.getLoggerClass().root.handlers[0].baseFilename
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
            QtGui.QMessageBox.information(self,"Not implemented, lazy!", "Errors: /n Stop fucking around!")
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