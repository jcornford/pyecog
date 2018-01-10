import sys
import os
import bisect
import traceback
import time

import numpy as np
import pandas as pd
import pickle as p

from PyQt5 import QtGui, QtWidgets#,# uic
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QTimer
from scipy import signal, stats
import pyqtgraph as pg
import inspect
import h5py

def throw_error(error_text = None):
    msgBox = QtWidgets.QMessageBox()
    if error_text is None:
        msgBox.setText('Error caught! \n'+str(traceback.format_exc(1)))
    else:
        msgBox.setText('Error caught! \n'+str(error_text))
    msgBox.exec_()
    return 0


class HDF5Plot(pg.PlotCurveItem):
    """
    Create a subclass of PlotCurveItem for displaying a very large
    data set from an HDF5 file that does not neccesarilly fit in memory.

    The basic approach is to override PlotCurveItem.viewRangeChanged such that it
    reads only the portion of the HDF5 data that is necessary to display the visible
    portion of the data. This is further downsampled to reduce the number of samples
    being displayed.

    A more clever implementation of this class would employ some kind of caching
    to avoid re-reading the entire visible waveform at every update.
    """
    def __init__(self, downsample_limit = 20000,viewbox = None, *args, **kwds):
        " TODO what are the args and kwds for PlotCurveItem class?"
        self.hdf5 = None
        self.hdf5_filtered_data = None
        self.time = None
        self.fs = None
        self.vb = viewbox
        self.limit = downsample_limit # maximum number of samples to be plotted, 10000 orginally
        self.display_filter = None
        self.hp_cutoff = None
        self.lp_cutoff = None
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        if pg.CONFIG_OPTIONS['background'] == 'w':
            self.pen = (0,0,0)
        else:
            self.pen = (255,255,255)


    def keyPressEvent(self, event):
        ''' this doesnt work, change key press to correct it.'''
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            pass
            #print(event.key())

    def setHDF5(self, data, time, fs):
        self.hdf5 = data
        self.time = time
        self.fs = fs
        #print ( self.hdf5.shape, self.time.shape)
        self.updateHDF5Plot()

    def set_display_filter_settings(self, display_filter, hp_cutoff, lp_cutoff, hp_toggle, lp_toggle):
        self.display_filter = display_filter
        self.hp_cutoff = hp_cutoff
        self.lp_cutoff = lp_cutoff
        self.hp_toggle = hp_toggle
        self.lp_toggle = lp_toggle


    def highpass_filter(self, data):
        '''
        Implements high pass digital butterworth filter, order 2.
        '''

        nyq = 0.5 * self.fs
        cutoff_decimal = self.hp_cutoff/nyq

        b, a = signal.butter(2, cutoff_decimal, 'highpass', analog=False)

        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    def lowpass_filter(self, data):
        '''
        Implements low pass digital butterworth filter, order 2.
        '''

        nyq = 0.5 * self.fs
        cutoff_decimal = self.lp_cutoff/nyq

        b, a = signal.butter(2, cutoff_decimal, 'lowpass', analog=False)
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    def wipe_filtered_data(self):
        self.hdf5_filtered_data = None

    def viewRangeChanged(self):
        self.updateHDF5Plot()

    def updateHDF5Plot(self):
        if self.hdf5 is None:
            self.setData([])
            return 0

        if self.display_filter:
            if self.hdf5_filtered_data is None:
                self.hdf5_filtered_data = self.hdf5
                if self.hp_toggle:
                    try: self.hdf5_filtered_data = self.highpass_filter(self.hdf5_filtered_data)
                    except: throw_error()
                if self.lp_toggle:
                    try: self.hdf5_filtered_data = self.lowpass_filter(self.hdf5_filtered_data)
                    except: throw_error()

            hdf5data = self.hdf5_filtered_data

        else:
            hdf5data = self.hdf5
        #vb = self.getViewBox()
        #if vb is None:
        #    return  # no ViewBox yet

        # Determine what data range must be read from HDF5
        xrange = [i*self.fs for i in self.vb.viewRange()[0]]
        start = max(0,int(xrange[0])-1)
        stop = min(len(hdf5data), int(xrange[1]+2))
        if stop-start < 1:
            print('didnt update')
            return 0
        # Decide by how much we should downsample
        ds = int((stop-start) / self.limit) + 1
        if ds == 1:
            # Small enough to display with no intervention.
            visible_y = hdf5data[start:stop]
            visible_x = self.time[start:stop]
            scale = 1
        else:
            # Here convert data into a down-sampled array suitable for visualizing.
            # Must do this piecewise to limit memory usage.
            samples = 1 + ((stop-start) // ds)
            visible_y = np.zeros(samples*2, dtype=hdf5data.dtype)
            visible_x = np.zeros(samples*2, dtype=self.time.dtype)
            sourcePtr = start
            targetPtr = 0

            # read data in chunks of ~1M samples
            chunkSize = (1000000//ds) * ds
            while sourcePtr < stop-1:

                chunk = hdf5data[sourcePtr:min(stop,sourcePtr+chunkSize)]
                chunk_x = self.time[sourcePtr:min(stop,sourcePtr+chunkSize)]
                sourcePtr += len(chunk)
                #print(chunk.shape, chunk_x.shape)

                # reshape chunk to be integral multiple of ds
                chunk = chunk[:(len(chunk)//ds) * ds].reshape(len(chunk)//ds, ds)
                chunk_x = chunk_x[:(len(chunk_x)//ds) * ds].reshape(len(chunk_x)//ds, ds)

                # compute max and min
                #chunkMax = chunk.max(axis=1)
                #chunkMin = chunk.min(axis=1)

                mx_inds = np.argmax(chunk, axis=1)
                mi_inds = np.argmin(chunk, axis=1)
                row_inds = np.arange(chunk.shape[0])

                chunkMax = chunk[row_inds, mx_inds]
                chunkMin = chunk[row_inds, mi_inds]
                chunkMax_x = chunk_x[row_inds, mx_inds]
                chunkMin_x = chunk_x[row_inds, mi_inds]

                # interleave min and max into plot data to preserve envelope shape
                visible_y[targetPtr:targetPtr+chunk.shape[0]*2:2] = chunkMin
                visible_y[1+targetPtr:1+targetPtr+chunk.shape[0]*2:2] = chunkMax
                visible_x[targetPtr:targetPtr+chunk_x.shape[0]*2:2] = chunkMin_x
                visible_x[1+targetPtr:1+targetPtr+chunk_x.shape[0]*2:2] = chunkMax_x

                targetPtr += chunk.shape[0]*2

            visible_x = visible_x[:targetPtr]
            visible_y = visible_y[:targetPtr]
            #print('**** now downsampling')
            #print(visible_y.shape, visible_x.shape)
            scale = ds * 0.5

        # TODO: setPos, scale, resetTransform methods... scale?

        self.setData(visible_x, visible_y, pen=self.pen) # update the plot
        #self.setPos(start, 0) # shift to match starting index ### Had comment out to stop it breaking... when limit is >0?!
        self.resetTransform()
        #self.scale(scale, 1)  # scale to match downsampling
