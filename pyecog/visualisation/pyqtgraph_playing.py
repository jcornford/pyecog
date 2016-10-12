# -*- coding: utf-8 -*-
"""
Plans:

** Dont understand the difference bewtween views, layout etc, documentation seems very hard to find
Lets just make miminum viable thing - scrolling and ability to load and select

 - left right arrow
 - space bar scrolling
 - nested tids/ seizure datasets
 - fft
 - animal video space
 - larger bottom
 - disable zooms on smaller
 - be able to navigate between files...

 Other examples...:
  - scrollling
  - nested
  - dialogue boxes
  - graph view...


"""

#import initExample ## Add path to library (just for examples; you do not need this)
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import h5py
import sys, os

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
        self.time = None
        self.fs = None
        self.vb = viewbox
        self.limit = downsample_limit # maximum number of samples to be plotted, 10000 orginally
        pg.PlotCurveItem.__init__(self, *args, **kwds)


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            print(key)

    def setHDF5(self, data, time, fs):
        self.hdf5 = data
        self.time = time
        self.fs = fs
        #print ( self.hdf5.shape, self.time.shape)
        self.updateHDF5Plot()

    def viewRangeChanged(self):
        self.updateHDF5Plot()

    def updateHDF5Plot(self):
        if self.hdf5 is None:
            self.setData([])
            return

        #vb = self.getViewBox()
        #if vb is None:
        #    return  # no ViewBox yet

        # Determine what data range must be read from HDF5
        xrange = [i*self.fs for i in self.vb.viewRange()[0]]
        start = max(0,int(xrange[0])-1)
        stop = min(len(self.hdf5), int(xrange[1]+2))

        # Decide by how much we should downsample
        ds = int((stop-start) / self.limit) + 1
        if ds == 1:
            # Small enough to display with no intervention.
            visible_y = self.hdf5[start:stop]
            visible_x = self.time[start:stop]
            scale = 1
        else:
            # Here convert data into a down-sampled array suitable for visualizing.
            # Must do this piecewise to limit memory usage.
            samples = 1 + ((stop-start) // ds)
            visible_y = np.zeros(samples*2, dtype=self.hdf5.dtype)
            visible_x = np.zeros(samples*2, dtype=self.time.dtype)
            sourcePtr = start
            targetPtr = 0

            # read data in chunks of ~1M samples
            chunkSize = (1000000//ds) * ds
            while sourcePtr < stop-1:

                chunk = self.hdf5[sourcePtr:min(stop,sourcePtr+chunkSize)]
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
        self.setData(visible_x, visible_y) # update the plot
        #self.setPos(start, 0) # shift to match starting index ### Had comment out to stop it breaking... when limit is >0?!
        self.resetTransform()
        #self.scale(scale, 1)  # scale to match downsampling

def load_library(path):
    "load seizure library files"
    pass

def load_ndf(path):
    pass

def load_h5(path):
    "flat h5 for predictions"
    pass

def createFile(finalSize=2000000000):
    """Create a large HDF5 data file for testing.
    Data consists of 1M random samples tiled through the end of the array.
    """

    chunk = np.random.normal(size=1000000).astype(np.float32)

    f = h5py.File('test.hdf5', 'w')
    f.create_dataset('data', data=chunk, chunks=True, maxshape=(None,))
    data = f['data']

    nChunks = finalSize // (chunk.size * chunk.itemsize)
    with pg.ProgressDialog("Generating test.hdf5...", 0, nChunks) as dlg:
        for i in range(nChunks):
            newshape = [data.shape[0] + chunk.shape[0]]
            data.resize(newshape)
            data[-chunk.shape[0]:] = chunk
            dlg += 1
            if dlg.wasCanceled():
                f.close()
                os.remove('test.hdf5')
                sys.exit()
        dlg += 1
    f.close()


fileName = '/Users/jonathan/Dropbox/gui_pyqt_dev/gl_library_for_vannila_features.h5'
f = h5py.File(fileName, 'r')


i = 0
datasets = [f[key] for key in list(f.keys())]
data = datasets[i]['data'][:]  # check - dont need to slice here? or do i?
# think i have broken the larger than memory part of this...
flat_data = np.ravel(data, order = 'C')
fs = datasets[i].attrs['fs']
time = np.arange(0, 3600*fs)/fs# assuming it is an hour
time = np.reshape(time, newshape=(data.shape), order = 'C') # should be same shape!
flat_time = np.ravel(time, order = 'C')


#pg.mkQApp()
app = QtGui.QApplication([])
#view = pg.GraphicsView()
#win = pg.GraphicsLayout()
#view.setCentralItem(win)

win = pg.GraphicsWindow()

win.resize(1000,500)
win.setWindowTitle('PyECoG ')


#bx2 = win.addViewBox(row = 1, col=1, colspan = 3)
plt1 = win.addPlot(row=1, col=1,colspan = 3, title = 'Overview ... ')
plt1.enableAutoRange(False, False)
plt1.setXRange(0, 3600)
plt1.setMouseEnabled(x = False, y = True)
bx1 = plt1.getViewBox()
curve1 = HDF5Plot(parent = plt1, viewbox = bx1)
curve1.setHDF5(flat_data, flat_time, fs)
lr = pg.LinearRegionItem([0,20])
lr.setZValue(-10)
plt1.addItem(lr)
plt1.addItem(curve1)

vid = win.addViewBox(lockAspect = True, row = 1, col = 4)
img = pg.ImageItem(np.random.normal(size=(150,150)))
vid.addItem(img)
vid.autoRange()

win.nextRow()

plt = win.addPlot(row=2, col=1 , colspan = 4, title = 'tid ... ')
plt.enableAutoRange(False, False)
plt.setXRange(0, 500)
bx = plt.getViewBox()
curve = HDF5Plot(parent = plt, viewbox = bx)
curve.setHDF5(flat_data, flat_time, fs)
plt.addItem(curve)

def updatePlot():
    plt.setXRange(*lr.getRegion(), padding=0)
def updateRegion():
    lr.setRegion(plt.getViewBox().viewRange()[0])
lr.sigRegionChanged.connect(updatePlot)
plt.sigXRangeChanged.connect(updateRegion)
updatePlot()



def scroll():
    data1[:-1] = data1[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
    data1[-1] = np.random.normal()
    curve1.setData(data1)

    ptr1 += 1
    curve2.setData(data1)
    curve2.setPos(ptr1, 0)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()