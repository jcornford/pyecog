import sys
import os
import numpy as np
import pandas as pd
from PyQt4 import QtGui#,# uic
from PyQt4.QtCore import QThread, SIGNAL, Qt
import pyqtgraph as pg

from  pyecog.visualisation import check_preds_design

from pyecog.ndf.h5loader import H5File
#from pyecog.visualisation.pyqtgraph_playing import HDF5Plot

#TODO
# - you are currently loading the entire h5 file into memory..

class PVio(QtGui.QMainWindow, check_preds_design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(PVio, self).__init__(parent)
        self.setupUi(self)

        self.fs = 256 # change !
        self.data_obj = None
        self.predictions_df = None
        self.h5directory = None
        self.tree_items = []
        self.home = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan'

        self.select_folder_btn.clicked.connect(self.set_h5_folder)
        self.load_preds_btn.clicked.connect(self.load_pred_file)

        self.plot_1 = self.GraphicsLayoutWidget.addPlot()
        self.tid_box.setValue(6)
        #self.traceSelector.valueChanged.connect(self.plot_traces)
        #self.channel_selector.valueChanged.connect(self.plot_traces)
        self.treeWidget.itemSelectionChanged.connect(self.tree_selection)

        #self.plot_1.scene().sigMouseMoved.connect(self.print_mouse_position)
        #self.mouseline = pg.InfiniteLine(pos = None)
        #self.plot_1.addItem(self.mouseline, ignoreBounds = True)
        #self.coords_label = pg.TextItem()
        #self.plot_1.addItem(self.coords_label)

        self.debug_load_pred_files()
    def tree_selection(self):
        "grab tree detail and use to plot"

        seizure_buffer = 5 # seconds either side of seizure to plot
        current_item = self.treeWidget.currentItem()
        if current_item.text(0).endswith('.h5'):
            root = current_item
            fields = current_item.parent()
        else:
            fields = current_item
            root = current_item.child(0)
            #print(fields.text(0))
        tid = int(float(fields.text(3)))
        start = int(float(fields.text(1)))
        end = int(float(fields.text(2)))
        index = int(float(fields.text(0)))
        fpath = os.path.join(self.h5directory, root.text(0))

        h5 = H5File(fpath)
        data_dict = h5[tid]

        #if not self.holdPlot.isChecked():
        self.plot_1.clear()
        self.bx1 = self.plot_1.getViewBox()

        hdf5_plot = HDF5Plot(parent = self.plot_1, viewbox = self.bx1)
        hdf5_plot.setHDF5(data_dict['data'], data_dict['time'], self.fs)
        self.plot_1.addItem(hdf5_plot)
        lpen = pg.mkPen((255,0,0), width=2, style= Qt.DashLine)
        self.plot_1.addItem(pg.InfiniteLine(pos=start, pen =lpen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,50), 'movable': True}))
        self.plot_1.addItem(pg.InfiniteLine(pos=end, pen = lpen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,50), 'movable': True}))

        self.plot_1.setXRange(start-seizure_buffer, end+seizure_buffer)
        self.plot_1.setTitle(str(index)+' - '+ root.text(0)+ '\n' + str(start)+' - ' +str(end))
        self.plot_1.setLabel('left', 'Voltage (uV)')
        self.plot_1.setLabel('bottom','Time (s)')
        #self.plot_traces()

    def plot_traces(self, data_dict):

        if not self.holdPlot.isChecked():
            self.plot_1.clear()
        # here you need to add the h5 file class with downsampling

        curve1 = HDF5Plot()#parent = self.plot_1, viewbox = bx1)
        curve1.setHDF5(data_dict['data'], data_dict['time'], self.fs)
        self.plot_1.addItem(hdf5_plot)

        #self.plot_1.addItem(pg.PlotCurveItem(data_dict['time'], data_dict['data']))
        self.plot_1.setXRange(row['Start'], row['End'])
        #self.plot_1.ti

    def set_h5_folder(self):
        self.h5directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)

    def populate_tree(self, row, tids):
        #self.treeWidget.setColumnCount(1)
        self.treeWidget.setColumnCount(4)
        #self.treeWidget.setFirstColumnSpanned(True)
        self.treeWidget.setHeaderLabels(['index', 'start', 'end', 'tid'])
        filename = row['Filename']
        index =  row['Index']
        start =  row['Start']
        end = row['End']
         # flipped these two to read better
        fname_entry = [str(filename)]
        details_entry = [str(index), str(start), str(end),str(tids[0])] # bad, should make only having one explcit
        item = QtGui.QTreeWidgetItem(details_entry)
        item.setFirstColumnSpanned(True)


        item.addChild(QtGui.QTreeWidgetItem(fname_entry))
        self.tree_items.append(item)

        self.treeWidget.addTopLevelItems(self.tree_items)

    def print_mouse_position(self, pos):
        mousepoint = self.plot_1.getViewBox().mapSceneToView(pos)
        print(mousepoint)
        self.mouseline.setPos(mousepoint.x())
        #self.coords_label.setText("<span style='font-size: 6pt'> x=%0.1f <div style='text-align: center'> , <span style='font-size: 6pt''color: red'>y=%0.1f</span>" % (1111,2222))
        #self.coords_label.setText(str(mousepoint.x()))

    def keyPressEvent(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        x,y = self.plot_1.getViewBox().viewRange()
        if key == Qt.Key_Up:
            self.plot_1.getViewBox().setYRange(min = y[0]*0.9, max = y[1]*0.9, padding = 0)
        if key == Qt.Key_Down:
            self.plot_1.getViewBox().setYRange(min = y[0]*1.1, max = y[1]*1.1,padding = 0)
        if key == Qt.Key_Right:
            scroll_i = (x[1]-x[0])*0.05
            self.plot_1.getViewBox().setXRange(min = x[0]+scroll_i, max = x[1]+scroll_i, padding=0)
        if key == Qt.Key_Left:
            scroll_i = (x[1]-x[0])*0.05
            self.plot_1.getViewBox().setXRange(min = x[0]-scroll_i, max = x[1]-scroll_i, padding=0)
        #if type(event) == QtGui.QKeyEvent:
            #print(event.key())
            #event.accept()

    def load_pred_file(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'select predicitons file', self.home)
        if fname.endswith('.csv'):
            self.predictions_df = pd.read_csv(fname)
        elif fname.endswith('.xlsx'):
            self.predictions_df = pd.read_excel(fname)
        else:
            print('Please select .csv or .xlsx file')
            return 0
        self.predictions_df['Index'] = self.predictions_df.index
        if self.h5directory is None:
            self.set_h5_folder()

        for i,row in list(self.predictions_df.iterrows()):
            fpath = os.path.join(self.h5directory,row['Filename'])
            #TODO
            #So not bothering to load the tids here as should be one only per seizure... can either  load on demand
            # or use only for the data explorer stuff. Can maybe have dynamic when you click to see the full tree.
            # problem is with files with many false positives, spend time loading for too long!

            # or have a button for this...

            #h5 = H5File(fpath)
            #tids = h5.attributes['t_ids']
            tids = [self.tid_box.value()]
            s,e = row['Start'], row['End']
            self.populate_tree(row, tids)

    def debug_load_pred_files(self): # stripped down version of the above for debugging gui code
        self.h5directory = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan/2016_10/M6'
        fname = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan/2016_10/clf_predictions_m6_201610_no_pks.csv'
        self.predictions_df = pd.read_csv(fname)
        self.predictions_df['Index'] = self.predictions_df.index
        for i,row in list(self.predictions_df.iterrows()):
            fpath = os.path.join(self.h5directory,row['Filename'])
            tids = [self.tid_box.value()]
            s,e = row['Start'], row['End']
            self.populate_tree(row, tids)

    def load_h5_file(self,fname):

        self.loading_thread = LoadFileThread(fname)
        #self.connect(self.loading_thread, SIGNAL("finished()"), self.done)
        self.connect(self.loading_thread, SIGNAL("catch_data(PyQt_PyObject)"), self.catch_data)
        self.loading_thread.start()

    def catch_data(self, h5obj):
        self.h5obj = h5obj
        #print(data_obj)
        #print(data_obj.channel_0[:,5])
        #print(data_obj.channel_0[:,5].shape)
        self.plot_traces()


    def done(self):
        QtGui.QMessageBox.information(self, "Done!", "Done loading!")


class LoadFileThread(QThread):

    def __init__(self, filename):
        QThread.__init__(self)
        self.filename = filename

    def __del__(self):
        self.wait()

    def load_file(self, filename):
        self.h5obj = H5File(filename)

    def run(self):
        print('sup, loading: '+self.filename)
        self.load_file(self.filename)
        self.emit(SIGNAL('catch_data(PyQt_PyObject)'), self.h5obj)

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


def main():
    app = QtGui.QApplication(sys.argv)
    form = PVio()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()