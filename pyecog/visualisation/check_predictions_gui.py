import sys
import os
import numpy as np
import pandas as pd
from PyQt4 import QtGui#,# uic
from PyQt4.QtCore import QThread, SIGNAL
import pyqtgraph as pg

from  pyecog.visualisation import check_preds_design

from pyecog.ndf.h5loader import H5File
from pyecog.visualisation.pyqtgraph_playing import HDF5Plot

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
        #self.traceSelector.valueChanged.connect(self.plot_traces)
        #self.channel_selector.valueChanged.connect(self.plot_traces)
        self.treeWidget.itemSelectionChanged.connect(self.tree_selection)

        #self.load_pred_file('/Users/jonathan/PhD/Data/PV_dendritic_intergration/Data/2016_05/2016_05_11/2016_05_11_slice1_cell1_cf2.ASC')
    def tree_selection(self):
        "this is messy"

        current_item = self.treeWidget.currentItem()
        if current_item.text(0).endswith('.h5'):
            root = current_item
            fields = current_item.child(0)
        else:
            fields = current_item
            root = current_item.parent()
            print(fields.text(0))
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
        self.plot_1.addItem(pg.PlotCurveItem(data_dict['time'], data_dict['data']))

        self.plot_1.setXRange(start, end)
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
        item = QtGui.QTreeWidgetItem([str(filename)])
        item.setFirstColumnSpanned(True)

        for tid in tids:
            #string_ =
            item.addChild(QtGui.QTreeWidgetItem([str(index), str(start), str(end),str(tid)]))
        self.tree_items.append(item)

        self.treeWidget.addTopLevelItems(self.tree_items)

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
            h5 = H5File(fpath)
            tids = h5.attributes['t_ids']
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


def main():
    app = QtGui.QApplication(sys.argv)
    form = PVio()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()