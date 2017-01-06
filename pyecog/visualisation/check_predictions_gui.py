import sys
import os
import numpy as np
import pandas as pd
from PyQt4 import QtGui#,# uic
from PyQt4.QtCore import QThread, SIGNAL, Qt, QRect, QTimer
import pyqtgraph as pg
import inspect

from  pyecog.visualisation import check_preds_design
from pyecog.ndf.h5loader import H5File
#from pyecog.visualisation.pyqtgraph_playing import HDF5Plot

#TODO
# - you are currently loading the entire h5 file into memory..

class CheckPredictionsGui(QtGui.QMainWindow, check_preds_design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(CheckPredictionsGui, self).__init__(parent)
        self.setupUi(self)
        self.scroll_flag = -1
        if self.blink_box.isChecked():
            self.blink      = 1
        else:
            self.blink      = -1
        self.scroll_sign = 1
        self.timer = QTimer()
        self.timer.timeout.connect(self.simple_scroll)
        self.blink_box.stateChanged.connect(self.blink_box_change)
        self.scroll_speed_box.valueChanged.connect(self.scroll_speed_change)

        self.fs = None # change !
        self.data_obj = None
        self.predictions_df = None
        self.h5directory = None
        self.tree_items = []
        self.home = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan'

        self.select_folder_btn.clicked.connect(self.set_h5_folder)
        self.load_preds_btn.clicked.connect(self.load_pred_file)
        self.export_csv.clicked.connect(self.tree_export_csv)

        self.plot_1 = self.GraphicsLayoutWidget.addPlot()
        self.plot_overview = self.overview_plot.addPlot()
        #self.tid_box.setValue(6)
        #self.traceSelector.valueChanged.connect(self.plot_traces)
        #self.channel_selector.valueChanged.connect(self.plot_traces)
        self.treeWidget.itemSelectionChanged.connect(self.tree_selection)

        self.debug_load_pred_files()

        # Below resizes to better geometries - should really use this to save etc!
        # doesnt quite work correctly!
        '''
        self.widget_dict = {'top_container':self.horizontalLayout, 'plot':self.GraphicsLayoutWidget,
                       'button_box' :self.buttons_layout, 'tree':self.treeWidget}
        resized = {'top_container' : QRect(12, 12, 1049, 304),
                   'plot':QRect(368,0,681,304),
                   'tree':QRect(0, 0,361,304),
                   'button_box':QRect(12, 330,1049,36)}
        for key in list(self.widget_dict.keys()):
            self.widget_dict[key].setGeometry(resized[key])
        '''
        self.print_widget_coords = False # use this to print out coords when clicking the plot stuff


    def tree_export_csv(self):
        root = self.treeWidget.invisibleRootItem()
        child_count = root.childCount()
        print(child_count)
        index, start, end, tid, fname = [],[],[],[],[]

        for i in range(child_count):
            item = root.child(i)
            index.append(item.text(0)) # text at first (0) column
            start.append(item.text(1))
            end.append(item.text(2))
            tid.append(item.text(3))
            fname.append(item.text(4))
        exported_df = pd.DataFrame(data = np.vstack([index,fname,start,end,tid]).T,columns = ['old_index','filename','start','end','tid'] )
        #print(exported_df)

        if self.h5directory:
            default_dir = os.path.dirname(self.h5directory)
        else:
            default_dir = ""
        save_name = QtGui.QFileDialog.getSaveFileName(self,'Save annotation .csv file',default_dir)
        save_name = save_name.strip('.csv')
        exported_df.to_csv(save_name+'.csv')


    # this method does too much
    def tree_selection(self):
        "grab tree detail and use to plot"

        seizure_buffer = 5 # seconds either side of seizure to plot
        current_item = self.treeWidget.currentItem()
        #if current_item.text(0).endswith('.h5'):
        #    root = current_item
        #    fields = current_item.parent()
        #else:
        #    fields = current_item
        #    root = current_item.child(0)
        #   #print(fields.text(0))
        fields = current_item
        tid = int(float(fields.text(3)))
        start = int(float(fields.text(1)))
        end = int(float(fields.text(2)))
        index = int(float(fields.text(0)))
        fpath = os.path.join(self.h5directory, fields.text(4))

        h5 = H5File(fpath)
        data_dict = h5[tid]
        self.fs = eval(h5.attributes['fs_dict'])[tid]

        self.plot_1.clear()
        self.bx1 = self.plot_1.getViewBox()

        hdf5_plot = HDF5Plot(parent = self.plot_1, viewbox = self.bx1)
        hdf5_plot.setHDF5(data_dict['data'], data_dict['time'], self.fs)
        self.plot_1.addItem(hdf5_plot)

        start_pen = pg.mkPen((85, 168, 104), width=3, style= Qt.DashLine)
        end_pen = pg.mkPen((210,88,88), width=3, style= Qt.DashLine)
        self.start_line = pg.InfiniteLine(pos=start, pen =start_pen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
        self.end_line = pg.InfiniteLine(pos=end, pen = end_pen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
        self.plot_1.addItem(self.start_line)
        self.plot_1.addItem(self.end_line)
        self.start_line.sigPositionChanged.connect(self.update_tree_element_start_time)
        self.end_line.sigPositionChanged.connect(self.update_tree_element_end_time)

        self.plot_1.setXRange(start-seizure_buffer, end+seizure_buffer)
        self.plot_1.setTitle(str(index)+' - '+ fields.text(4)+ '\n' + str(start)+' - ' +str(end))
        self.plot_1.setLabel('left', 'Voltage (uV)')
        self.plot_1.setLabel('bottom','Time (s)')

        # hit up the linked view here
        self.plot_overview.clear()
        self.plot_overview.enableAutoRange(False,False)
        self.plot_overview.setXRange(0,3600) # hardcoding in the hour here...
        self.plot_overview.setMouseEnabled(x = False, y= True)
        self.bx_overview = self.plot_overview.getViewBox()
        hdf5_plotoverview = HDF5Plot(parent = self.plot_overview, viewbox = self.bx_overview)
        hdf5_plotoverview.setHDF5(data_dict['data'], data_dict['time'], self.fs)
        self.plot_overview.addItem(hdf5_plotoverview)
        self.plot_overview.setLabel('left', 'Voltage (uV)')
        self.plot_overview.setLabel('bottom','Time (s)')
        self.plot_overview.setTitle('Overview of file: '+str(index)+' - '+ fields.text(4))


        self.lr = pg.LinearRegionItem(self.plot_1.getViewBox().viewRange()[0])
        self.lr.setZValue(-10)
        self.plot_overview.addItem(self.lr)
        # is this good practice?

        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.plot_1.sigXRangeChanged.connect(self.updateRegion)
        self.updatePlot()

        # not part of the method, was silly stuff to reset  the gui when reloading
        if self.print_widget_coords:
            for key in list(self.widget_dict.keys()):
                print(key)
                obj = self.widget_dict[key]
                print(obj.geometry().x(),obj.geometry().y())
                #print(object.frameGeometry())
                print(obj.geometry().width())
                print(obj.geometry().height())
                print('******')
        #self.plot_traces()
    # these two methods are for the lr plot connection, refactor names
    def updatePlot(self):
        self.plot_1.setXRange(*self.lr.getRegion(), padding=0)
    def updateRegion(self):
        self.lr.setRegion(self.plot_1.getViewBox().viewRange()[0])

    def update_tree_element_start_time(self):
        tree_row = self.treeWidget.currentItem()
        tree_row.setText(1,'{:.2f}'.format(self.start_line.x()))

    def update_tree_element_end_time(self):
        tree_row = self.treeWidget.currentItem()
        tree_row.setText(2,'{:.2f}'.format(self.end_line.x()))

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

    def clear_QTreeWidget(self):
            root = self.treeWidget.invisibleRootItem()
            n_kids = root.childCount()
            print(n_kids, 'n kids')
            for i in range(n_kids):
                #child = root.takeChild(i)
                child = self.treeWidget.itemAt(i)
                root.removeChild(child)


    def populate_tree(self, row, tids):

        #self.treeWidget.setColumnCount(1)
        self.treeWidget.setColumnCount(5)
        #self.treeWidget.setFirstColumnSpanned(True)
        self.treeWidget.setHeaderLabels(['index', 'start', 'end', 'tid', 'fname'])
        filename = row['Filename']
        index =  row['Index']
        start =  row['Start']
        end = row['End']
         # flipped these two to read better
        fname_entry = [str(filename)]
        details_entry = [str(index), str(start), str(end),str(tids[0]),str(filename)] # bad, should make only having one explcit
        item = QtGui.QTreeWidgetItem(details_entry)
        item.setFirstColumnSpanned(True)

        #item.addChild(QtGui.QTreeWidgetItem(fname_entry))
        self.tree_items.append(item)

        self.treeWidget.addTopLevelItems(self.tree_items)

    #TODO implement this clicking stuff! click and move the region
    def print_mouse_position(self, pos):
        # not used
        mousepoint = self.plot_1.getViewBox().mapSceneToView(pos)
        print(mousepoint)
        self.mouseline.setPos(mousepoint.x())
        #self.coords_label.setText("<span style='font-size: 6pt'> x=%0.1f <div style='text-align: center'> , <span style='font-size: 6pt''color: red'>y=%0.1f</span>" % (1111,2222))
        #self.coords_label.setText(str(mousepoint.x()))

    def mousePressEvent(self, QMouseEvent):
        #print('click!')
        #event = QMouseEvent
        #position = QMouseEvent.pos()
        #print(position)
        print('gloabl mouse position is...',QMouseEvent.globalPos())

        #print(self.plot_1.sceneBoundingRect())
        #print(self.plot_1.getViewBox().sceneBoundingRect())
        #if self.plot_1.getViewBox().sceneBoundingRect().contains(position):
        #    print('plot 1 contains postion!')
        #print(QMouseEvent.posF())
        #mousepoint = self.plot_1.getViewBox().mapSceneToView(position)
        #print(mousepoint)
        #region_coords = self.plot_1.getViewBox().viewRange()[0]
        #print(region_coords)
        #self.plot_1.setXRange(QMouseEvent.x(), region_coords[1])

    def keyPressEvent(self, eventQKeyEvent):
        key = eventQKeyEvent.key()

        x,y = self.plot_1.getViewBox().viewRange()
        if key == Qt.Key_Up:
            if self.scroll_flag==True:
                scroll_rate = self.scroll_speed_box.value()
                new_rate = scroll_rate * 2
                self.scroll_speed_box.setValue(new_rate)
                if self.blink ==True: self.reset_timer()
            else:
                self.plot_1.getViewBox().setYRange(min = y[0]*0.9, max = y[1]*0.9, padding = 0)

        if key == Qt.Key_Down:
            if self.scroll_flag==True:
                scroll_rate = self.scroll_speed_box.value()
                if scroll_rate > 1:
                    new_rate = int(scroll_rate / 2)
                    self.scroll_speed_box.setValue(new_rate)
                    if self.blink ==True: self.reset_timer()
            else: # just zoom
                self.plot_1.getViewBox().setYRange(min = y[0]*1.1, max = y[1]*1.1,padding = 0)

        if key == Qt.Key_Right:
            if self.scroll_flag==True:
                self.scroll_sign = 1
            else:
                scroll_i = (x[1]-x[0])*0.001
                self.plot_1.getViewBox().setXRange(min = x[0]+scroll_i, max = x[1]+scroll_i, padding=0)

        if key == Qt.Key_Left:
            if self.scroll_flag==True:
                self.scroll_sign = -1
            else:
                scroll_i = (x[1]-x[0])*0.001
                self.plot_1.getViewBox().setXRange(min = x[0]-scroll_i, max = x[1]-scroll_i, padding=0)

        if key == Qt.Key_Backspace or key == Qt.Key_Delete:
            current_item = self.treeWidget.currentItem()
            root = self.treeWidget.invisibleRootItem()
            root.removeChild(current_item)

        if key == Qt.Key_B:
            if self.scroll_flag==True:
                self.blink *= -1
                if self.blink == True:
                    self.blink_box.setChecked(True)
                else:
                    self.blink_box.setChecked(False)
                self.reset_timer()

        if key == Qt.Key_Space:
            self.scroll_sign = 1
            self.scroll_flag *= -1
            self.reset_timer()
# TODO call this when blink or scroll boxa are changed through the gui, not just buttons
    def blink_box_change(self):
        self.reset_timer()
        #print('someone changed the blink box')
    def scroll_speed_change(self):
        self.reset_timer()

    def reset_timer(self):
        scroll_rate = self.scroll_speed_box.value()
        if self.scroll_flag==True:
            self.timer.stop()
            if self.blink_box.isChecked():
                rate = int(2000/scroll_rate)
                #print(rate)
                self.timer.start(rate)
            else:
                self.timer.start(20)
        else:
            self.timer.stop()

    def simple_scroll(self):
        x,y = self.plot_1.getViewBox().viewRange()
        scroll_rate = self.scroll_speed_box.value()
        if self.blink_box.isChecked() != True:
            scroll_i = (x[1]-x[0])*(0.001*scroll_rate)*self.scroll_sign
            self.plot_1.getViewBox().setXRange(min = x[0]+scroll_i, max = x[1]+scroll_i, padding=0)
        elif self.blink_box.isChecked():
            scroll_i = (x[1]-x[0])*self.scroll_sign
            self.plot_1.getViewBox().setXRange(min = x[0]+scroll_i, max = x[1]+scroll_i, padding=0)

    def load_pred_file(self):
        #self.clear_QTreeWidget()
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
            #TODO : decide what to do with tids, this is not thought out at the moment
            #So not bothering to load the tids here as should be one only per seizure... can either  load on demand
            # or use only for the data explorer stuff. Can maybe have dynamic when you click to see the full tree.
            # problem is with files with many false positives, spend time loading for too long!

            # or have a button for this...

            #h5 = H5File(fpath)
            #tids = h5.attributes['t_ids']
            tids = [int(fpath.split(']')[0].split('[')[1])]
            #tids = [int(fpath.split(']')[0].split]
            s,e = row['Start'], row['End']
            self.populate_tree(row, tids)

    def debug_load_pred_files(self): # stripped down version of the above for debugging gui code
        self.h5directory = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan/2016_10/M6'
        fname = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan/2016_10/clf_predictions_m6_201610_no_pks.csv'
        self.predictions_df = pd.read_csv(fname)
        self.predictions_df['Index'] = self.predictions_df.index
        for i,row in list(self.predictions_df.iterrows()):
            fpath = os.path.join(self.h5directory,row['Filename'])
            tids = [int(fpath.split(']')[0].split('[')[1])]
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
    form = CheckPredictionsGui()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()