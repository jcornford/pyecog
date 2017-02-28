import sys
import os
import numpy as np
import pandas as pd
from PyQt5 import QtGui#,# uic
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QTimer
import pyqtgraph as pg
import inspect
import h5py

# todo test if these work without being called from main_gui at pyecog level
from . import check_preds_design, loading_subwindow, convert_ndf_window
from ndf.h5loader import H5File
from . import subwindows

#from ndf.datahandler import DataHandler
#from pyecog.visualisation.pyqtgraph_playing import HDF5Plot

#TODO - you are currently loading the entire h5 file into memory..
class MainGui(QtGui.QMainWindow, check_preds_design.Ui_MainWindow):
    def __init__(self, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        super(MainGui, self).__init__(parent)
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

        self.home = os.getcwd()
        #self.home = '/Volumes/G-DRIVE with Thunderbolt/'

        #self.select_folder_btn.clicked.connect(self.set_h5_folder)
        #self.load_preds_btn.clicked.connect(self.load_pred_file)

        # Hook up the file bar stuff here
        self.actionLoad_Predictions.triggered.connect(self.load_predictions_gui)
        self.actionSave_annotations.triggered.connect(self.master_tree_export_csv)
        self.actionLoad_Library.triggered.connect(self.load_seizure_library)
        self.actionLoad_h5_folder.triggered.connect(self.not_done_yet) # this is still to do in its entireity
        self.actionSet_default_folder.triggered.connect(self.set_home)

        # Hook up analyse menu bar to functions here
        self.actionConvert_ndf_to_h5.triggered.connect(self.convert_ndf_folder_to_h5)
        self.actionLibrary_logistics.triggered.connect(self.load_library_management_subwindow)
        self.actionClassifier_components.triggered.connect(self.load_clf_subwindow)
        self.actionAdd_features_to_h5_folder.triggered.connect(self.load_add_prediction_features_subwindow)

        self.plot_1 = self.GraphicsLayoutWidget.addPlot()
        self.plot_overview = self.overview_plot.addPlot()
        #self.tid_box.setValue(6)
        #self.traceSelector.valueChanged.connect(self.plot_traces)
        #self.channel_selector.valueChanged.connect(self.plot_traces)
        self.treeWidget.itemSelectionChanged.connect(self.master_tree_selection)

        self.predictions_up = False
        self.library_up = False
        self.file_dir_up = False


        #self.debug_load_pred_files()

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
        '''
        self.print_widget_coords = False # use this to print out coords when clicking the plot stuff

    def not_done_yet(self):
        QtGui.QMessageBox.information(self,"Not implemented, lazy!", "Not implemented yet! Jonny has been lazy!")

    def load_clf_subwindow(self):
        child = subwindows.ClfWindow()
        child.show()
        child.home = self.home
        if child.exec_():
            print('exec_() was called')
        return 0

    def load_add_prediction_features_subwindow(self):
        child = subwindows.AddPredictionFeaturesWindow()
        child.show()
        child.home = self.home
        if child.exec_():
            print('exec_() was called')
        return 0

    def load_library_management_subwindow(self):
        child = subwindows.LibraryWindow()
        child.show()
        child.home = self.home
        if child.exec_():
            print('exec_was called')
        return 0

    def convert_ndf_folder_to_h5(self):
        child = subwindows.ConvertingNDFsWindow()
        child.show()
        child.home = self.home
        if child.exec_():
            print('exec_was called')
        return 0

    def set_home(self):
        self.home = QtGui.QFileDialog.getExistingDirectory(self, "Set a default folder to load from", self.home)

    def load_seizure_library(self):

        self.library = QtGui.QFileDialog.getOpenFileName(self, "Pick a h5 library file", self.home)[0]
        if self.library is '':
            print('nothing selected')
            return 0
        print(self.library)
        print(type(self.library))
        self.clear_QTreeWidget()
        # todo here check if got chunked thing from lirbary.. if not, throw messagebox error and return 0
        with h5py.File(self.library) as f:
            group_names = list(f.keys())
            groups = [f[key] for key in group_names]

            for i, group in enumerate(groups):
                for seizure_i in range(group.attrs['precise_annotation'].shape[0]):
                    row = {'name' : group_names[i],
                           'start': group.attrs['precise_annotation'][seizure_i, 0],
                           'end'  : group.attrs['precise_annotation'][seizure_i, 1],
                           'tid'  : group.attrs['tid'],
                           'index': i, # not sure if this is gonna work/ not consistent with the predictions
                           'chunk_start': group.attrs['chunked_annotation'][seizure_i, 0],
                           'chunk_end': group.attrs['chunked_annotation'][seizure_i, 1],
                           }
                    self.populate_tree_items_from_library(row)
            self.treeWidget.addTopLevelItems(self.tree_items)

        self.predictions_up = False
        self.library_up = True
        self.file_dir_up = False

    def populate_tree_items_from_library(self, row):

        self.treeWidget.setColumnCount(7)
        self.treeWidget.setHeaderLabels(['index','start','end','duration','chunk_start','chunk_end', 'tid','name'])
        details_entry = [str(row['index']),
                         str(row['start']),
                         str(row['end']),
                         str((row['end'] - row['start']) ),
                         str(row['chunk_start']),
                         str(row['chunk_end']),
                         str(row['tid']),
                         str(row['name'])]

        item = QtGui.QTreeWidgetItem(details_entry)
        item.setFirstColumnSpanned(True)

        self.tree_items.append(item)

    def master_tree_export_csv(self):
        if self.predictions_up:
            self.predictions_tree_export_csv()
        elif self.library_up:
            self.library_tree_export_csv()
        elif self.file_dir_up:
            self.file_tree_export_csv()
    def file_tree_export_csv(self):
        pass

    def library_tree_export_csv(self):
        if self.h5directory:
            default_dir = os.path.dirname(self.h5directory)
        else:
            default_dir = ""
        save_name = QtGui.QFileDialog.getSaveFileName(self,'Save library details in a .csv file',default_dir)[0]
        if save_name is '':
            print('nothing selected')
            return 0
        # now build dataframe from the tree
        root = self.treeWidget.invisibleRootItem()
        child_count = root.childCount()
        index, start, end, tid, fname, duration = [],[],[],[],[], []
        for i in range(child_count):
            item = root.child(i)
            index.append(item.text(0))
            start.append(item.text(1))
            end.append(item.text(2))
            tid.append(item.text(6))
            fname.append(item.text(7))
            duration.append(item.text(3))
        exported_df = pd.DataFrame(data = np.vstack([fname,start,end,duration,tid]).T,columns = ['filename','start','end','duration','transmitter'] )

        save_name = save_name.strip('.csv')
        exported_df.to_csv(save_name+'.csv')

    def predictions_tree_export_csv(self):
        if self.h5directory:
            default_dir = os.path.dirname(self.h5directory)
        else:
            default_dir = ""
        save_name = QtGui.QFileDialog.getSaveFileName(self,'Save annotation .csv file',default_dir)[0]
        if save_name is '':
            print('nothing selected')
            return 0
        # now build dataframe from the tree
        root = self.treeWidget.invisibleRootItem()
        child_count = root.childCount()
        index, start, end, tid, fname, duration = [],[],[],[],[], []
        for i in range(child_count):
            item = root.child(i)
            index.append(item.text(0))
            start.append(item.text(1))
            end.append(item.text(2))
            tid.append(item.text(4))
            fname.append(item.text(5))
            duration.append(item.text(3))
        exported_df = pd.DataFrame(data = np.vstack([index,fname,start,end,duration,tid]).T,columns = ['old_index','filename','start','end','duration','transmitter'] )

        save_name = save_name.strip('.csv')
        exported_df.to_csv(save_name+'.csv')

    def master_tree_selection(self):
        if not self.deleteing:                     # this is a hack as was being called as I was clearing the items
            if self.predictions_up:
                self.tree_selection_predictions()
            elif self.library_up:
                self.tree_selection_library()
            elif self.file_dir_up:
                self.tree_selection_file_dir

    def tree_selection_file_dir(self):
        pass

    def tree_selection_library(self):
        seizure_buffer = 5 # seconds either side of seizure to plot
        current_item = self.treeWidget.currentItem()
        fields = current_item

        index = int(float(fields.text(0)))
        start = float(fields.text(1))
        end = float(fields.text(2))
        duration = float(fields.text(3))
        chunk_start = float(fields.text(4))
        chunk_end = float(fields.text(5))
        tid = float(fields.text(6))
        key = fields.text(7)

        with h5py.File(self.library) as f:
            dataset = f[key]
            self.fs = dataset.attrs['fs']
            # todo, then assumes you have calculated labels need to be calculated second?
            # i guess you can just add labels before?

            self.plot_1.clear()
            self.bx1 = self.plot_1.getViewBox()

            data = dataset['data'][:]
            time = np.linspace(0, data.shape[0]/self.fs, data.shape[0])
            self.add_data_to_plots(data,time)

            start_pen = pg.mkPen((85, 168, 104), width=3, style= Qt.DashLine)
            end_pen = pg.mkPen((210,88,88), width=3, style= Qt.DashLine)
            coarse_pen = pg.mkPen((210,210,210), width=3, style= Qt.DashLine)

            self.start_line = pg.InfiniteLine(pos=start, pen =start_pen, movable=True,label='{value:0.2f}',
                           labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
            self.end_line = pg.InfiniteLine(pos=end, pen = end_pen, movable=True,label='{value:0.2f}',
                           labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})

            self.start_coarse = pg.InfiniteLine(pos=chunk_start, pen =coarse_pen, movable=False,label='{value:0.2f}',
                           labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
            self.end_coarse   = pg.InfiniteLine(pos=chunk_end, pen =coarse_pen, movable=False,label='{value:0.2f}',
                           labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})

            self.plot_1.addItem(self.start_line)
            self.plot_1.addItem(self.end_line)
            self.plot_1.addItem(self.start_coarse)
            self.plot_1.addItem(self.end_coarse)
            self.start_line.sigPositionChanged.connect(self.update_tree_element_start_time)
            self.end_line.sigPositionChanged.connect(self.update_tree_element_end_time)
            self.plot_1.setXRange(chunk_start-seizure_buffer, chunk_end+seizure_buffer)
            self.plot_1.setTitle(str(index)+' - '+ key+ '\n' + str(start)+' - ' +str(end))
            self.plot_overview.setTitle('Overview of file: '+str(index)+' - '+ key)
            self.updatePlot()

    def tree_selection_predictions(self):
        # this method does too much
        "grab tree detail and use to plot"
        seizure_buffer = 5 # seconds either side of seizure to plot
        current_item = self.treeWidget.currentItem()

        fields = current_item
        tid = int(float(fields.text(4)))
        start = float(fields.text(1))
        end = float(fields.text(2))
        index = float(fields.text(0))
        # duration is fields.text(3)
        fpath = os.path.join(self.h5directory, fields.text(5))

        h5 = H5File(fpath)
        data_dict = h5[tid]
        self.fs = eval(h5.attributes['fs_dict'])[tid]

        self.add_data_to_plots(data_dict['data'], data_dict['time'])
        start_pen = pg.mkPen((85, 168, 104), width=3, style= Qt.DashLine)
        end_pen = pg.mkPen((210,88,88), width=3, style= Qt.DashLine)
        self.start_line = pg.InfiniteLine(pos=start, pen =start_pen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
        self.end_line = pg.InfiniteLine(pos=end, pen = end_pen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
        # todo add all lines per file in one go - more than one seizure
        self.plot_1.addItem(self.start_line)
        self.plot_1.addItem(self.end_line)
        self.start_line.sigPositionChanged.connect(self.update_tree_element_start_time)
        self.end_line.sigPositionChanged.connect(self.update_tree_element_end_time)

        self.plot_1.setXRange(start-seizure_buffer, end+seizure_buffer)
        self.plot_1.setTitle(str(index)+' - '+ fpath+ '\n' + str(start)+' - ' +str(end))
        self.plot_overview.setTitle('Overview of file: '+str(index)+' - '+ fpath)
        self.updatePlot()

    def add_data_to_plots(self, data, time):
        self.plot_1.clear()
        self.bx1 = self.plot_1.getViewBox()
        hdf5_plot = HDF5Plot(parent = self.plot_1, viewbox = self.bx1)
        hdf5_plot.setHDF5(data, time, self.fs)
        self.plot_1.addItem(hdf5_plot)
        self.plot_1.setLabel('left', 'Voltage (uV)')
        self.plot_1.setLabel('bottom','Time (s)')

        # hit up the linked view here
        self.plot_overview.clear()
        self.plot_overview.enableAutoRange(False,False)
        self.plot_overview.setXRange(0,3600) # hardcoding in the hour here...
        self.plot_overview.setMouseEnabled(x = False, y= True)
        self.bx_overview = self.plot_overview.getViewBox()
        hdf5_plotoverview = HDF5Plot(parent = self.plot_overview, viewbox = self.bx_overview)
        hdf5_plotoverview.setHDF5(data, time, self.fs)
        self.plot_overview.addItem(hdf5_plotoverview)
        self.plot_overview.setLabel('left', 'Voltage (uV)')
        self.plot_overview.setLabel('bottom','Time (s)')

        self.lr = pg.LinearRegionItem(self.plot_1.getViewBox().viewRange()[0])
        self.lr.setZValue(-10)
        self.plot_overview.addItem(self.lr)
        # is this good practice?
        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.plot_1.sigXRangeChanged.connect(self.updateRegion)
        self.updatePlot()

    # these two methods are for the lr plot connection, refactor names
    def updatePlot(self):
        self.plot_1.setXRange(*self.lr.getRegion(), padding=0)
    def updateRegion(self):
        self.lr.setRegion(self.plot_1.getViewBox().viewRange()[0])

    def update_tree_element_start_time(self):
        tree_row = self.treeWidget.currentItem()
        tree_row.setText(1,'{:.2f}'.format(self.start_line.x()))
        self.update_tree_element_duration()

    def update_tree_element_end_time(self):
        tree_row = self.treeWidget.currentItem()
        tree_row.setText(2,'{:.2f}'.format(self.end_line.x()))
        self.update_tree_element_duration()

    def update_tree_element_duration(self):
        tree_row = self.treeWidget.currentItem()
        duration = float(tree_row.text(2))-float(tree_row.text(1))
        tree_row.setText(3, '{:.2f}'.format(duration))

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

    def clear_QTreeWidget(self):
        # not sure if i need this top bit
        self.deleteing = True
        if self.treeWidget.currentItem():
            current_item = self.treeWidget.currentItem()
            root = self.treeWidget.invisibleRootItem()
            root.removeChild(current_item)

        root = self.treeWidget.invisibleRootItem()
        n_kids = root.childCount()
        for i in np.arange(n_kids)[::-1]:
            child = self.treeWidget.topLevelItem(i)
            root.removeChild(child)

        self.tree_items = []
        self.deleteing = False

    def load_predictions_gui(self):
        ''' here make the window for choosing h5 file directory and predictions '''
        # we want something displaying the two files and lets you optionally change the h5 folder.
        print('loading new window!')
        child = subwindows.LoadingSubwindow()
        child.show()
        child.home = self.home # will this inherit? :p
        if child.exec_():
            print(child.predictions_fname)
            self.h5directory = child.h5directory
            self.predictions_fname = child.predictions_fname

            self.update_h5_folder_display()
            self.update_predictionfile_display()
            self.load_pred_file()

    def update_h5_folder_display(self):
        self.h5_folder_display.setText(str(self.h5directory))

    def update_predictionfile_display(self):
        self.predictions_file_display.setText(str(self.predictions_fname))

    def load_pred_file(self):
        self.clear_QTreeWidget()
        if self.predictions_fname.endswith('.csv'):
            self.predictions_df = pd.read_csv(self.predictions_fname)
        elif self.predictions_fname.endswith('.xlsx'):
            self.predictions_df = pd.read_excel(self.predictions_fname)
        else:
            print('Please select .csv or .xlsx file')
            return 0
        self.update_predictionfile_display()
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
            s,e = row['Start'], row['End']
            self.populate_tree_items_list_from_predictions(row, tids)    # this just populates self.tree_items
        self.treeWidget.addTopLevelItems(self.tree_items)

        self.predictions_up = True
        self.library_up = False
        self.file_dir_up = False

    def debug_load_pred_files(self): # stripped down version of the above for debugging gui code
        self.clear_QTreeWidget()
        try:
            self.h5directory = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan/2016_10/M6'
            self.update_h5_folder_display()
            self.predictions_fname = '/Volumes/G-DRIVE with Thunderbolt/GL_Steffan/2016_10/clf_predictions_m6_201610_no_pks.csv'
            self.update_predictionfile_display()
            self.predictions_df = pd.read_csv(self.predictions_fname)
            self.predictions_df['Index'] = self.predictions_df.index
            print('loading up files for debug')
            for i,row in list(self.predictions_df.iterrows()):
                fpath = os.path.join(self.h5directory,row['Filename'])
                tids = [int(fpath.split(']')[0].split('[')[1])]
                s,e = row['Start'], row['End']
                self.populate_tree_items_list_from_predictions(row, tids)
            self.treeWidget.addTopLevelItems(self.tree_items)

            self.predictions_up = True
            self.library_up = False
            self.file_dir_up = False
        except:
            print('Error, with debug loading')

    def populate_tree_items_list_from_predictions(self, row, tids):
        # todo refactor this name

        self.treeWidget.setColumnCount(5)
        self.treeWidget.setHeaderLabels(['index', 'start', 'end','duration', 'tid', 'fname'])
        filename = row['Filename']
        index =  row['Index']
        start =  row['Start']
        end = row['End']
        duration = row['End']-row['Start']

        fname_entry = [str(filename)]
        details_entry = [str(index),
                         str(start),
                         str(end),
                         str(duration),
                         str(tids[0]), # bad, should make only tid having one explicit
                         str(filename)]
        item = QtGui.QTreeWidgetItem(details_entry)
        item.setFirstColumnSpanned(True)

        self.tree_items.append(item)

        #self.treeWidget.addTopLevelItems(self.tree_items) # now beeing called once all items are there.

    #TODO implement this clicking stuff! click and move the region, these two methods below
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

    def load_h5_file(self,fname):

        self.loading_thread = LoadFileThread(fname)
        #self.connect(self.loading_thread, SIGNAL("finished()"), self.done)
        self.connect(self.loading_thread, pyqtSignal("catch_data(PyQt_PyObject)"), self.catch_data)
        self.loading_thread.start()

    def catch_data(self, h5obj):
        self.h5obj = h5obj
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
        self.emit(pyqtSignal('catch_data(PyQt_PyObject)'), self.h5obj)

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
        if pg.CONFIG_OPTIONS['background'] == 'w':
            self.pen = (0,0,0)
        else:
            self.pen = (255,255,255)


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


        self.setData(visible_x, visible_y, pen=self.pen) # update the plot
        #self.setPos(start, 0) # shift to match starting index ### Had comment out to stop it breaking... when limit is >0?!
        self.resetTransform()
        #self.scale(scale, 1)  # scale to match downsampling


def main():
    app = QtGui.QApplication(sys.argv)
    form = MainGui()
    form.show()
    app.exec_()

