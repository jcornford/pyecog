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
try: # for speed checking
    from line_profiler import LineProfiler

    def lprofile():
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)

                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner
except:
    pass

if __name__ != '__main__':
    from . import check_preds_design, loading_subwindow, convert_ndf_window
    from . import subwindows
    from .context import ndf
else:
    import check_preds_design, loading_subwindow, convert_ndf_window
    import subwindows
    from context import ndf
from ndf.h5loader import H5File
from ndf.datahandler import DataHandler

def throw_error(error_text = None):
    msgBox = QtWidgets.QMessageBox()
    if error_text is None:
        msgBox.setText('Error caught! \n'+str(traceback.format_exc(1)))
    else:
        msgBox.setText('Error caught! \n'+str(error_text))
    msgBox.exec_()
    return 0

class TreeWidgetItem(QtGui.QTreeWidgetItem ):
    def __init__(self, parent=None):
        QtGui.QTreeWidgetItem.__init__(self, parent)

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()
        try:
            return float(self.text(column) ) > float( otherItem.text(column) )
        except ValueError:
            return self.text(column) > otherItem.text(column)

class MainGui(QtGui.QMainWindow, check_preds_design.Ui_MainWindow):
    def __init__(self, parent=None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        super(MainGui, self).__init__(parent)
        self.setupUi(self)
        self.handler = DataHandler()


        self.scroll_flag = -1
        self.deleted_tree_items = []
        #self.splitter.setSizes([50,20])
        #self.splitter_2.setSizes([50,20])
        #self.splitter_3.setSizes([50,20])
        #self.bottom_splitter.setSizes([200,300])
        #self.full_splitter.setSizes([300,200,150])
        if self.blink_box.isChecked():
            self.blink      = 1
        else:
            self.blink      = -1
        self.scroll_sign = 1
        self.timer = QTimer()
        self.timer.timeout.connect(self.simple_scroll)
        self.blink_box.stateChanged.connect(self.blink_box_change)

        self.scroll_speed_box.valueChanged.connect(self.scroll_speed_change)
        self.checkBox_scrolling.stateChanged.connect(self.scroll_checkbox_statechange)
        self.xrange_spinBox.valueChanged.connect(self.xrange_change)
        self.tid_spinBox.valueChanged.connect(self.tid_spinBox_change)
        self.checkbox_filter_toggle.stateChanged.connect(self.plot1_display_filter_toggled)
        self.hp_filter_freq.valueChanged.connect(self.hp_filter_settings_changed)

        self.fs = None # change !
        self.previously_displayed_tid = None
        self.data_obj = None
        self.predictions_df = None
        self.h5directory = None
        self.tree_items = []
        self.valid_h5_tids = None

        self.hdf5_plot = None
        self.valid_tids_to_indexes = None
        self.indexes_to_valid_tids = None
        self.tid_spinbox_just_changed = False
        self.annotation_change_tid = False

        if os.path.exists('pyecog_temp_file.pickle'):
            with open('pyecog_temp_file.pickle', "rb") as temp_file:
                self.home = p.load(temp_file)
        else:
            self.home = os.getcwd()


        # Hook up the file bar stuff here
        self.substates_timewindow_secs = 6
        self.actionLoad_Predictions.triggered.connect(self.load_predictions_gui)
        self.actionSave_annotations.triggered.connect(self.master_tree_export_csv)
        self.actionLoad_Library.triggered.connect(self.load_seizure_library)
        self.actionLoad_h5_folder.triggered.connect(self.load_h5_folder) # this is still to do in its entireity
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
        self.treeWidget.setSortingEnabled(True)
        self.treeWidget.itemSelectionChanged.connect(self.master_tree_selection)

        self.predictions_up = False
        self.library_up = False
        self.file_dir_up = False
        self.substates_up = False
        self.substate_child_selected = False

    def hp_filter_settings_changed(self):
        if self.hdf5_plot is not None:
            self.hdf5_plot.wipe_filtered_data()
            self.plot1_display_filter_toggled()

    def plot1_display_filter_toggled(self):
        # set filter settings on trace
        if self.hdf5_plot is not None:
            toggle, hp, lp = self.get_plot1_display_filter_settings_from_maingui()
            self.hdf5_plot.set_display_filter_settings(toggle , hp, lp)
            self.hdf5_plot.updateHDF5Plot()

    def get_plot1_display_filter_settings_from_maingui(self):
        ''' Returns the state, high pass and low pass values from main gui'''
        hp = self.hp_filter_freq.value()
        lp = self.lp_filter_freq.value()
        if hp <= 0:
            self.hp_filter_freq.setValue(1.0)
        toggle = self.checkbox_filter_toggle.isChecked()
        return toggle, hp, lp

    def not_done_yet(self):
        QtGui.QMessageBox.information(self," ", "Not implemented yet! Jonny has been lazy!")

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
        with open("pyecog_temp_file.pickle", "wb") as f:
            p.dump(self.home, f)

    def get_h5_folder_fnames(self):
        new_directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a h5 folder", self.home)
        if new_directory == '':
            print('No folder selected')
            return 0
        self.h5directory = new_directory
        self.clear_QTreeWidget()
        self.build_startswith_to_filename()
        fnames = [f for f in os.listdir(self.h5directory) if f.endswith('.h5') if not f.startswith('.') ]
        return fnames



    def load_h5_folder(self):
        fnames = self.get_h5_folder_fnames()
        if fnames == 0:
            return 0
        for i,fname in enumerate(fnames):
            try:
                tids = eval('['+fname.split(']')[0].split('[')[1]+']')
                self.populate_tree_items_list_from_h5_folder(i,fname, tids)    # this just populates self.tree_items
            except:

                print('Failed to add: '+ str(fname))

        self.treeWidget.addTopLevelItems(self.tree_items)
        self.update_h5_folder_display()
        self.predictions_up = False
        self.library_up = False
        self.file_dir_up = True
        self.substates_up = False

    def populate_tree_items_list_from_h5_folder(self,index,fpath,tids):
        self.deleted_tree_items = []
        self.treeWidget.setColumnCount(6)
        self.treeWidget.setHeaderLabels(['index', 'start', 'end','duration', 'tids', 'fname', 'real_start','real_end'])

        details_entry = [str(index),
                         '',
                         '',
                         '',
                         str(tids),
                         str(fpath),
                         '',
                         '']

        item = TreeWidgetItem(details_entry)
        item.setFirstColumnSpanned(True)

        self.tree_items.append(item)

    def load_seizure_library(self):
        try:
            self.library = QtGui.QFileDialog.getOpenFileName(self, "Pick a h5 library file", self.home)[0]
            if self.library is '':
                print('nothing selected')
                return 0
            print(self.library)
            print(type(self.library))
            self.clear_QTreeWidget()

            with h5py.File(self.library) as f:
                group_names = list(f.keys())
                groups = [f[key] for key in group_names]

                for i, group in enumerate(groups):
                    for seizure_i in range(group.attrs['precise_annotation'].shape[0]):
                        real_start = self.handler.get_time_from_seconds_and_filepath(group_names[i],
                                                                                     group.attrs['precise_annotation'][seizure_i, 0],
                                                                                     split_on_underscore=True).round('s')

                        real_end = self.handler.get_time_from_seconds_and_filepath(group_names[i],
                                                                                     group.attrs['precise_annotation'][seizure_i, 1],
                                                                                     split_on_underscore=True).round('s')
                        row = {'name' : group_names[i],
                               'start': group.attrs['precise_annotation'][seizure_i, 0],
                               'end'  : group.attrs['precise_annotation'][seizure_i, 1],
                               'tid'  : group.attrs['tid'],
                               'index': i,
                               'chunk_start': group.attrs['chunked_annotation'][seizure_i, 0],
                               'chunk_end': group.attrs['chunked_annotation'][seizure_i, 1],
                               'real_start':real_start,
                               'real_end':real_end
                               }
                        self.populate_tree_items_from_library(row)
                self.treeWidget.addTopLevelItems(self.tree_items)

            self.predictions_up = False
            self.library_up = True
            self.file_dir_up = False
            self.substates_up = True
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText('Error caught at load_seizure_library() \n'+str(traceback.format_exc(1)))
            msgBox.exec_()


    def populate_tree_items_from_library(self, row):
        self.deleted_tree_items = []
        self.treeWidget.setColumnCount(9)
        self.treeWidget.setHeaderLabels(['index','start','end','duration','chunk_start','chunk_end', 'tid','name', 'real_start', 'real_end'])

        details_entry = [str(row['index']),
                         str(row['start']),
                         str(row['end']),
                         str((row['end'] - row['start']) ),
                         str(row['chunk_start']),
                         str(row['chunk_end']),
                         str(row['tid']),
                         str(row['name']),
                         str(row['real_start']),
                         str(row['real_end'])
                         ]

        item = TreeWidgetItem(details_entry)
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
        # we just call predicitions tree export as should be same idea
        self.predictions_tree_export_csv()

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

        try:
            exported_df.to_csv(save_name+'.csv')
        except PermissionError:
            throw_error('Error - permission error! Is the file open somewhere else?')
            return 1

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
        index, start, end, tid, fname, duration,real_end,real_start = [],[],[],[],[], [],[],[]
        for i in range(child_count):
            item = root.child(i)
            index.append(item.text(0))
            start.append(item.text(1))
            end.append(item.text(2))
            try:
                tid_str = int(item.text(4))
            except:
                # tid is probably a '[x]'
                tid_str = eval(item.text(4))
                if hasattr(tid_str, '__iter__'):
                    tid_str = str(tid_str)
            tid.append(tid_str)
            fname.append(item.text(5))
            duration.append(item.text(3))
            real_end.append(item.text(6))
            real_start.append(item.text(7))
        exported_df = pd.DataFrame(data = np.vstack([index,fname,start,end,duration,tid, real_end,real_start]).T,columns = ['old_index','filename','start','end',
                                                                                                       'duration','transmitter', 'real_start', 'real_end'] )

        save_name = save_name.strip('.csv')
        try:
            exported_df.to_csv(save_name+'.csv')
        except PermissionError:
            throw_error('Error - permission error! Is the file open somewhere else?')
            return 1

    def master_tree_selection(self):
        if not self.deleteing:                     # this is a hack as was being called as I was clearing the items
            if self.predictions_up:
                #todo Jonny hacking awway again, this actuall loops back to tree_selections_preductions
                self.tree_selection_file_dir()
                #self.tree_selection_predictions()
            elif self.library_up:
                self.tree_selection_library()
            elif self.file_dir_up:
                self.tree_selection_file_dir()
            elif self.substates_up:
                self.tree_selection_substates()

    def get_next_tree_item(self):
        print('not implememented: try to grab next treewidget item!')

    def set_valid_h5_ids(self, tid_list):
        self.valid_h5_tids = tid_list
        self.valid_tids_to_indexes
        self.valid_tids_to_indexes = {tid:i for i, tid in enumerate(self.valid_h5_tids)}
        self.indexes_to_valid_tids = {i:tid for i, tid in enumerate(self.valid_h5_tids)}
        self.previously_displayed_tid = None # you also want to "wipe the list?"

    def tree_selection_substates(self):
        item = self.treeWidget.currentItem()
        if item.text(2).startswith('M'):
            #print('Filerow')
            self.treeWidget.substate_child_selected = False
            tids = item.text(1)
            self.set_valid_h5_ids(eval(tids))
            self.handle_tid_for_file_dir_plotting() # this will automatically call the plotting by changing the v
        else:
            #print('child')
            self.treeWidget.substate_child_selected = True
            i = item.text(0)
            chunk = item.text(1)
            t_window = chunk.split('-')
            xmin = int(t_window[0])
            xmax = int(t_window[1])
            self.plot_1.getViewBox().setXRange(min = xmin,
                                               max = xmax, padding=0)
            category = item.text(2)

    def tree_selection_file_dir(self):
        # this method does too much
        "grab tree detail and use to plot"
        self.h5directory = self.h5directory # shitty but you had diff variabels?!
        current_item = self.treeWidget.currentItem()
        if current_item.text(1) != '':
            try:
                self.tree_selection_predictions()
            except:
                if current_item.text(2) == '':
                    msgBox = QtWidgets.QMessageBox()
                    msgBox.setText('Make an end and start line at the same time for best functionality')
                    msgBox.exec_()

                # do something else here
                self.tree_selection_predictions()
                #tids = current_item.text(4)
                #self.set_valid_h5_ids(eval(tids))
                #self.handle_tid_for_file_dir_plotting() # this will automatically call the plotting by changing the v
        else:
            tids = current_item.text(4)
            self.set_valid_h5_ids(eval(tids))
            self.handle_tid_for_file_dir_plotting() # this will automatically call the plotting by changing the v


    def load_filedir_h5_file(self,tid):
        '''
        Splitting tree_selection_file_dir up so changing the tid spinBox can reload easily
        '''
        import time
        start = time.time()
        fields = self.treeWidget.currentItem()
        path = fields.text(5)
        index = float(fields.text(0))
        fpath = os.path.join(self.h5directory, path)
        h5 = H5File(fpath)
        data_dict = h5[tid]
        self.fs = eval(h5.attributes['fs_dict'])[tid]
        self.add_data_to_plots(data_dict['data'], data_dict['time'])
        if self.checkbox_hold_trace_position.isChecked():
            xlims  = self.plot_1.getViewBox().viewRange()[0]
            x_min = xlims[0]
            x_max = xlims[1]
        else:
            xrange = self.xrange_spinBox.value()
            x_min = 0
            x_max = xrange
        self.plot_1.setXRange(x_min, x_max, padding=0)
        self.plot_1.setTitle(str(index)+' - '+ fpath+ '\n')
        self.plot_overview.setTitle('Overview of file: '+str(index)+' - '+ fpath)
        self.updatePlot()

    def handle_tid_for_file_dir_plotting(self):
        # this is called when clicking on the tree structure
        # therefore first check if can use the same tid or not...
        current_spinbox_id = self.tid_spinBox.value()
        if current_spinbox_id not in self.valid_h5_tids:
            self.tid_spinBox.setValue(self.valid_h5_tids[0]) # no reason to default to first
            # here you add something to hold id if needed
            #print('File tid changed as previous tid not valid')
            # this will now automatically call the tid_spinBox_change method - as you have tid changed it
        else:
            # can use the same tid so plot
            self.load_filedir_h5_file(current_spinbox_id)

        # as id number will now be the last value next time changed

    def tid_spinBox_change(self):
        ''' called when box data changes '''
        if self.tid_spinBox.value() == self.previously_displayed_tid:
            # catching the loop which occurs if setting the spinbox after finding next tid
            return 0
        elif self.annotation_change_tid == True:
            self.annotation_change_tid = False
            return 0
        else:
            self.tid_spinBox_handling()

    def set_tid_spinbox(self, value):
        '''
        This wil
        '''
        #print('Tid_spinbox has been set by code' )
        self.tid_spinBox.setValue(value)
        self.tid_spinbox_just_changed = True

    def tid_spinBox_handling(self):
        #print('tid spin box handling called')
        try:
            # tid_spinbox.valueChanged connects to here
            new_val = self.tid_spinBox.value()
            #print(time.time(), 'New spinbox value is ', new_val)
            set_tid_box = True
            if new_val in self.valid_h5_tids:
                set_tid_box = False # dont need to overwrite box
                new_tid = new_val
            elif new_val < min(self.valid_h5_tids): # this is rolling 0
                new_tid = self.valid_h5_tids[-1]

            elif new_val > max(self.valid_h5_tids): # this is rolling 0
                new_tid = self.valid_h5_tids[0]

            else:
                if self.previously_displayed_tid is not None:
                    step = new_val - self.previously_displayed_tid
                    old_index = self.valid_tids_to_indexes[self.previously_displayed_tid]
                    new_index = old_index+step
                    new_tid   = self.indexes_to_valid_tids[new_index]
                else:
                    i = bisect.bisect_left(self.valid_h5_tids,new_val)
                    new_tid = self.valid_h5_tids[i%len(self.valid_h5_tids)]

            #print('New is:', new_tid)
            #print('Last was:', self.previously_displayed_tid)
            self.previously_displayed_tid = new_tid
            self.load_filedir_h5_file(new_tid)
            if set_tid_box:
                self.set_tid_spinbox(new_tid)

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print (traceback.print_exception(exc_type, exc_value, exc_traceback))
            print('Error caught at: pyecog_main_gui.tid_spinBox_handling()')
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText('Error caught at tid_spinBox_handling() \n'+str(traceback.format_exc()))
            msgBox.exec_()



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
            self.plot_1.setXRange(chunk_start-seizure_buffer, chunk_end+seizure_buffer, padding=0)
            self.plot_1.setTitle(str(index)+' - '+ key+ '\n' + str(start)+' - ' +str(end))
            self.plot_overview.setTitle('Overview of file: '+str(index)+' - '+ key)
            self.updatePlot()

        self.annotation_change_tid = True
        self.set_tid_spinbox(tid)


    def build_startswith_to_filename(self):
        ''' split either on the bracket of the .'''
        self.startname_to_full = {}

        for f in os.listdir(self.h5directory):
            self.startname_to_full[f[:11]] = f

    def tree_selection_predictions(self):
        # this method does too much
        "grab tree detail and use to plot"
        seizure_buffer = 5 # seconds either side of seizure to plot
        current_item = self.treeWidget.currentItem()

        fields = current_item
        try:
            tid = int(fields.text(4))
        except:
            # tid is probably a '[x]'
            tid = eval(fields.text(4))
            if hasattr(tid, '__iter__'):
                tid = tid[0]


        start = float(fields.text(1))
        try:
            end = float(fields.text(2))
        except:
            end = start + 1
            print(' caught you not clicking an end, line 651, need to code this better')
        index = float(fields.text(0))
        # duration is fields.text(3)

        try:
            correct_file = self.startname_to_full[fields.text(5)[:11]]
        except KeyError:
                throw_error()
                return 0
        fpath = os.path.join(self.h5directory, correct_file)

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

        self.plot_1.setXRange(start-seizure_buffer, end+seizure_buffer, padding=0)
        self.plot_1.setTitle(str(index)+' - '+ fpath+ '\n' + str(start)+' - ' +str(end))
        self.plot_overview.setTitle('Overview of file: '+str(index)+' - '+ fpath)
        self.updatePlot()

        # you need to change the spinbox - should be caught if already the same?
        self.annotation_change_tid = True
        self.set_tid_spinbox(tid)

    #@lprofile()
    def add_data_to_plots(self, data, time):
        self.plot_1.clear()
        self.bx1 = self.plot_1.getViewBox()
        self.hdf5_plot = HDF5Plot(parent = self.plot_1, viewbox = self.bx1)
        if self.checkbox_filter_toggle.isChecked():
            toggle, hp, lp = self.get_plot1_display_filter_settings_from_maingui()
            self.hdf5_plot.set_display_filter_settings(toggle,hp,lp)
        self.hdf5_plot.setHDF5(data, time, self.fs)
        self.plot_1.addItem(self.hdf5_plot)
        self.plot_1.setLabel('left', 'Voltage (uV)')
        self.plot_1.setLabel('bottom','Time (s)')

        # hit up the linked view here
        self.plot_overview.clear()
        self.plot_overview.enableAutoRange(False,False)
        self.plot_overview.setXRange(0,3600, padding=0) # hardcoding in the hour here...
        self.plot_overview.setMouseEnabled(x = False, y= True)
        self.bx_overview = self.plot_overview.getViewBox()
        hdf5_plotoverview = HDF5Plot(parent = self.plot_overview, viewbox = self.bx_overview)
        hdf5_plotoverview.setHDF5(data, time, self.fs)
        self.plot_overview.addItem(hdf5_plotoverview)
        self.plot_overview.setXRange(time[0],time[-1], padding=0)
        self.plot_overview.setLabel('left', 'Voltage (uV)')
        self.plot_overview.setLabel('bottom','Time (s)')
        # mousePressEvent,mouseDoubleClickEvent ,sigMouseClicked,sigMouseMoved,wheelEvent
        # should you just be overwriting class methods for this stuff?
        self.proxy2 = pg.SignalProxy(self.plot_1.scene().sigMouseClicked,rateLimit=30,slot=self.mouse_click_on_main)
        self.proxy = pg.SignalProxy(self.plot_overview.scene().sigMouseClicked,rateLimit=30,slot=self.mouse_click_in_overview)
        #print(dir(self.plot_overview.scene()))

        self.lr = pg.LinearRegionItem(self.plot_1.getViewBox().viewRange()[0])
        self.lr.setZValue(-10)
        self.plot_overview.addItem(self.lr)
        # is this good practice?
        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.plot_1.sigXRangeChanged.connect(self.updateRegion) # xlims?
        self.plot_1.sigXRangeChanged.connect(self.xrange_changed_on_plot)
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
        self.check_for_blank()
        tree_row = self.treeWidget.currentItem()
        tree_row.setText(2,'{:.2f}'.format(self.end_line.x()))
        self.update_tree_element_duration()

    def check_for_blank(self):
        try:
            if int(self.end_line.x()) ==0 and int(self.start_line.x()) ==0:
                if self.start_line.x() != 0:
                    self.start_line.setValue(0)
                if self.end_line.x() != 0:
                    self.end_line.setValue(0)
                    print('Blank entered, setting to 0')

        except:
            print('Error when checking for blank')
            traceback.print_exception(1)


    def update_tree_element_duration(self):
        try:
            tree_row = self.treeWidget.currentItem()
            duration = float(tree_row.text(2))-float(tree_row.text(1))
            tree_row.setText(3, '{:.2f}'.format(duration))
            self.update_real_times()
        except:
            print('caught error at 777')

    def plot_traces(self, data_dict):

        if not self.holdPlot.isChecked():
            self.plot_1.clear()
        # here you need to add the h5 file class with downsampling

        curve1 = HDF5Plot()#parent = self.plot_1, viewbox = bx1)
        curve1.setHDF5(data_dict['data'], data_dict['time'], self.fs)
        self.plot_1.addItem(hdf5_plot)

        #self.plot_1.addItem(pg.PlotCurveItem(data_dict['time'], data_dict['data']))
        self.plot_1.setXRange(row['Start'], row['End'], padding=0)
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
        child.home = self.home # this doesnt overwrite when called when on those windows...
        if child.exec_():
            print(child.predictions_fname)
            self.h5directory = child.h5directory
            self.predictions_fname = child.predictions_fname

            self.update_h5_folder_display()
            self.update_predictionfile_display()
            self.load_pred_file()
            self.build_startswith_to_filename()

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
        self.predictions_df.columns = [colname.lower() for colname in self.predictions_df.columns]
        self.predictions_df.fillna(value = '', inplace=True)
        if self.h5directory is None:
            self.set_h5_folder()
        #print(self.predictions_df)
        for i,row in list(self.predictions_df.iterrows()):
            #todo correct this

            fpath = os.path.join(self.h5directory, row['filename'])
            #TODO : decide what to do with tids, this is not thought out at the moment
            #So not bothering to load the tids here as should be one only per seizure... can either  load on demand
            # or use only for the data explorer stuff. Can maybe have dynamic when you click to see the full tree.
            # problem is with files with many false positives, spend time loading for too long!
            # or have a button for this...

            #h5 = H5File(fpath)
            #tids = h5.attributes['t_ids']
            try:
                tids = row['transmitter']
            except:
                # this is legacy from when there was only one
                print('WARNING: DO NOT RELY ON ONE TID PER FILE - TELL JONNY')
                tids = [int(fpath.split(']')[0].split('[')[1])]

            s, e = row['start'], row['end']

            self.populate_tree_items_list_from_predictions(row, tids)    # this just populates self.tree_items
        self.treeWidget.addTopLevelItems(self.tree_items)

        self.predictions_up = True
        self.library_up = False
        self.file_dir_up = False
        self.substates_up = False

    def populate_tree_items_list_from_predictions(self, row, tids):
        # todo refactor this name to annoations etc
        self.deleted_tree_items = []
        self.treeWidget.setColumnCount(7)
        self.treeWidget.setHeaderLabels(['index', 'start', 'end','duration', 'tid', 'fname', 'real_start', 'real_end'])
        filename = row['filename']
        index =  row['index']
        start =  row['start']
        end = row['end']
        if row['start'] !='' and row['end'] !='':
            try: # if made with, then will have both
                real_start = row['real_start']
                real_end   = row['real_end']
            except:
                real_start = self.handler.get_time_from_seconds_and_filepath(filename,
                                                                             start,
                                                                             split_on_underscore=True).round('s')

                real_end   = self.handler.get_time_from_seconds_and_filepath(filename,
                                                                             end,
                                                                             split_on_underscore=True).round('s')
        else:
            real_start, real_end = '',''
        try:
            duration = row['end']-row['start']
        except:
            duration = ''

        fname_entry = [str(filename)]
        details_entry = [str(index),
                         str(start),
                         str(end),
                         str(duration),
                         str(tids), # bad, should make only tid having one explicit - predictions should only have one!
                         str(filename),
                         str(real_start),
                         str(real_end)]
        item = TreeWidgetItem(details_entry)
        item.setFirstColumnSpanned(True)

        self.tree_items.append(item)

    def make_new_tree_entry_from_current(self,item,xpos):
        ''' for adding seizures with start '''
        current_tid = [self.tid_spinBox.value()]
        details_entry = [str(item.text(0)),
                         str(''),
                         str(''),
                         str(''),
                         str(current_tid), # bad, should make only tid having one explicit
                         str(item.text(5)),
                         str(''),
                         str('')]

        new_item = TreeWidgetItem(details_entry)
        return new_item

    def add_new_entry_to_tree_widget(self,xpos):
        ''' this is for when file dir is up'''

        current_item = self.treeWidget.currentItem()
        root =  self.treeWidget.invisibleRootItem()
        index= root.indexOfChild(current_item)
        test_item = self.make_new_tree_entry_from_current(current_item,xpos)
        self.tree_items.insert(index+1, test_item)
        self.treeWidget.insertTopLevelItem(index+1, test_item)
        xlims  = self.plot_1.getViewBox().viewRange()[0]
        self.treeWidget.setCurrentItem(test_item)
        self.plot_1.getViewBox().setXRange(min = xlims[0],max = xlims[1], padding=0)




    def add_start_line_to_h5_file(self,xpos):

        self.add_new_entry_to_tree_widget(xpos) # this makes new slects it, and sets xrange to the same.

        # now make lines and wipe end line
        start_pen = pg.mkPen((85, 168, 104), width=3, style= Qt.DashLine)
        self.start_line = pg.InfiniteLine(pos=xpos, pen =start_pen, movable=True,label='{value:0.2f}',
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
        self.plot_1.addItem(self.start_line)

        self.treeWidget.currentItem().setText(1,'{:.2f}'.format(self.start_line.x()))
        self.start_line.sigPositionChanged.connect(self.update_tree_element_start_time)
        self.end_line = None

    def set_end_and_calc_duration(self):
        current_item = self.treeWidget.currentItem()
        current_item.setText(2,'{:.2f}'.format(self.end_line.x()))
        self.update_tree_element_duration()


    def update_real_times(self):
        # this is called by the update_tree_element_duration
        try:
            tree_row = self.treeWidget.currentItem()
            fname = tree_row.text(5)
            real_start = self.handler.get_time_from_seconds_and_filepath(fname,float(tree_row.text(1)), split_on_underscore = True).round('s')
            real_end   =  self.handler.get_time_from_seconds_and_filepath(fname,float(tree_row.text(2)), split_on_underscore = True ).round('s')
            tree_row.setText(6, str(real_start))
            tree_row.setText(7, str(real_end))
        except:
            throw_error()

            print('caught error at 777')


    def add_end_line_to_h5(self, xpos):
        # do something to move existing end line self.check_end_line_exists()
        if self.end_line is None:
            end_pen = pg.mkPen((210,88,88), width=3, style= Qt.DashLine)
            self.end_line = pg.InfiniteLine(pos=xpos, pen = end_pen, movable=True,label='{value:0.2f}',
                           labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,0), 'movable': True})
            self.plot_1.addItem(self.end_line)
            self.end_line.sigPositionChanged.connect(self.update_tree_element_end_time)
        else:
            self.end_line.setValue(xpos)
        self.set_end_and_calc_duration()

    def mouse_click_on_main(self,evt):
        pos = evt[0].scenePos()
        if self.plot_1.sceneBoundingRect().contains(pos):
            mousePoint = self.bx1.mapSceneToView(pos) # bx1 is just self.plot_1.getViewBox()

        modifier = evt[0].modifiers()

        if modifier == Qt.ShiftModifier:
            if self.library_up:
                throw_error('Unfortunately unable to add to library at the moment. You have to edit the annotations csv that was used to make the library, sorry.' )
                return 0
            self.add_start_line_to_h5_file(mousePoint.x())

        elif modifier == Qt.AltModifier:
            if self.library_up:
                throw_error('Unfortunately unable to add to library at the moment. You have to edit the annotations csv that was used to make the library, sorry.' )
                return 0
            self.add_end_line_to_h5(mousePoint.x())

    def mouse_click_in_overview(self,evt):
        # signal for this is coming from self.data,
        # evt[0] should be a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent

        pos = evt[0].scenePos()
        if self.plot_overview.sceneBoundingRect().contains(pos):
            mousePoint = self.bx_overview.mapSceneToView(pos)
            x = int(mousePoint.x())
            y = int(mousePoint.y())

            xrange, _ = self.get_main_plot_xrange_and_mid()
            try:
                assert xrange > 0
            except:
                print('Your view range is messed up')

            self.plot_1.getViewBox().setXRange(min = x - xrange/2.0,
                                               max = x + xrange/2.0, padding=0)

    def get_main_plot_xrange_and_mid(self):
        xlims  = self.plot_1.getViewBox().viewRange()[0]
        xrange = xlims[1]-xlims[0]
        xmid   = xlims[0]+xrange/2.0
        return xrange, xmid

    def xrange_change(self):
        #self.xrange_spinBox.valueChanged.connect(self.xrange_change)
        xrange = self.xrange_spinBox.value()
        if xrange>0:
            if self.plot_change == False:
                _, xmid = self.get_main_plot_xrange_and_mid()
                self.plot_1.getViewBox().setXRange(min = xmid - xrange/2.0,
                                                   max = xmid + xrange/2.0, padding=0)
            elif self.plot_change == True:
                # changing because plot has already changed - not key or spinbox alteration
                self.plot_change = False
        else:
            pass

    def xrange_changed_on_plot(self):
        xrange, xmid = self.get_main_plot_xrange_and_mid()
        self.plot_change = True
        self.xrange_spinBox.setValue(xrange)


    def undo_tree_deletion(self):

        if len(self.deleted_tree_items) == 0:
            print('Nothing to undo')
            return 0
        old_item = self.deleted_tree_items[-1]
        del self.deleted_tree_items[-1]
        original_index = int(old_item.text(0))
        tree_index = self.get_new_index_for_deleted_tree_element(original_index)

        self.treeWidget.insertTopLevelItem(tree_index, old_item)
        self.treeWidget.setCurrentItem(old_item)

    def get_new_index_for_deleted_tree_element(self, deleted_element_index):

        root = self.treeWidget.invisibleRootItem()
        child_count = root.childCount()
        index_list = []
        for i in range(child_count):
            item = root.child(i)
            index_list.append(int(item.text(0)))
        correct_tree_index = bisect.bisect_left(index_list, int(deleted_element_index))
        return correct_tree_index

    def keyPressEvent(self, eventQKeyEvent):

        key_id = eventQKeyEvent.key()
        modifier = eventQKeyEvent.modifiers()

        key_id_to_numbers = {eval('Qt.Key_'+str(i)):i for i in range(1,10)}
        if key_id in list(key_id_to_numbers.keys()):
            self.plot_change = False # disable this as key now entered
            key_val = key_id_to_numbers[key_id]
            if key_val == self.xrange_spinBox.value():
                self.xrange_change() # just call again
            else:
                self.xrange_spinBox.setValue(key_val)
                # connected trigger will call xrange change

        x,y = self.plot_1.getViewBox().viewRange()

        if key_id == Qt.Key_Delete or key_id == Qt.Key_Backspace:
            # store the deleted element so you can undo it
            tree_entry = self.treeWidget.currentItem()
            self.deleted_tree_items.append(tree_entry)

        if key_id ==  Qt.Key_Z and modifier == Qt.ControlModifier:
            self.undo_tree_deletion()

        if key_id ==  Qt.Key_Z :
            self.undo_tree_deletion()

        if key_id == Qt.Key_Up:
            if self.scroll_flag==True:
                scroll_rate = self.scroll_speed_box.value()
                new_rate = scroll_rate * 2
                self.scroll_speed_box.setValue(new_rate)
                if self.blink ==True: self.reset_timer()
            else:
                self.plot_1.getViewBox().setYRange(min = y[0]*0.9, max = y[1]*0.9, padding = 0)

        if key_id == Qt.Key_Down:
            if self.scroll_flag==True:
                scroll_rate = self.scroll_speed_box.value()
                if scroll_rate > 1:
                    new_rate = int(scroll_rate / 2)
                    self.scroll_speed_box.setValue(new_rate)
                    if self.blink ==True: self.reset_timer()
            else: # just zoom
                self.plot_1.getViewBox().setYRange(min = y[0]*1.1, max = y[1]*1.1,padding = 0)

        if key_id == Qt.Key_Right:
            if self.scroll_flag==True:
                self.scroll_sign = 1
            else:
                #scroll_i = (x[1]-x[0])*0.01*self.scroll_speed_box.value()
                #if scroll_i > x[1]-x[0]: scroll_i = x[1]-x[0]
                #self.plot_1.getViewBox().setXRange(min = x[0]+scroll_i, max = x[1]+scroll_i, padding=0)

                scroll_i = (x[1]-x[0])*1
                new_min =  x[0]+scroll_i
                new_max =  x[1]+scroll_i
                self.plot_1.getViewBox().setXRange(min =new_min, max = new_max, padding=0)

        if key_id == Qt.Key_Left:
            if self.scroll_flag==True:
                self.scroll_sign = -1
            else:
                #scroll_i = (x[1]-x[0])*0.01*self.scroll_speed_box.value()
                #if scroll_i > x[1]-x[0]: scroll_i = x[1]-x[0]
                #self.plot_1.getViewBox().setXRange(min = x[0]-scroll_i, max = x[1]-scroll_i, padding=0)

                scroll_i = (x[1]-x[0])*-1
                new_min =  x[0]+scroll_i
                new_max =  x[1]+scroll_i
                #if new_max < xmax:
                #self.get_next_tree_item()
                self.plot_1.getViewBox().setXRange(min =new_min, max = new_max, padding=0)

        if key_id == Qt.Key_Backspace or key_id == Qt.Key_Delete:
            current_item = self.treeWidget.currentItem()
            root = self.treeWidget.invisibleRootItem()
            root.removeChild(current_item)

        if key_id == Qt.Key_B:
            if self.scroll_flag==True:
                self.blink *= -1
                if self.blink == True:
                    self.blink_box.setChecked(True)
                else:
                    self.blink_box.setChecked(False)
                self.reset_timer()

        if key_id == Qt.Key_Space:
            self.scroll_sign = 1
            self.checkBox_scrolling.setChecked([1,0][self.checkBox_scrolling.isChecked()])
            #self.scroll_flag *= -1

    def scroll_checkbox_statechange(self):
        self.scroll_sign = 1
        self.scroll_flag = [-1,1][self.checkBox_scrolling.isChecked()]
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
        xlims = self.plot_overview.getViewBox().viewRange()[0]
        xmax = xlims[1]
        if self.blink_box.isChecked() != True:
            scroll_i = (x[1]-x[0])*(0.001*scroll_rate)*self.scroll_sign
            new_min =  x[0]+scroll_i
            new_max =  x[1]+scroll_i
            if new_max < xmax-1:
                #self.get_next_tree_item()
                self.plot_1.getViewBox().setXRange(min =new_min, max = new_max, padding=0)

        elif self.blink_box.isChecked():
            scroll_i = (x[1]-x[0])*self.scroll_sign
            new_min =  x[0]+scroll_i
            new_max =  x[1]+scroll_i
            if new_max < xmax:
                #self.get_next_tree_item()
                self.plot_1.getViewBox().setXRange(min =new_min, max = new_max, padding=0)

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

    def set_display_filter_settings(self, display_filter, hp_cutoff, lp_cutoff):
        self.display_filter = display_filter
        self.hp_cutoff = hp_cutoff
        self.lp_cutoff = lp_cutoff


    def highpass_filter(self, data):
        '''
        Implements high pass digital butterworth filter, order 2.

        Args:
            cutoff_hz: default is 1hz
        '''

        nyq = 0.5 * self.fs
        cutoff_decimal = self.hp_cutoff/nyq
        b, a = signal.butter(2, cutoff_decimal, 'highpass', analog=False)

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
                self.hdf5_filtered_data = hdf5data = self.highpass_filter(self.hdf5)
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


def main():
    app = QtGui.QApplication(sys.argv)
    maingui = MainGui()

    # cannot get menu bar to show on first launch - need to click of and back
    #maingui.menuBar.raise_()
    #maingui.menuBar.show()
    #maingui.menuBar.activateWindow()
    #maingui.menuBar.focusWidget(True)
    maingui.menuBar.setNativeMenuBar(False) # therefore turn of native

    maingui.raise_()
    maingui.show()

    app.exec_()

if __name__ == '__main__':
    main()
