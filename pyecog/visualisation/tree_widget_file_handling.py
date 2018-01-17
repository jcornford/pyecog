

class PyecogTreeClass():

    def __init__(self):
        pass

    def build_startswith_to_filename(self):
        ''' split either on the bracket of the .'''
        self.startname_to_full = {}

        for f in os.listdir(self.h5directory):
            self.startname_to_full[f[:11]] = f

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

    def set_valid_h5_ids(self, tid_list):
        self.valid_h5_tids = tid_list
        self.valid_tids_to_indexes
        self.valid_tids_to_indexes = {tid:i for i, tid in enumerate(self.valid_h5_tids)}
        self.indexes_to_valid_tids = {i:tid for i, tid in enumerate(self.valid_h5_tids)}
        self.previously_displayed_tid = None # you also want to "wipe the list?"

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


