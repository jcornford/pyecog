from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import Qt
import os
import pandas as pd
import numpy as np

class PecogQTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self,*args, **kwargs):
        super(PecogQTreeWidget, self).__init__(*args, **kwargs)
        self.substate_child_selected = False
        self.h5folder = None
        self.counter = 0

    def pyecog_save(self, save_mmsgbox = True):

        # only save if on a child - then we can grab parent for csv name
        #
        savepath = os.path.split(self.h5folder)[0]

        parent_node = self.currentItem().parent()
        savename = 'substates_'+parent_node.text(2).strip('.h5')
        n_children = parent_node.childCount()
        index, chunk, state = [],[],[]
        for i in range(n_children):
            child = parent_node.child(i)
            index.append(child.text(0))
            chunk.append(child.text(1))
            state.append(child.text(2))
        exported_df = pd.DataFrame(data = np.vstack([index,chunk,state,]).T,columns = ['index','chunk','state'] )
        fullsavename = os.path.join(savepath, savename)+'.csv'
        exported_df.to_csv(fullsavename,index=False)

        print('Saved: '+fullsavename) # should be 3600

        if save_mmsgbox:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText('Saved annotations for this hour: \n'+fullsavename)
            msgBox.exec_()

    def keyPressEvent(self, QKeyEvent):
        key_id = QKeyEvent.key()

        key_id_to_numbers = {eval('Qt.Key_'+str(i)):i for i in range(0,10)}
        if self.substate_child_selected:

            if key_id in list(key_id_to_numbers.keys()):

                key_val = key_id_to_numbers[key_id]
                self.currentItem().setText(2, str(key_val))
                # make down press command
                fake_down_press =QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Down, QtCore.Qt.NoModifier)
                super(PecogQTreeWidget,self).keyPressEvent(fake_down_press)
                self.counter +=1
                if self.counter >=10:
                    self.pyecog_save(save_mmsgbox=False)
                    self.counter =0


            elif key_id == Qt.Key_S:
                self.pyecog_save()

            else:
                super(PecogQTreeWidget,self).keyPressEvent(QKeyEvent)

        else:
            super(PecogQTreeWidget,self).keyPressEvent(QKeyEvent)