# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'check_preds_design.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(812, 367)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.treeWidget = QtGui.QTreeWidget(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeWidget.sizePolicy().hasHeightForWidth())
        self.treeWidget.setSizePolicy(sizePolicy)
        self.treeWidget.setAutoExpandDelay(-1)
        self.treeWidget.setColumnCount(4)
        self.treeWidget.setObjectName(_fromUtf8("treeWidget"))
        self.treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.treeWidget.headerItem().setText(1, _fromUtf8("2"))
        self.treeWidget.headerItem().setText(2, _fromUtf8("3"))
        self.treeWidget.headerItem().setText(3, _fromUtf8("4"))
        self.GraphicsLayoutWidget = GraphicsLayoutWidget(self.splitter)
        self.GraphicsLayoutWidget.setEnabled(True)
        self.GraphicsLayoutWidget.setObjectName(_fromUtf8("GraphicsLayoutWidget"))
        self.horizontalLayout.addWidget(self.splitter)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.select_folder_btn = QtGui.QPushButton(self.centralwidget)
        self.select_folder_btn.setObjectName(_fromUtf8("select_folder_btn"))
        self.horizontalLayout_2.addWidget(self.select_folder_btn)
        self.load_preds_btn = QtGui.QPushButton(self.centralwidget)
        self.load_preds_btn.setObjectName(_fromUtf8("load_preds_btn"))
        self.horizontalLayout_2.addWidget(self.load_preds_btn)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.select_folder_btn.setText(_translate("MainWindow", "Select h5 folder", None))
        self.load_preds_btn.setText(_translate("MainWindow", "Load predictions", None))

from pyqtgraph import GraphicsLayoutWidget
