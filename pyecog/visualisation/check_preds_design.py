# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'check_preds_design_v2.ui'
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
        MainWindow.resize(936, 624)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.splitter_3 = QtGui.QSplitter(self.centralwidget)
        self.splitter_3.setOrientation(QtCore.Qt.Vertical)
        self.splitter_3.setObjectName(_fromUtf8("splitter_3"))
        self.splitter = QtGui.QSplitter(self.splitter_3)
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
        #self.treeWidget.headerItem().setText(0, _fromUtf8("1"))
        #self.treeWidget.headerItem().setText(1, _fromUtf8("2"))
        #self.treeWidget.headerItem().setText(2, _fromUtf8("3"))
        #self.treeWidget.headerItem().setText(3, _fromUtf8("4"))
        self.GraphicsLayoutWidget = GraphicsLayoutWidget(self.splitter)
        self.GraphicsLayoutWidget.setEnabled(True)
        self.GraphicsLayoutWidget.setObjectName(_fromUtf8("GraphicsLayoutWidget"))
        self.overview_plot = GraphicsLayoutWidget(self.splitter_3)
        self.overview_plot.setMinimumSize(QtCore.QSize(912, 100))
        self.overview_plot.setBaseSize(QtCore.QSize(912, 100))
        self.overview_plot.setObjectName(_fromUtf8("overview_plot"))
        self.splitter_2 = QtGui.QSplitter(self.splitter_3)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.widget = QtGui.QWidget(self.splitter_2)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout_3 = QtGui.QGridLayout(self.widget)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.h5_folder_display = QtGui.QLineEdit(self.widget)
        self.h5_folder_display.setMaximumSize(QtCore.QSize(150, 100))
        self.h5_folder_display.setObjectName(_fromUtf8("h5_folder_display"))
        self.gridLayout_3.addWidget(self.h5_folder_display, 1, 1, 1, 1)
        self.load_library = QtGui.QPushButton(self.widget)
        self.load_library.setObjectName(_fromUtf8("load_library"))
        self.gridLayout_3.addWidget(self.load_library, 0, 0, 1, 1)
        self.predictions_file_display = QtGui.QLineEdit(self.widget)
        self.predictions_file_display.setMaximumSize(QtCore.QSize(150, 16777215))
        self.predictions_file_display.setObjectName(_fromUtf8("predictions_file_display"))
        self.gridLayout_3.addWidget(self.predictions_file_display, 2, 1, 1, 1)
        self.load_preds_btn = QtGui.QPushButton(self.widget)
        self.load_preds_btn.setObjectName(_fromUtf8("load_preds_btn"))
        self.gridLayout_3.addWidget(self.load_preds_btn, 2, 0, 1, 1)
        self.select_folder_btn = QtGui.QPushButton(self.widget)
        self.select_folder_btn.setObjectName(_fromUtf8("select_folder_btn"))
        self.gridLayout_3.addWidget(self.select_folder_btn, 1, 0, 1, 1)
        self.export_csv = QtGui.QPushButton(self.widget)
        self.export_csv.setObjectName(_fromUtf8("export_csv"))
        self.gridLayout_3.addWidget(self.export_csv, 3, 0, 1, 1)
        self.label = QtGui.QLabel(self.widget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_3.addWidget(self.label, 0, 2, 1, 1)
        self.scroll_speed_box = QtGui.QSpinBox(self.widget)
        self.scroll_speed_box.setMaximumSize(QtCore.QSize(100, 16777215))
        self.scroll_speed_box.setMinimum(1)
        self.scroll_speed_box.setObjectName(_fromUtf8("scroll_speed_box"))
        self.gridLayout_3.addWidget(self.scroll_speed_box, 0, 3, 1, 1)
        self.label_2 = QtGui.QLabel(self.widget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_3.addWidget(self.label_2, 1, 2, 1, 1)
        self.blink_box = QtGui.QCheckBox(self.widget)
        self.blink_box.setObjectName(_fromUtf8("blink_box"))
        self.gridLayout_3.addWidget(self.blink_box, 1, 3, 1, 1)
        self.textBrowser = QtGui.QTextBrowser(self.splitter_2)
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        self.verticalLayout_2.addWidget(self.splitter_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.load_library.setText(_translate("MainWindow", "Examine Library", None))
        self.load_preds_btn.setText(_translate("MainWindow", "Load predictions", None))
        self.select_folder_btn.setText(_translate("MainWindow", "Set h5 folder", None))
        self.export_csv.setText(_translate("MainWindow", "Export seizure times", None))
        self.label.setText(_translate("MainWindow", "Scroll speed", None))
        self.label_2.setText(_translate("MainWindow", "Scroll type:", None))
        self.blink_box.setText(_translate("MainWindow", "Blinking", None))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; font-weight:600;\">Keyboard Shortcuts</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:14pt; font-weight:600;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Up: Zoom in / speed up</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Down: Zoom out/ slow down</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Right arrow: Step right/ go forwards</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Left arrow: Step left/ go backwards</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:14pt;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">SPACE: Start scrolling (or blinking)</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">B key: Toggle blink vs scroll</span></p></body></html>", None))

from pyqtgraph import GraphicsLayoutWidget
