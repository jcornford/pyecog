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
        MainWindow.resize(936, 716)
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
        self.treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.treeWidget.headerItem().setText(1, _fromUtf8("2"))
        self.treeWidget.headerItem().setText(2, _fromUtf8("3"))
        self.treeWidget.headerItem().setText(3, _fromUtf8("4"))
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
        self.h5_folder_display.setMaximumSize(QtCore.QSize(300, 100))
        self.h5_folder_display.setDragEnabled(True)
        self.h5_folder_display.setReadOnly(True)
        self.h5_folder_display.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.h5_folder_display.setObjectName(_fromUtf8("h5_folder_display"))
        self.gridLayout_3.addWidget(self.h5_folder_display, 1, 1, 1, 1)
        self.blink_box = QtGui.QCheckBox(self.widget)
        self.blink_box.setObjectName(_fromUtf8("blink_box"))
        self.gridLayout_3.addWidget(self.blink_box, 1, 3, 1, 1)
        self.label_2 = QtGui.QLabel(self.widget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_3.addWidget(self.label_2, 1, 2, 1, 1)
        self.scroll_speed_box = QtGui.QSpinBox(self.widget)
        self.scroll_speed_box.setMaximumSize(QtCore.QSize(100, 16777215))
        self.scroll_speed_box.setMinimum(1)
        self.scroll_speed_box.setObjectName(_fromUtf8("scroll_speed_box"))
        self.gridLayout_3.addWidget(self.scroll_speed_box, 0, 3, 1, 1)
        self.label = QtGui.QLabel(self.widget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_3.addWidget(self.label, 0, 2, 1, 1)
        self.predictions_file_display = QtGui.QLineEdit(self.widget)
        self.predictions_file_display.setMaximumSize(QtCore.QSize(300, 16777215))
        self.predictions_file_display.setObjectName(_fromUtf8("predictions_file_display"))
        self.gridLayout_3.addWidget(self.predictions_file_display, 0, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.widget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.widget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.textBrowser = QtGui.QTextBrowser(self.splitter_2)
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        self.verticalLayout_2.addWidget(self.splitter_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 936, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        self.menuFile = QtGui.QMenu(self.menuBar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuAnalyse = QtGui.QMenu(self.menuBar)
        self.menuAnalyse.setObjectName(_fromUtf8("menuAnalyse"))
        MainWindow.setMenuBar(self.menuBar)
        self.actionSave_annotations = QtGui.QAction(MainWindow)
        self.actionSave_annotations.setObjectName(_fromUtf8("actionSave_annotations"))
        self.actionLoad_Library = QtGui.QAction(MainWindow)
        self.actionLoad_Library.setObjectName(_fromUtf8("actionLoad_Library"))
        self.actionLoad_Predictions = QtGui.QAction(MainWindow)
        self.actionLoad_Predictions.setObjectName(_fromUtf8("actionLoad_Predictions"))
        self.actionLoad_h5_folder = QtGui.QAction(MainWindow)
        self.actionLoad_h5_folder.setObjectName(_fromUtf8("actionLoad_h5_folder"))
        self.actionConvert_dir_to_h5 = QtGui.QAction(MainWindow)
        self.actionConvert_dir_to_h5.setObjectName(_fromUtf8("actionConvert_dir_to_h5"))
        self.actionConvert_ndf_to_h5 = QtGui.QAction(MainWindow)
        self.actionConvert_ndf_to_h5.setObjectName(_fromUtf8("actionConvert_ndf_to_h5"))
        self.actionMake_library = QtGui.QAction(MainWindow)
        self.actionMake_library.setObjectName(_fromUtf8("actionMake_library"))
        self.actionAdd_to_library = QtGui.QAction(MainWindow)
        self.actionAdd_to_library.setObjectName(_fromUtf8("actionAdd_to_library"))
        self.actionAdd_labels_to_library = QtGui.QAction(MainWindow)
        self.actionAdd_labels_to_library.setObjectName(_fromUtf8("actionAdd_labels_to_library"))
        self.actionAdd_features_to_library = QtGui.QAction(MainWindow)
        self.actionAdd_features_to_library.setObjectName(_fromUtf8("actionAdd_features_to_library"))
        self.actionTrain_classifier_on_library = QtGui.QAction(MainWindow)
        self.actionTrain_classifier_on_library.setObjectName(_fromUtf8("actionTrain_classifier_on_library"))
        self.actionRun_classifer_on_h5_dir = QtGui.QAction(MainWindow)
        self.actionRun_classifer_on_h5_dir.setObjectName(_fromUtf8("actionRun_classifer_on_h5_dir"))
        self.actionRun_classifer_on_ndf_dir = QtGui.QAction(MainWindow)
        self.actionRun_classifer_on_ndf_dir.setObjectName(_fromUtf8("actionRun_classifer_on_ndf_dir"))
        self.actionSet_default_folder = QtGui.QAction(MainWindow)
        self.actionSet_default_folder.setObjectName(_fromUtf8("actionSet_default_folder"))
        self.menuFile.addAction(self.actionLoad_Library)
        self.menuFile.addAction(self.actionLoad_Predictions)
        self.menuFile.addAction(self.actionLoad_h5_folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_annotations)
        self.menuFile.addAction(self.actionSet_default_folder)
        self.menuAnalyse.addAction(self.actionConvert_ndf_to_h5)
        self.menuAnalyse.addSeparator()
        self.menuAnalyse.addAction(self.actionMake_library)
        self.menuAnalyse.addAction(self.actionAdd_to_library)
        self.menuAnalyse.addSeparator()
        self.menuAnalyse.addAction(self.actionAdd_labels_to_library)
        self.menuAnalyse.addAction(self.actionAdd_features_to_library)
        self.menuAnalyse.addSeparator()
        self.menuAnalyse.addAction(self.actionTrain_classifier_on_library)
        self.menuAnalyse.addAction(self.actionRun_classifer_on_h5_dir)
        self.menuAnalyse.addAction(self.actionRun_classifer_on_ndf_dir)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuAnalyse.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.blink_box.setText(_translate("MainWindow", "Blinking", None))
        self.label_2.setText(_translate("MainWindow", "Scroll type:", None))
        self.label.setText(_translate("MainWindow", "Scroll speed", None))
        self.label_3.setText(_translate("MainWindow", "Prediction File:", None))
        self.label_4.setText(_translate("MainWindow", "h5 Folder:", None))
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
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuAnalyse.setTitle(_translate("MainWindow", "Analyse", None))
        self.actionSave_annotations.setText(_translate("MainWindow", "Save annotations", None))
        self.actionLoad_Library.setText(_translate("MainWindow", "Load Library", None))
        self.actionLoad_Predictions.setText(_translate("MainWindow", "Load Predictions", None))
        self.actionLoad_h5_folder.setText(_translate("MainWindow", "Load h5 folder", None))
        self.actionConvert_dir_to_h5.setText(_translate("MainWindow", "Convert dir to h5", None))
        self.actionConvert_ndf_to_h5.setText(_translate("MainWindow", "Convert ndf folder to h5", None))
        self.actionMake_library.setText(_translate("MainWindow", "Make library", None))
        self.actionAdd_to_library.setText(_translate("MainWindow", "Add to library", None))
        self.actionAdd_labels_to_library.setText(_translate("MainWindow", "Add labels to library", None))
        self.actionAdd_features_to_library.setText(_translate("MainWindow", "Add features to library", None))
        self.actionTrain_classifier_on_library.setText(_translate("MainWindow", "Train classifier on library", None))
        self.actionRun_classifer_on_h5_dir.setText(_translate("MainWindow", "Run classifer on h5 dir", None))
        self.actionRun_classifer_on_ndf_dir.setText(_translate("MainWindow", "Run classifer on ndf dir", None))
        self.actionSet_default_folder.setText(_translate("MainWindow", "Set default folder", None))

from pyqtgraph import GraphicsLayoutWidget
