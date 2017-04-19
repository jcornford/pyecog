# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'clf_window.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ClfManagement(object):
    def setupUi(self, ClfManagement):
        ClfManagement.setObjectName("ClfManagement")
        ClfManagement.resize(790, 990)
        ClfManagement.setMinimumSize(QtCore.QSize(790, 990))
        ClfManagement.setMaximumSize(QtCore.QSize(790, 990))
        ClfManagement.setSizeGripEnabled(False)
        self.library_path_display = QtWidgets.QLineEdit(ClfManagement)
        self.library_path_display.setGeometry(QtCore.QRect(20, 110, 750, 31))
        self.library_path_display.setMinimumSize(QtCore.QSize(500, 31))
        self.library_path_display.setReadOnly(True)
        self.library_path_display.setObjectName("library_path_display")
        self.label = QtWidgets.QLabel(ClfManagement)
        self.label.setGeometry(QtCore.QRect(20, 80, 750, 31))
        self.label.setMinimumSize(QtCore.QSize(0, 31))
        self.label.setObjectName("label")
        self.progressBar = QtWidgets.QProgressBar(ClfManagement)
        self.progressBar.setGeometry(QtCore.QRect(20, 921, 750, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar_label_below = QtWidgets.QLabel(ClfManagement)
        self.progressBar_label_below.setGeometry(QtCore.QRect(20, 960, 750, 31))
        self.progressBar_label_below.setText("")
        self.progressBar_label_below.setObjectName("progressBar_label_below")
        self.progressBar_lable_above1 = QtWidgets.QLabel(ClfManagement)
        self.progressBar_lable_above1.setGeometry(QtCore.QRect(20, 890, 750, 31))
        self.progressBar_lable_above1.setObjectName("progressBar_lable_above1")
        self.progressBar_label_above2 = QtWidgets.QLabel(ClfManagement)
        self.progressBar_label_above2.setGeometry(QtCore.QRect(20, 840, 750, 31))
        self.progressBar_label_above2.setText("")
        self.progressBar_label_above2.setObjectName("progressBar_label_above2")
        self.set_library = QtWidgets.QPushButton(ClfManagement)
        self.set_library.setGeometry(QtCore.QRect(20, 144, 200, 46))
        font = QtGui.QFont()
        font.setItalic(False)
        self.set_library.setFont(font)
        self.set_library.setObjectName("set_library")
        self.line_2 = QtWidgets.QFrame(ClfManagement)
        self.line_2.setGeometry(QtCore.QRect(20, 690, 750, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(ClfManagement)
        self.line_3.setGeometry(QtCore.QRect(20, 880, 750, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_5 = QtWidgets.QLabel(ClfManagement)
        self.label_5.setGeometry(QtCore.QRect(20, 20, 750, 50))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setIndent(-1)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(ClfManagement)
        self.label_6.setGeometry(QtCore.QRect(20, 660, 750, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setIndent(-1)
        self.label_6.setObjectName("label_6")
        self.clf_path_display = QtWidgets.QLineEdit(ClfManagement)
        self.clf_path_display.setGeometry(QtCore.QRect(20, 228, 750, 31))
        self.clf_path_display.setReadOnly(True)
        self.clf_path_display.setObjectName("clf_path_display")
        self.label_2 = QtWidgets.QLabel(ClfManagement)
        self.label_2.setGeometry(QtCore.QRect(20, 198, 750, 31))
        self.label_2.setObjectName("label_2")
        self.make_classifier = QtWidgets.QPushButton(ClfManagement)
        self.make_classifier.setGeometry(QtCore.QRect(220, 144, 200, 46))
        self.make_classifier.setObjectName("make_classifier")
        self.save_classifier = QtWidgets.QPushButton(ClfManagement)
        self.save_classifier.setGeometry(QtCore.QRect(20, 262, 200, 46))
        self.save_classifier.setObjectName("save_classifier")
        self.load_classifier = QtWidgets.QPushButton(ClfManagement)
        self.load_classifier.setGeometry(QtCore.QRect(220, 262, 200, 46))
        self.load_classifier.setObjectName("load_classifier")
        self.label_3 = QtWidgets.QLabel(ClfManagement)
        self.label_3.setGeometry(QtCore.QRect(20, 710, 750, 31))
        self.label_3.setObjectName("label_3")
        self.h5_folder_display = QtWidgets.QLineEdit(ClfManagement)
        self.h5_folder_display.setGeometry(QtCore.QRect(20, 744, 750, 31))
        self.h5_folder_display.setMinimumSize(QtCore.QSize(500, 0))
        self.h5_folder_display.setReadOnly(True)
        self.h5_folder_display.setObjectName("h5_folder_display")
        self.set_h5_folder = QtWidgets.QPushButton(ClfManagement)
        self.set_h5_folder.setGeometry(QtCore.QRect(20, 780, 200, 46))
        self.set_h5_folder.setObjectName("set_h5_folder")
        self.run_clf_on_folder = QtWidgets.QPushButton(ClfManagement)
        self.run_clf_on_folder.setGeometry(QtCore.QRect(220, 780, 200, 46))
        self.run_clf_on_folder.setObjectName("run_clf_on_folder")
        self.line_4 = QtWidgets.QFrame(ClfManagement)
        self.line_4.setGeometry(QtCore.QRect(20, 570, 750, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_7 = QtWidgets.QLabel(ClfManagement)
        self.label_7.setGeometry(QtCore.QRect(20, 540, 750, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setIndent(-1)
        self.label_7.setObjectName("label_7")
        self.estimate_error = QtWidgets.QPushButton(ClfManagement)
        self.estimate_error.setGeometry(QtCore.QRect(20, 590, 200, 46))
        self.estimate_error.setObjectName("estimate_error")
        self.cv_nfolds = QtWidgets.QLineEdit(ClfManagement)
        self.cv_nfolds.setGeometry(QtCore.QRect(400, 590, 31, 31))
        self.cv_nfolds.setObjectName("cv_nfolds")
        self.label_4 = QtWidgets.QLabel(ClfManagement)
        self.label_4.setGeometry(QtCore.QRect(230, 590, 161, 31))
        self.label_4.setObjectName("label_4")
        self.error_label = QtWidgets.QLabel(ClfManagement)
        self.error_label.setGeometry(QtCore.QRect(450, 590, 320, 31))
        self.error_label.setText("")
        self.error_label.setObjectName("error_label")
        self.frame = QtWidgets.QFrame(ClfManagement)
        self.frame.setGeometry(QtCore.QRect(20, 330, 440, 151))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setContentsMargins(12, 12, 12, 12)
        self.gridLayout.setHorizontalSpacing(12)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.n_cores = QtWidgets.QLineEdit(self.frame)
        self.n_cores.setMaximumSize(QtCore.QSize(80, 16777215))
        self.n_cores.setObjectName("n_cores")
        self.gridLayout.addWidget(self.n_cores, 0, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.frame)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 1, 0, 1, 1)
        self.downsample_bl = QtWidgets.QLineEdit(self.frame)
        self.downsample_bl.setMaximumSize(QtCore.QSize(80, 16777215))
        self.downsample_bl.setObjectName("downsample_bl")
        self.gridLayout.addWidget(self.downsample_bl, 1, 1, 1, 1)
        self.BL_weight = QtWidgets.QLineEdit(self.frame)
        self.BL_weight.setMaximumSize(QtCore.QSize(80, 16777215))
        self.BL_weight.setObjectName("BL_weight")
        self.gridLayout.addWidget(self.BL_weight, 1, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.frame)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 0, 1, 1)
        self.upsample_s_factor = QtWidgets.QLineEdit(self.frame)
        self.upsample_s_factor.setMaximumSize(QtCore.QSize(80, 16777215))
        self.upsample_s_factor.setObjectName("upsample_s_factor")
        self.gridLayout.addWidget(self.upsample_s_factor, 2, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 3, 0, 1, 1)
        self.n_trees = QtWidgets.QLineEdit(self.frame)
        self.n_trees.setMaximumSize(QtCore.QSize(80, 16777215))
        self.n_trees.setObjectName("n_trees")
        self.gridLayout.addWidget(self.n_trees, 3, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 0, 2, 1, 2)
        self.s_weight = QtWidgets.QLineEdit(self.frame)
        self.s_weight.setMaximumSize(QtCore.QSize(80, 16777215))
        self.s_weight.setObjectName("s_weight")
        self.gridLayout.addWidget(self.s_weight, 2, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.frame)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 2, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.frame)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 1, 2, 1, 1)
        self.label_12.raise_()
        self.s_weight.raise_()
        self.label_13.raise_()
        self.downsample_bl.raise_()
        self.upsample_s_factor.raise_()
        self.BL_weight.raise_()
        self.n_trees.raise_()
        self.label_11.raise_()
        self.label_10.raise_()
        self.label_9.raise_()
        self.label_8.raise_()
        self.label_14.raise_()
        self.n_cores.raise_()
        self.train_clf = QtWidgets.QPushButton(ClfManagement)
        self.train_clf.setGeometry(QtCore.QRect(20, 490, 200, 46))
        self.train_clf.setObjectName("train_clf")
        self.layoutWidget = QtWidgets.QWidget(ClfManagement)
        self.layoutWidget.setGeometry(QtCore.QRect(480, 340, 291, 141))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_n_baseline = QtWidgets.QLabel(self.layoutWidget)
        self.label_n_baseline.setText("")
        self.label_n_baseline.setObjectName("label_n_baseline")
        self.gridLayout_2.addWidget(self.label_n_baseline, 0, 0, 1, 1)
        self.label_n_seizures = QtWidgets.QLabel(self.layoutWidget)
        self.label_n_seizures.setText("")
        self.label_n_seizures.setObjectName("label_n_seizures")
        self.gridLayout_2.addWidget(self.label_n_seizures, 1, 0, 1, 1)
        self.label_resampled_numbers = QtWidgets.QLabel(self.layoutWidget)
        self.label_resampled_numbers.setText("")
        self.label_resampled_numbers.setObjectName("label_resampled_numbers")
        self.gridLayout_2.addWidget(self.label_resampled_numbers, 2, 0, 1, 1)
        self.layoutWidget.raise_()
        self.frame.raise_()
        self.library_path_display.raise_()
        self.label.raise_()
        self.progressBar.raise_()
        self.progressBar_label_below.raise_()
        self.progressBar_lable_above1.raise_()
        self.progressBar_label_above2.raise_()
        self.set_library.raise_()
        self.line_2.raise_()
        self.line_3.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.clf_path_display.raise_()
        self.label_2.raise_()
        self.make_classifier.raise_()
        self.save_classifier.raise_()
        self.load_classifier.raise_()
        self.label_3.raise_()
        self.h5_folder_display.raise_()
        self.set_h5_folder.raise_()
        self.run_clf_on_folder.raise_()
        self.line_4.raise_()
        self.label_7.raise_()
        self.estimate_error.raise_()
        self.cv_nfolds.raise_()
        self.label_4.raise_()
        self.error_label.raise_()
        self.train_clf.raise_()

        self.retranslateUi(ClfManagement)
        QtCore.QMetaObject.connectSlotsByName(ClfManagement)

    def retranslateUi(self, ClfManagement):
        _translate = QtCore.QCoreApplication.translate
        ClfManagement.setWindowTitle(_translate("ClfManagement", "Dialog"))
        self.label.setText(_translate("ClfManagement", "Library Path:"))
        self.progressBar_lable_above1.setText(_translate("ClfManagement", "Progress"))
        self.set_library.setText(_translate("ClfManagement", "Set Library"))
        self.label_5.setText(_translate("ClfManagement", "Make classifier from library"))
        self.label_6.setText(_translate("ClfManagement", "Apply classifier to a h5 folder"))
        self.label_2.setText(_translate("ClfManagement", "Classifier Path:"))
        self.make_classifier.setText(_translate("ClfManagement", "Intialise Classifier"))
        self.save_classifier.setText(_translate("ClfManagement", "Save Classifier"))
        self.load_classifier.setText(_translate("ClfManagement", "Load Classifier"))
        self.label_3.setText(_translate("ClfManagement", "h5 folder:"))
        self.set_h5_folder.setText(_translate("ClfManagement", "Choose h5 folder"))
        self.run_clf_on_folder.setText(_translate("ClfManagement", "Run clf on folder"))
        self.label_7.setText(_translate("ClfManagement", "Estimate error"))
        self.estimate_error.setText(_translate("ClfManagement", " Estimate Error"))
        self.cv_nfolds.setText(_translate("ClfManagement", "3"))
        self.label_4.setText(_translate("ClfManagement", "Cross Val n folds:"))
        self.label_8.setText(_translate("ClfManagement", "Set n cores:"))
        self.n_cores.setText(_translate("ClfManagement", "all"))
        self.label_9.setText(_translate("ClfManagement", "Downsample BL:"))
        self.downsample_bl.setText(_translate("ClfManagement", "1"))
        self.BL_weight.setText(_translate("ClfManagement", "auto"))
        self.label_10.setText(_translate("ClfManagement", "Upsample seizures:"))
        self.upsample_s_factor.setText(_translate("ClfManagement", "1"))
        self.label_14.setText(_translate("ClfManagement", "No. Trees"))
        self.n_trees.setText(_translate("ClfManagement", "800"))
        self.label_11.setText(_translate("ClfManagement", "Class weights"))
        self.s_weight.setText(_translate("ClfManagement", "auto"))
        self.label_13.setText(_translate("ClfManagement", "S"))
        self.label_12.setText(_translate("ClfManagement", "BL"))
        self.train_clf.setText(_translate("ClfManagement", " Train Classifier"))

