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
        ClfManagement.resize(745, 639)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ClfManagement.sizePolicy().hasHeightForWidth())
        ClfManagement.setSizePolicy(sizePolicy)
        ClfManagement.setMaximumSize(QtCore.QSize(745, 639))
        ClfManagement.setSizeGripEnabled(False)
        self.gridLayout_7 = QtWidgets.QGridLayout(ClfManagement)
        self.gridLayout_7.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.groupBox = QtWidgets.QGroupBox(ClfManagement)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_6.setHorizontalSpacing(9)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.frame_4 = QtWidgets.QFrame(self.groupBox)
        self.frame_4.setMaximumSize(QtCore.QSize(16777215, 62))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setContentsMargins(4, 4, 4, 4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setIndent(-1)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 0, 0, 1, 1)
        self.set_library = QtWidgets.QPushButton(self.frame_4)
        font = QtGui.QFont()
        font.setItalic(False)
        self.set_library.setFont(font)
        self.set_library.setObjectName("set_library")
        self.gridLayout_4.addWidget(self.set_library, 0, 1, 1, 1)
        self.library_path_display = QtWidgets.QLineEdit(self.frame_4)
        self.library_path_display.setMinimumSize(QtCore.QSize(0, 0))
        self.library_path_display.setReadOnly(True)
        self.library_path_display.setObjectName("library_path_display")
        self.gridLayout_4.addWidget(self.library_path_display, 0, 2, 1, 1)
        self.gridLayout_6.addWidget(self.frame_4, 0, 0, 1, 1)
        self.widget_6 = QtWidgets.QWidget(self.groupBox)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_2 = QtWidgets.QFrame(self.widget_6)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_8.setContentsMargins(13, 9, 13, 2)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_8.addWidget(self.label_2, 0, 0, 1, 2)
        self.clf_path_display = QtWidgets.QLineEdit(self.frame_2)
        self.clf_path_display.setReadOnly(True)
        self.clf_path_display.setObjectName("clf_path_display")
        self.gridLayout_8.addWidget(self.clf_path_display, 1, 0, 1, 2)
        self.save_classifier = QtWidgets.QPushButton(self.frame_2)
        self.save_classifier.setObjectName("save_classifier")
        self.gridLayout_8.addWidget(self.save_classifier, 2, 0, 1, 1)
        self.load_classifier = QtWidgets.QPushButton(self.frame_2)
        self.load_classifier.setObjectName("load_classifier")
        self.gridLayout_8.addWidget(self.load_classifier, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setEnabled(False)
        self.label_4.setObjectName("label_4")
        self.gridLayout_8.addWidget(self.label_4, 3, 0, 1, 1)
        self.cv_nfolds = QtWidgets.QLineEdit(self.frame_2)
        self.cv_nfolds.setEnabled(False)
        self.cv_nfolds.setObjectName("cv_nfolds")
        self.gridLayout_8.addWidget(self.cv_nfolds, 3, 1, 1, 1)
        self.estimate_error = QtWidgets.QPushButton(self.frame_2)
        self.estimate_error.setEnabled(False)
        self.estimate_error.setObjectName("estimate_error")
        self.gridLayout_8.addWidget(self.estimate_error, 4, 0, 1, 1)
        self.error_label = QtWidgets.QLabel(self.frame_2)
        self.error_label.setObjectName("error_label")
        self.gridLayout_8.addWidget(self.error_label, 5, 0, 1, 2)
        self.load_classifier.raise_()
        self.label_2.raise_()
        self.save_classifier.raise_()
        self.estimate_error.raise_()
        self.clf_path_display.raise_()
        self.label_4.raise_()
        self.cv_nfolds.raise_()
        self.error_label.raise_()
        self.horizontalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.widget_6)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_5.setContentsMargins(13, 2, 13, 9)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.progressBar_lable_above1 = QtWidgets.QLabel(self.frame_3)
        self.progressBar_lable_above1.setObjectName("progressBar_lable_above1")
        self.gridLayout_5.addWidget(self.progressBar_lable_above1, 7, 0, 1, 2)
        self.run_clf_on_folder = QtWidgets.QPushButton(self.frame_3)
        self.run_clf_on_folder.setObjectName("run_clf_on_folder")
        self.gridLayout_5.addWidget(self.run_clf_on_folder, 5, 1, 1, 1)
        self.progressBar_label_above2 = QtWidgets.QLabel(self.frame_3)
        self.progressBar_label_above2.setText("")
        self.progressBar_label_above2.setObjectName("progressBar_label_above2")
        self.gridLayout_5.addWidget(self.progressBar_label_above2, 6, 0, 1, 1)
        self.progressBar_label_below = QtWidgets.QLabel(self.frame_3)
        self.progressBar_label_below.setText("")
        self.progressBar_label_below.setObjectName("progressBar_label_below")
        self.gridLayout_5.addWidget(self.progressBar_label_below, 6, 1, 1, 1)
        self.set_h5_folder = QtWidgets.QPushButton(self.frame_3)
        self.set_h5_folder.setObjectName("set_h5_folder")
        self.gridLayout_5.addWidget(self.set_h5_folder, 5, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.frame_3)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_5.addWidget(self.progressBar, 8, 0, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setIndent(-1)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 4, 0, 1, 1)
        self.h5_folder_display = QtWidgets.QLineEdit(self.frame_3)
        self.h5_folder_display.setReadOnly(True)
        self.h5_folder_display.setObjectName("h5_folder_display")
        self.gridLayout_5.addWidget(self.h5_folder_display, 4, 1, 1, 1)
        self.horizontalLayout.addWidget(self.frame_3)
        self.gridLayout_6.addWidget(self.widget_6, 2, 0, 1, 1)
        self.frame_41 = QtWidgets.QFrame(self.groupBox)
        self.frame_41.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_41.setObjectName("frame_41")
        self.formLayout = QtWidgets.QFormLayout(self.frame_41)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setHorizontalSpacing(9)
        self.formLayout.setVerticalSpacing(0)
        self.formLayout.setObjectName("formLayout")
        self.widget = QtWidgets.QWidget(self.frame_41)
        self.widget.setObjectName("widget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_3.setContentsMargins(2, 2, 2, 2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.make_classifier = QtWidgets.QPushButton(self.widget)
        self.make_classifier.setObjectName("make_classifier")
        self.gridLayout_3.addWidget(self.make_classifier, 0, 0, 1, 1)
        self.train_clf = QtWidgets.QPushButton(self.widget)
        self.train_clf.setObjectName("train_clf")
        self.gridLayout_3.addWidget(self.train_clf, 0, 1, 1, 1)
        self.frame = QtWidgets.QFrame(self.widget)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setContentsMargins(12, 12, 12, 12)
        self.gridLayout.setHorizontalSpacing(12)
        self.gridLayout.setVerticalSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.label_15 = QtWidgets.QLabel(self.frame)
        self.label_15.setEnabled(False)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 20, 2, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.frame)
        self.label_16.setEnabled(False)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 20, 3, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.cc_sampling_box = QtWidgets.QCheckBox(self.frame)
        self.cc_sampling_box.setEnabled(False)
        self.cc_sampling_box.setObjectName("cc_sampling_box")
        self.gridLayout.addWidget(self.cc_sampling_box, 20, 0, 1, 2)
        self.n_trees = QtWidgets.QLineEdit(self.frame)
        self.n_trees.setMaximumSize(QtCore.QSize(80, 16777215))
        self.n_trees.setObjectName("n_trees")
        self.gridLayout.addWidget(self.n_trees, 8, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame)
        self.label_11.setEnabled(False)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 0, 3, 1, 2)
        self.probabilistic_hmm_box = QtWidgets.QCheckBox(self.frame)
        self.probabilistic_hmm_box.setObjectName("probabilistic_hmm_box")
        self.gridLayout.addWidget(self.probabilistic_hmm_box, 18, 0, 1, 1)
        self.cc_total_hours = QtWidgets.QLineEdit(self.frame)
        self.cc_total_hours.setEnabled(False)
        self.cc_total_hours.setMaximumSize(QtCore.QSize(80, 16777215))
        self.cc_total_hours.setObjectName("cc_total_hours")
        self.gridLayout.addWidget(self.cc_total_hours, 20, 4, 1, 1)
        self.line_9 = QtWidgets.QFrame(self.frame)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.gridLayout.addWidget(self.line_9, 16, 0, 1, 1)
        self.line_7 = QtWidgets.QFrame(self.frame)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout.addWidget(self.line_7, 1, 2, 1, 1)
        self.line_6 = QtWidgets.QFrame(self.frame)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout.addWidget(self.line_6, 1, 0, 1, 1)
        self.lg_box = QtWidgets.QCheckBox(self.frame)
        self.lg_box.setObjectName("lg_box")
        self.gridLayout.addWidget(self.lg_box, 2, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.frame)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 14, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 3, 0, 1, 1)
        self.rf_box = QtWidgets.QCheckBox(self.frame)
        self.rf_box.setEnabled(True)
        self.rf_box.setChecked(True)
        self.rf_box.setAutoExclusive(False)
        self.rf_box.setObjectName("rf_box")
        self.gridLayout.addWidget(self.rf_box, 7, 0, 1, 2)
        self.line_5 = QtWidgets.QFrame(self.frame)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout.addWidget(self.line_5, 9, 2, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.frame)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 3, 2, 1, 1)
        self.line_8 = QtWidgets.QFrame(self.frame)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.gridLayout.addWidget(self.line_8, 19, 0, 1, 1)
        self.n_cores = QtWidgets.QLineEdit(self.frame)
        self.n_cores.setMaximumSize(QtCore.QSize(80, 16777215))
        self.n_cores.setObjectName("n_cores")
        self.gridLayout.addWidget(self.n_cores, 0, 2, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.frame)
        self.label_13.setEnabled(False)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 3, 3, 1, 1)
        self.downsample_bl = QtWidgets.QLineEdit(self.frame)
        self.downsample_bl.setMaximumSize(QtCore.QSize(80, 16777215))
        self.downsample_bl.setObjectName("downsample_bl")
        self.gridLayout.addWidget(self.downsample_bl, 15, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.frame)
        self.label_12.setEnabled(False)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 2, 3, 1, 1)
        self.s_weight = QtWidgets.QLineEdit(self.frame)
        self.s_weight.setEnabled(False)
        self.s_weight.setMaximumSize(QtCore.QSize(80, 16777215))
        self.s_weight.setObjectName("s_weight")
        self.gridLayout.addWidget(self.s_weight, 3, 4, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 8, 0, 1, 2)
        self.line_4 = QtWidgets.QFrame(self.frame)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 9, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.frame)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 15, 0, 1, 1)
        self.upsample_s_factor = QtWidgets.QLineEdit(self.frame)
        self.upsample_s_factor.setMaximumSize(QtCore.QSize(80, 16777215))
        self.upsample_s_factor.setObjectName("upsample_s_factor")
        self.gridLayout.addWidget(self.upsample_s_factor, 14, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 10, 0, 1, 3)
        self.BL_weight = QtWidgets.QLineEdit(self.frame)
        self.BL_weight.setEnabled(False)
        self.BL_weight.setMaximumSize(QtCore.QSize(80, 16777215))
        self.BL_weight.setObjectName("BL_weight")
        self.gridLayout.addWidget(self.BL_weight, 2, 4, 1, 1)
        self.n_trees.raise_()
        self.label_8.raise_()
        self.label_14.raise_()
        self.n_cores.raise_()
        self.label_7.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.upsample_s_factor.raise_()
        self.downsample_bl.raise_()
        self.label_11.raise_()
        self.rf_box.raise_()
        self.lg_box.raise_()
        self.BL_weight.raise_()
        self.label_12.raise_()
        self.s_weight.raise_()
        self.label_13.raise_()
        self.line_4.raise_()
        self.line_5.raise_()
        self.line_6.raise_()
        self.line_7.raise_()
        self.cc_sampling_box.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.cc_total_hours.raise_()
        self.line_9.raise_()
        self.probabilistic_hmm_box.raise_()
        self.line_8.raise_()
        self.line.raise_()
        self.line_3.raise_()
        self.gridLayout_3.addWidget(self.frame, 1, 0, 1, 2)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.widget)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_n_seizures = QtWidgets.QLabel(self.frame_41)
        self.label_n_seizures.setText("")
        self.label_n_seizures.setObjectName("label_n_seizures")
        self.gridLayout_2.addWidget(self.label_n_seizures, 1, 0, 1, 1)
        self.label_resampled_numbers = QtWidgets.QLabel(self.frame_41)
        self.label_resampled_numbers.setText("")
        self.label_resampled_numbers.setObjectName("label_resampled_numbers")
        self.gridLayout_2.addWidget(self.label_resampled_numbers, 2, 0, 1, 1)
        self.label_n_baseline = QtWidgets.QLabel(self.frame_41)
        self.label_n_baseline.setText("")
        self.label_n_baseline.setObjectName("label_n_baseline")
        self.gridLayout_2.addWidget(self.label_n_baseline, 0, 0, 1, 1)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.gridLayout_2)
        self.gridLayout_6.addWidget(self.frame_41, 1, 0, 1, 1)
        self.widget_6.raise_()
        self.frame_4.raise_()
        self.frame_4.raise_()
        self.gridLayout_7.addWidget(self.groupBox, 0, 0, 1, 1)

        self.retranslateUi(ClfManagement)
        QtCore.QMetaObject.connectSlotsByName(ClfManagement)

    def retranslateUi(self, ClfManagement):
        _translate = QtCore.QCoreApplication.translate
        ClfManagement.setWindowTitle(_translate("ClfManagement", "Dialog"))
        self.label_5.setText(_translate("ClfManagement", "Make classifier from library:"))
        self.set_library.setText(_translate("ClfManagement", "Set Library"))
        self.label_2.setText(_translate("ClfManagement", "Classifier Path:"))
        self.save_classifier.setText(_translate("ClfManagement", "Save Classifier"))
        self.load_classifier.setText(_translate("ClfManagement", "Load Classifier"))
        self.label_4.setText(_translate("ClfManagement", "Cross Val n folds"))
        self.cv_nfolds.setText(_translate("ClfManagement", "3"))
        self.estimate_error.setText(_translate("ClfManagement", " Estimate Error (todo)"))
        self.error_label.setText(_translate("ClfManagement", "."))
        self.progressBar_lable_above1.setText(_translate("ClfManagement", "Progress:"))
        self.run_clf_on_folder.setText(_translate("ClfManagement", "Run classifier on folder"))
        self.set_h5_folder.setText(_translate("ClfManagement", "Choose h5 folder"))
        self.label_6.setText(_translate("ClfManagement", "Apply classifier to a h5 folder"))
        self.label_3.setText(_translate("ClfManagement", "h5 folder:"))
        self.make_classifier.setText(_translate("ClfManagement", "Intialise Classifier"))
        self.train_clf.setText(_translate("ClfManagement", " Train Classifier"))
        self.label_15.setText(_translate("ClfManagement", "Total  hours"))
        self.label_16.setText(_translate("ClfManagement", "analysed"))
        self.label_8.setText(_translate("ClfManagement", "Set n cores:"))
        self.cc_sampling_box.setText(_translate("ClfManagement", "Case control sampling"))
        self.n_trees.setText(_translate("ClfManagement", "800"))
        self.label_11.setText(_translate("ClfManagement", "Class weights (todo)"))
        self.probabilistic_hmm_box.setText(_translate("ClfManagement", "Probabilisitc HMM "))
        self.cc_total_hours.setText(_translate("ClfManagement", "(todo)"))
        self.lg_box.setText(_translate("ClfManagement", "Logistic regression"))
        self.label_10.setText(_translate("ClfManagement", "Upsample seizures:"))
        self.rf_box.setText(_translate("ClfManagement", "Random Forest"))
        self.n_cores.setText(_translate("ClfManagement", "all"))
        self.label_13.setText(_translate("ClfManagement", "S"))
        self.downsample_bl.setText(_translate("ClfManagement", "1"))
        self.label_12.setText(_translate("ClfManagement", "BL"))
        self.s_weight.setText(_translate("ClfManagement", "auto"))
        self.label_14.setText(_translate("ClfManagement", "No. Trees"))
        self.label_9.setText(_translate("ClfManagement", "Downsample BL:"))
        self.upsample_s_factor.setText(_translate("ClfManagement", "1"))
        self.label_7.setText(_translate("ClfManagement", "Resample chunks for descriminative clf"))
        self.BL_weight.setText(_translate("ClfManagement", "auto"))
