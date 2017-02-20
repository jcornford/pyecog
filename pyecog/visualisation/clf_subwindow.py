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
        ClfManagement.resize(881, 557)
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        ClfManagement.setFont(font)
        self.library_path_display = QtWidgets.QLineEdit(ClfManagement)
        self.library_path_display.setGeometry(QtCore.QRect(50, 60, 741, 21))
        self.library_path_display.setMinimumSize(QtCore.QSize(500, 0))
        self.library_path_display.setReadOnly(True)
        self.library_path_display.setObjectName("library_path_display")
        self.label = QtWidgets.QLabel(ClfManagement)
        self.label.setGeometry(QtCore.QRect(50, 40, 81, 16))
        self.label.setObjectName("label")
        self.progressBar = QtWidgets.QProgressBar(ClfManagement)
        self.progressBar.setGeometry(QtCore.QRect(40, 480, 751, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.progressBar_label_below = QtWidgets.QLabel(ClfManagement)
        self.progressBar_label_below.setGeometry(QtCore.QRect(40, 510, 751, 16))
        self.progressBar_label_below.setText("")
        self.progressBar_label_below.setObjectName("progressBar_label_below")
        self.progressBar_lable_above1 = QtWidgets.QLabel(ClfManagement)
        self.progressBar_lable_above1.setGeometry(QtCore.QRect(40, 460, 741, 16))
        self.progressBar_lable_above1.setObjectName("progressBar_lable_above1")
        self.progressBar_label_above2 = QtWidgets.QLabel(ClfManagement)
        self.progressBar_label_above2.setGeometry(QtCore.QRect(40, 440, 751, 16))
        self.progressBar_label_above2.setText("")
        self.progressBar_label_above2.setObjectName("progressBar_label_above2")
        self.set_library = QtWidgets.QPushButton(ClfManagement)
        self.set_library.setGeometry(QtCore.QRect(50, 90, 131, 32))
        self.set_library.setObjectName("set_library")
        self.line_2 = QtWidgets.QFrame(ClfManagement)
        self.line_2.setGeometry(QtCore.QRect(50, 340, 761, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(ClfManagement)
        self.line_3.setGeometry(QtCore.QRect(50, 420, 751, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_5 = QtWidgets.QLabel(ClfManagement)
        self.label_5.setGeometry(QtCore.QRect(320, 10, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(19)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setIndent(-1)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(ClfManagement)
        self.label_6.setGeometry(QtCore.QRect(310, 310, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(19)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setIndent(-1)
        self.label_6.setObjectName("label_6")
        self.clf_path_display = QtWidgets.QLineEdit(ClfManagement)
        self.clf_path_display.setGeometry(QtCore.QRect(50, 150, 741, 21))
        self.clf_path_display.setMinimumSize(QtCore.QSize(500, 0))
        self.clf_path_display.setReadOnly(True)
        self.clf_path_display.setObjectName("clf_path_display")
        self.label_2 = QtWidgets.QLabel(ClfManagement)
        self.label_2.setGeometry(QtCore.QRect(50, 130, 111, 16))
        self.label_2.setObjectName("label_2")
        self.make_classifier = QtWidgets.QPushButton(ClfManagement)
        self.make_classifier.setGeometry(QtCore.QRect(180, 90, 131, 32))
        self.make_classifier.setObjectName("make_classifier")
        self.save_classifier = QtWidgets.QPushButton(ClfManagement)
        self.save_classifier.setGeometry(QtCore.QRect(50, 180, 131, 32))
        self.save_classifier.setObjectName("save_classifier")
        self.load_classifier = QtWidgets.QPushButton(ClfManagement)
        self.load_classifier.setGeometry(QtCore.QRect(180, 180, 131, 32))
        self.load_classifier.setObjectName("load_classifier")
        self.label_3 = QtWidgets.QLabel(ClfManagement)
        self.label_3.setGeometry(QtCore.QRect(60, 360, 61, 16))
        self.label_3.setObjectName("label_3")
        self.h5_folder_display = QtWidgets.QLineEdit(ClfManagement)
        self.h5_folder_display.setGeometry(QtCore.QRect(120, 360, 671, 21))
        self.h5_folder_display.setMinimumSize(QtCore.QSize(500, 0))
        self.h5_folder_display.setReadOnly(True)
        self.h5_folder_display.setObjectName("h5_folder_display")
        self.set_h5_folder = QtWidgets.QPushButton(ClfManagement)
        self.set_h5_folder.setGeometry(QtCore.QRect(50, 390, 131, 32))
        self.set_h5_folder.setObjectName("set_h5_folder")
        self.run_clf_on_folder = QtWidgets.QPushButton(ClfManagement)
        self.run_clf_on_folder.setGeometry(QtCore.QRect(180, 390, 131, 32))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.run_clf_on_folder.setFont(font)
        self.run_clf_on_folder.setObjectName("run_clf_on_folder")
        self.train_clf = QtWidgets.QPushButton(ClfManagement)
        self.train_clf.setGeometry(QtCore.QRect(320, 180, 131, 32))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.train_clf.setFont(font)
        self.train_clf.setObjectName("train_clf")
        self.line_4 = QtWidgets.QFrame(ClfManagement)
        self.line_4.setGeometry(QtCore.QRect(50, 250, 761, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_7 = QtWidgets.QLabel(ClfManagement)
        self.label_7.setGeometry(QtCore.QRect(360, 220, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica Neue")
        font.setPointSize(19)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setIndent(-1)
        self.label_7.setObjectName("label_7")
        self.estimate_error = QtWidgets.QPushButton(ClfManagement)
        self.estimate_error.setGeometry(QtCore.QRect(50, 270, 131, 32))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.estimate_error.setFont(font)
        self.estimate_error.setObjectName("estimate_error")
        self.cv_nfolds = QtWidgets.QLineEdit(ClfManagement)
        self.cv_nfolds.setGeometry(QtCore.QRect(310, 270, 31, 21))
        self.cv_nfolds.setObjectName("cv_nfolds")
        self.label_4 = QtWidgets.QLabel(ClfManagement)
        self.label_4.setGeometry(QtCore.QRect(200, 270, 101, 20))
        self.label_4.setObjectName("label_4")
        self.error_label = QtWidgets.QLabel(ClfManagement)
        self.error_label.setGeometry(QtCore.QRect(370, 269, 441, 31))
        self.error_label.setText("")
        self.error_label.setObjectName("error_label")
        self.n_cores = QtWidgets.QLineEdit(ClfManagement)
        self.n_cores.setGeometry(QtCore.QRect(740, 190, 31, 21))
        self.n_cores.setObjectName("n_cores")
        self.label_8 = QtWidgets.QLabel(ClfManagement)
        self.label_8.setGeometry(QtCore.QRect(650, 190, 91, 20))
        self.label_8.setObjectName("label_8")

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
        self.train_clf.setText(_translate("ClfManagement", " Train Classifier"))
        self.label_7.setText(_translate("ClfManagement", "Estimate error"))
        self.estimate_error.setText(_translate("ClfManagement", " Estimate Error"))
        self.cv_nfolds.setText(_translate("ClfManagement", "3"))
        self.label_4.setText(_translate("ClfManagement", "Cross Val n folds:"))
        self.n_cores.setText(_translate("ClfManagement", "3"))
        self.label_8.setText(_translate("ClfManagement", "Set n cores:"))

