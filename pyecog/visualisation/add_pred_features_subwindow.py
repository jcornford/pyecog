# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'add_prediction_features.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_make_features(object):
    def setupUi(self, make_features):
        make_features.setObjectName("make_features")
        make_features.resize(755, 562)
        self.formLayoutWidget = QtWidgets.QWidget(make_features)
        self.formLayoutWidget.setGeometry(QtCore.QRect(70, 70, 537, 131))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.set_h5_folder = QtWidgets.QPushButton(self.formLayoutWidget)
        self.set_h5_folder.setMinimumSize(QtCore.QSize(200, 0))
        self.set_h5_folder.setMaximumSize(QtCore.QSize(200, 32))
        self.set_h5_folder.setObjectName("set_h5_folder")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.set_h5_folder)
        self.h5_display = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.h5_display.setMinimumSize(QtCore.QSize(300, 0))
        self.h5_display.setObjectName("h5_display")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.h5_display)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label)
        self.fs_box = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.fs_box.setObjectName("fs_box")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.fs_box)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.cores_to_use = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.cores_to_use.setObjectName("cores_to_use")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.cores_to_use)
        self.transmitter_ids = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.transmitter_ids.setObjectName("transmitter_ids")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.transmitter_ids)
        self.progressBar = QtWidgets.QProgressBar(make_features)
        self.progressBar.setGeometry(QtCore.QRect(60, 342, 541, 31))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.progress_bar_label = QtWidgets.QLabel(make_features)
        self.progress_bar_label.setGeometry(QtCore.QRect(60, 330, 461, 16))
        self.progress_bar_label.setObjectName("progress_bar_label")
        self.convert_button = QtWidgets.QPushButton(make_features)
        self.convert_button.setGeometry(QtCore.QRect(280, 250, 181, 32))
        self.convert_button.setObjectName("convert_button")
        self.logpath_dsplay = QtWidgets.QLabel(make_features)
        self.logpath_dsplay.setGeometry(QtCore.QRect(60, 380, 600, 16))
        self.logpath_dsplay.setMinimumSize(QtCore.QSize(600, 0))
        self.logpath_dsplay.setObjectName("logpath_dsplay")
        self.hidden_label = QtWidgets.QLabel(make_features)
        self.hidden_label.setGeometry(QtCore.QRect(60, 300, 600, 16))
        self.hidden_label.setMinimumSize(QtCore.QSize(600, 0))
        self.hidden_label.setText("")
        self.hidden_label.setObjectName("hidden_label")
        self.label_4 = QtWidgets.QLabel(make_features)
        self.label_4.setGeometry(QtCore.QRect(180, 30, 441, 16))
        self.label_4.setObjectName("label_4")

        self.retranslateUi(make_features)
        QtCore.QMetaObject.connectSlotsByName(make_features)

    def retranslateUi(self, make_features):
        _translate = QtCore.QCoreApplication.translate
        make_features.setWindowTitle(_translate("make_features", "Dialog"))
        self.set_h5_folder.setText(_translate("make_features", "Choose h5 folder"))
        self.label_2.setText(_translate("make_features", "Select transmitter IDs"))
        self.label.setText(_translate("make_features", "Fs (Hz): **Not used as h5 "))
        self.fs_box.setText(_translate("make_features", "512"))
        self.label_3.setText(_translate("make_features", "Number of CPU cores to use"))
        self.cores_to_use.setText(_translate("make_features", "all"))
        self.transmitter_ids.setText(_translate("make_features", "all"))
        self.progress_bar_label.setText(_translate("make_features", "Progress:"))
        self.convert_button.setText(_translate("make_features", "Extract Features"))
        self.logpath_dsplay.setText(_translate("make_features", "Path to Logfile:"))
        self.label_4.setText(_translate("make_features", "** EVENTUALLY WILL IMPLEMENT EXTRACTION FROM NDFs **"))

