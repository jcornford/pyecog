# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'predictions_loading_subwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(582, 418)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(310, 130, 169, 23))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayoutWidget = QtWidgets.QWidget(Dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(70, 40, 421, 71))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.prediction_display = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.prediction_display.setObjectName("prediction_display")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.prediction_display)
        self.h5_display = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.h5_display.setObjectName("h5_display")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.h5_display)
        self.set_prediction_file = QtWidgets.QPushButton(self.formLayoutWidget)
        self.set_prediction_file.setMaximumSize(QtCore.QSize(170, 16777215))
        self.set_prediction_file.setObjectName("set_prediction_file")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.set_prediction_file)
        self.set_h5_folder = QtWidgets.QPushButton(self.formLayoutWidget)
        self.set_h5_folder.setMaximumSize(QtCore.QSize(141, 32))
        self.set_h5_folder.setObjectName("set_h5_folder")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.set_h5_folder)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.set_prediction_file.setText(_translate("Dialog", "Choose Prediction file"))
        self.set_h5_folder.setText(_translate("Dialog", "Choose h5 folder"))

