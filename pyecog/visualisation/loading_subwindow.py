# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'predictions_loading_subwindow.ui'
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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(582, 418)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(310, 130, 169, 23))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.formLayoutWidget = QtGui.QWidget(Dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(70, 40, 421, 71))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout_2 = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.prediction_display = QtGui.QLineEdit(self.formLayoutWidget)
        self.prediction_display.setObjectName(_fromUtf8("prediction_display"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.prediction_display)
        self.h5_display = QtGui.QLineEdit(self.formLayoutWidget)
        self.h5_display.setObjectName(_fromUtf8("h5_display"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.h5_display)
        self.set_prediction_file = QtGui.QPushButton(self.formLayoutWidget)
        self.set_prediction_file.setMaximumSize(QtCore.QSize(170, 16777215))
        self.set_prediction_file.setObjectName(_fromUtf8("set_prediction_file"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.set_prediction_file)
        self.set_h5_folder = QtGui.QPushButton(self.formLayoutWidget)
        self.set_h5_folder.setMaximumSize(QtCore.QSize(141, 32))
        self.set_h5_folder.setObjectName(_fromUtf8("set_h5_folder"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.LabelRole, self.set_h5_folder)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.set_prediction_file.setText(_translate("Dialog", "Choose Prediction file", None))
        self.set_h5_folder.setText(_translate("Dialog", "Choose h5 folder", None))

