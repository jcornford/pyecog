# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'converting_ndf_h5_subwindow.ui'
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

class Ui_convert_ndf_to_h5(object):
    def setupUi(self, convert_ndf_to_h5):
        convert_ndf_to_h5.setObjectName(_fromUtf8("convert_ndf_to_h5"))
        convert_ndf_to_h5.resize(661, 417)
        self.formLayoutWidget = QtGui.QWidget(convert_ndf_to_h5)
        self.formLayoutWidget.setGeometry(QtCore.QRect(70, 70, 531, 161))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout_2 = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.ndf_display = QtGui.QLineEdit(self.formLayoutWidget)
        self.ndf_display.setMinimumSize(QtCore.QSize(300, 0))
        self.ndf_display.setObjectName(_fromUtf8("ndf_display"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.ndf_display)
        self.h5_display = QtGui.QLineEdit(self.formLayoutWidget)
        self.h5_display.setMinimumSize(QtCore.QSize(300, 0))
        self.h5_display.setObjectName(_fromUtf8("h5_display"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.h5_display)
        self.select_ndf_folder = QtGui.QPushButton(self.formLayoutWidget)
        self.select_ndf_folder.setMaximumSize(QtCore.QSize(170, 16777215))
        self.select_ndf_folder.setObjectName(_fromUtf8("select_ndf_folder"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.select_ndf_folder)
        self.set_h5_folder = QtGui.QPushButton(self.formLayoutWidget)
        self.set_h5_folder.setMinimumSize(QtCore.QSize(141, 0))
        self.set_h5_folder.setMaximumSize(QtCore.QSize(200, 32))
        self.set_h5_folder.setObjectName(_fromUtf8("set_h5_folder"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.LabelRole, self.set_h5_folder)
        self.label_2 = QtGui.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_2)
        self.transmitter_ids = QtGui.QLineEdit(self.formLayoutWidget)
        self.transmitter_ids.setObjectName(_fromUtf8("transmitter_ids"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.transmitter_ids)
        self.label = QtGui.QLabel(self.formLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.LabelRole, self.label)
        self.fs_box = QtGui.QLineEdit(self.formLayoutWidget)
        self.fs_box.setObjectName(_fromUtf8("fs_box"))
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.FieldRole, self.fs_box)
        self.label_3 = QtGui.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_3)
        self.cores_to_use = QtGui.QLineEdit(self.formLayoutWidget)
        self.cores_to_use.setObjectName(_fromUtf8("cores_to_use"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.FieldRole, self.cores_to_use)
        self.progressBar = QtGui.QProgressBar(convert_ndf_to_h5)
        self.progressBar.setGeometry(QtCore.QRect(60, 342, 541, 31))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.progress_bar_label = QtGui.QLabel(convert_ndf_to_h5)
        self.progress_bar_label.setGeometry(QtCore.QRect(60, 330, 461, 16))
        self.progress_bar_label.setObjectName(_fromUtf8("progress_bar_label"))
        self.convert_button = QtGui.QPushButton(convert_ndf_to_h5)
        self.convert_button.setGeometry(QtCore.QRect(240, 250, 110, 32))
        self.convert_button.setObjectName(_fromUtf8("convert_button"))
        self.time_elapsed = QtGui.QLabel(convert_ndf_to_h5)
        self.time_elapsed.setGeometry(QtCore.QRect(60, 370, 600, 16))
        self.time_elapsed.setMinimumSize(QtCore.QSize(600, 0))
        self.time_elapsed.setObjectName(_fromUtf8("time_elapsed"))
        self.hidden_label = QtGui.QLabel(convert_ndf_to_h5)
        self.hidden_label.setGeometry(QtCore.QRect(60, 300, 600, 16))
        self.hidden_label.setMinimumSize(QtCore.QSize(600, 0))
        self.hidden_label.setText(_fromUtf8(""))
        self.hidden_label.setObjectName(_fromUtf8("hidden_label"))

        self.retranslateUi(convert_ndf_to_h5)
        QtCore.QMetaObject.connectSlotsByName(convert_ndf_to_h5)

    def retranslateUi(self, convert_ndf_to_h5):
        convert_ndf_to_h5.setWindowTitle(_translate("convert_ndf_to_h5", "Dialog", None))
        self.select_ndf_folder.setText(_translate("convert_ndf_to_h5", "Choose ndf folder", None))
        self.set_h5_folder.setText(_translate("convert_ndf_to_h5", "Choose destination h5 folder", None))
        self.label_2.setText(_translate("convert_ndf_to_h5", "Select transmitter IDs to convert", None))
        self.transmitter_ids.setText(_translate("convert_ndf_to_h5", "all", None))
        self.label.setText(_translate("convert_ndf_to_h5", "Sampling Frequency (Hz)", None))
        self.fs_box.setText(_translate("convert_ndf_to_h5", "512", None))
        self.label_3.setText(_translate("convert_ndf_to_h5", "Number of CPU cores to use", None))
        self.cores_to_use.setText(_translate("convert_ndf_to_h5", "all", None))
        self.progress_bar_label.setText(_translate("convert_ndf_to_h5", "Progress:", None))
        self.convert_button.setText(_translate("convert_ndf_to_h5", "Convert!", None))
        self.time_elapsed.setText(_translate("convert_ndf_to_h5", "Time elapsed:", None))

