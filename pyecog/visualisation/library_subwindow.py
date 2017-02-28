# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'library_window.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

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

class Ui_LibraryManagement(object):
    def setupUi(self, LibraryManagement):
        LibraryManagement.setObjectName(_fromUtf8("LibraryManagement"))
        LibraryManagement.resize(881, 557)
        self.formLayoutWidget = QtGui.QWidget(LibraryManagement)
        self.formLayoutWidget.setGeometry(QtCore.QRect(40, 50, 748, 80))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.annotations_display = QtGui.QLineEdit(self.formLayoutWidget)
        self.annotations_display.setMinimumSize(QtCore.QSize(500, 0))
        self.annotations_display.setDragEnabled(True)
        self.annotations_display.setReadOnly(True)
        self.annotations_display.setObjectName(_fromUtf8("annotations_display"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.annotations_display)
        self.select_annotations = QtGui.QPushButton(self.formLayoutWidget)
        self.select_annotations.setMinimumSize(QtCore.QSize(200, 0))
        self.select_annotations.setObjectName(_fromUtf8("select_annotations"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.select_annotations)
        self.select_h5_folder = QtGui.QPushButton(self.formLayoutWidget)
        self.select_h5_folder.setMinimumSize(QtCore.QSize(200, 0))
        self.select_h5_folder.setObjectName(_fromUtf8("select_h5_folder"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.select_h5_folder)
        self.h5_folder_display = QtGui.QLineEdit(self.formLayoutWidget)
        self.h5_folder_display.setMinimumSize(QtCore.QSize(500, 0))
        self.h5_folder_display.setObjectName(_fromUtf8("h5_folder_display"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.h5_folder_display)
        self.new_library = QtGui.QPushButton(LibraryManagement)
        self.new_library.setGeometry(QtCore.QRect(40, 140, 110, 32))
        self.new_library.setObjectName(_fromUtf8("new_library"))
        self.add_to_library = QtGui.QPushButton(LibraryManagement)
        self.add_to_library.setGeometry(QtCore.QRect(150, 140, 110, 32))
        self.add_to_library.setObjectName(_fromUtf8("add_to_library"))
        self.library_path_display = QtGui.QLineEdit(LibraryManagement)
        self.library_path_display.setGeometry(QtCore.QRect(40, 250, 741, 21))
        self.library_path_display.setMinimumSize(QtCore.QSize(500, 0))
        self.library_path_display.setReadOnly(True)
        self.library_path_display.setObjectName(_fromUtf8("library_path_display"))
        self.label = QtGui.QLabel(LibraryManagement)
        self.label.setGeometry(QtCore.QRect(40, 230, 81, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.progressBar = QtGui.QProgressBar(LibraryManagement)
        self.progressBar.setGeometry(QtCore.QRect(40, 480, 751, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.progressBar_label_below = QtGui.QLabel(LibraryManagement)
        self.progressBar_label_below.setGeometry(QtCore.QRect(40, 510, 751, 16))
        self.progressBar_label_below.setText(_fromUtf8(""))
        self.progressBar_label_below.setObjectName(_fromUtf8("progressBar_label_below"))
        self.progressBar_lable_above1 = QtGui.QLabel(LibraryManagement)
        self.progressBar_lable_above1.setGeometry(QtCore.QRect(40, 460, 741, 16))
        self.progressBar_lable_above1.setObjectName(_fromUtf8("progressBar_lable_above1"))
        self.progressBar_label_above2 = QtGui.QLabel(LibraryManagement)
        self.progressBar_label_above2.setGeometry(QtCore.QRect(40, 440, 751, 16))
        self.progressBar_label_above2.setText(_fromUtf8(""))
        self.progressBar_label_above2.setObjectName(_fromUtf8("progressBar_label_above2"))
        self.set_library = QtGui.QPushButton(LibraryManagement)
        self.set_library.setGeometry(QtCore.QRect(40, 280, 131, 32))
        self.set_library.setObjectName(_fromUtf8("set_library"))
        self.clear_library = QtGui.QPushButton(LibraryManagement)
        self.clear_library.setGeometry(QtCore.QRect(170, 280, 131, 32))
        self.clear_library.setObjectName(_fromUtf8("clear_library"))
        self.line_2 = QtGui.QFrame(LibraryManagement)
        self.line_2.setGeometry(QtCore.QRect(40, 210, 761, 16))
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.line_3 = QtGui.QFrame(LibraryManagement)
        self.line_3.setGeometry(QtCore.QRect(50, 420, 751, 16))
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.gridLayoutWidget = QtGui.QWidget(LibraryManagement)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(100, 330, 141, 51))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.chunk_length = QtGui.QLineEdit(self.gridLayoutWidget)
        self.chunk_length.setObjectName(_fromUtf8("chunk_length"))
        self.gridLayout.addWidget(self.chunk_length, 0, 2, 1, 1)
        self.label_2 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 3, 1, 1)
        self.add_labels = QtGui.QPushButton(LibraryManagement)
        self.add_labels.setGeometry(QtCore.QRect(260, 320, 171, 32))
        self.add_labels.setObjectName(_fromUtf8("add_labels"))
        self.label_4 = QtGui.QLabel(LibraryManagement)
        self.label_4.setGeometry(QtCore.QRect(450, 320, 361, 31))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.add_features = QtGui.QPushButton(LibraryManagement)
        self.add_features.setGeometry(QtCore.QRect(260, 350, 171, 32))
        self.add_features.setObjectName(_fromUtf8("add_features"))
        self.label_5 = QtGui.QLabel(LibraryManagement)
        self.label_5.setGeometry(QtCore.QRect(380, 10, 121, 31))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica Neue"))
        font.setPointSize(19)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setIndent(-1)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_6 = QtGui.QLabel(LibraryManagement)
        self.label_6.setGeometry(QtCore.QRect(320, 180, 241, 31))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica Neue"))
        font.setPointSize(19)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setIndent(-1)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_7 = QtGui.QLabel(LibraryManagement)
        self.label_7.setGeometry(QtCore.QRect(450, 350, 351, 31))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.overwrite_box = QtGui.QCheckBox(LibraryManagement)
        self.overwrite_box.setGeometry(QtCore.QRect(50, 400, 341, 18))
        self.overwrite_box.setObjectName(_fromUtf8("overwrite_box"))
        self.use_peaks = QtGui.QCheckBox(LibraryManagement)
        self.use_peaks.setGeometry(QtCore.QRect(440, 400, 341, 18))
        self.use_peaks.setObjectName(_fromUtf8("use_peaks"))
        self.formLayoutWidget_2 = QtGui.QWidget(LibraryManagement)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(630, 140, 160, 51))
        self.formLayoutWidget_2.setObjectName(_fromUtf8("formLayoutWidget_2"))
        self.formLayout_2 = QtGui.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.label_8 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_8)
        self.fs_box = QtGui.QLineEdit(self.formLayoutWidget_2)
        self.fs_box.setMaximumSize(QtCore.QSize(50, 16777215))
        self.fs_box.setObjectName(_fromUtf8("fs_box"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.fs_box)
        self.formLayoutWidget.raise_()
        self.select_annotations.raise_()
        self.new_library.raise_()
        self.add_to_library.raise_()
        self.library_path_display.raise_()
        self.label.raise_()
        self.progressBar.raise_()
        self.progressBar_label_below.raise_()
        self.progressBar_lable_above1.raise_()
        self.progressBar_label_above2.raise_()
        self.set_library.raise_()
        self.clear_library.raise_()
        self.line_2.raise_()
        self.line_3.raise_()
        self.gridLayoutWidget.raise_()
        self.add_labels.raise_()
        self.label_4.raise_()
        self.add_features.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.overwrite_box.raise_()
        self.use_peaks.raise_()
        self.formLayoutWidget_2.raise_()

        self.retranslateUi(LibraryManagement)
        QtCore.QMetaObject.connectSlotsByName(LibraryManagement)

    def retranslateUi(self, LibraryManagement):
        LibraryManagement.setWindowTitle(_translate("LibraryManagement", "Dialog", None))
        self.select_annotations.setText(_translate("LibraryManagement", "Select Annotations", None))
        self.select_h5_folder.setText(_translate("LibraryManagement", "Select h5 folder", None))
        self.new_library.setText(_translate("LibraryManagement", "New Library", None))
        self.add_to_library.setText(_translate("LibraryManagement", "Add to library", None))
        self.label.setText(_translate("LibraryManagement", "Library Path:", None))
        self.progressBar_lable_above1.setText(_translate("LibraryManagement", "Progress", None))
        self.set_library.setText(_translate("LibraryManagement", "Set Library", None))
        self.clear_library.setText(_translate("LibraryManagement", "Clear Library path ", None))
        self.chunk_length.setText(_translate("LibraryManagement", "5", None))
        self.label_2.setText(_translate("LibraryManagement", "Chunk length", None))
        self.label_3.setText(_translate("LibraryManagement", "s", None))
        self.add_labels.setText(_translate("LibraryManagement", "Add labels to library", None))
        self.label_4.setText(_translate("LibraryManagement", "Quickly add labels for looking at how it chunks up in main gui", None))
        self.add_features.setText(_translate("LibraryManagement", "Add features to library", None))
        self.label_5.setText(_translate("LibraryManagement", "Make Library", None))
        self.label_6.setText(_translate("LibraryManagement", "Add Features and Labels", None))
        self.label_7.setText(_translate("LibraryManagement", "Warning, this will be slow! Labels added automatically.", None))
        self.overwrite_box.setText(_translate("LibraryManagement", "Overwrite previously calculated features in library", None))
        self.use_peaks.setText(_translate("LibraryManagement", "Use peaks and valleys features, slower!", None))
        self.label_8.setText(_translate("LibraryManagement", "Fs (Hz)", None))
        self.fs_box.setText(_translate("LibraryManagement", "512", None))
