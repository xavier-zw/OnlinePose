# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(928, 694)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.img_box = QtWidgets.QLabel(self.centralwidget)
        self.img_box.setGeometry(QtCore.QRect(10, 40, 911, 621))
        self.img_box.setAutoFillBackground(True)
        self.img_box.setText("")
        self.img_box.setScaledContents(True)
        self.img_box.setObjectName("img_box")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 0, 831, 25))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.load_video = QtWidgets.QPushButton(self.widget)
        self.load_video.setObjectName("load_video")
        self.horizontalLayout.addWidget(self.load_video)
        self.open_camre = QtWidgets.QPushButton(self.widget)
        self.open_camre.setObjectName("open_camre")
        self.horizontalLayout.addWidget(self.open_camre)
        self.detect = QtWidgets.QPushButton(self.widget)
        self.detect.setObjectName("detect")
        self.horizontalLayout.addWidget(self.detect)
        self.track = QtWidgets.QPushButton(self.widget)
        self.track.setObjectName("track")
        self.horizontalLayout.addWidget(self.track)
        self.pose = QtWidgets.QPushButton(self.widget)
        self.pose.setObjectName("pose")
        self.horizontalLayout.addWidget(self.pose)
        self.action_count = QtWidgets.QPushButton(self.widget)
        self.action_count.setObjectName("action_count")
        self.horizontalLayout.addWidget(self.action_count)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 928, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_video.setText(_translate("MainWindow", "加载本地视频"))
        self.open_camre.setText(_translate("MainWindow", "打开摄像头"))
        self.detect.setText(_translate("MainWindow", "检测"))
        self.track.setText(_translate("MainWindow", "跟踪"))
        self.pose.setText(_translate("MainWindow", "姿态估计"))
        self.action_count.setText(_translate("MainWindow", "动作计数"))