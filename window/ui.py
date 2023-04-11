# -*- coding: utf-8 -*-
import sys

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QFile, QTextStream, QPropertyAnimation
from PyQt5.QtGui import QMouseEvent

from main import StyleTransferTab
from window.history_img import ImageListWidget
from window.home import ArtExhibition
from window.style_page import StylePage
from window.upload import Upload


def read_qss(style):
    with open(style, 'r') as f:
        return f.read()

class Ui_MainWindow(object):
    """
    @ClassName：Ui_MainWindow
    @Description：init main window ui
    @Author：zuoqiumama
    @ToDo: modify btn, tab and title widget
    """
    _startPos = None
    _endPos = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setWindowFlags(Qt.FramelessWindowHint)
        MainWindow.setAttribute(Qt.WA_TranslucentBackground)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.widget.setStyleSheet(
            """
            background:rgb(85, 170, 0);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            padding-top:5px;
            """.format(30)
        )

        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")

        """
        tile init
        """
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setMinimumSize(QtCore.QSize(764, 50))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 50))
        self.widget_2.setContentsMargins(9, 9, 9, 9)
        self.widget_2.setStyleSheet("background-color: rgb(85, 170, 0);")
        self.gridLayout.addWidget(self.widget_2, 0, 0, 1, 2)

        self.gridLayout_ = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout.setObjectName("gridLayout")

        """
        @Todo: change the tile, icon and button style sheet
        """
        self.label_icon = QtWidgets.QLabel(self.widget_2)
        self.label_icon.setMinimumSize(QtCore.QSize(32, 32))
        self.gridLayout_.addWidget(self.label_icon, 0, 0)
        self.label_icon.setStyleSheet(
            read_qss("E:\毕设\style_transfer\window\qss\\btn_qss\label_icon.qss")
        )

        self.label_title = QtWidgets.QLabel(self.widget_2)
        self.label_title.setMinimumSize(QtCore.QSize(300, 25))
        self.gridLayout_.addWidget(self.label_title, 0, 1)

        self.temp = QtWidgets.QWidget(self.widget_2)
        self.temp.setMinimumSize(QtCore.QSize(950, 45))
        self.temp.setMaximumSize(QtCore.QSize(16777215, 45))
        self.gridLayout_.addWidget(self.temp, 0, 2)

        self.btn_close = QtWidgets.QPushButton(self.widget_2)
        self.btn_close.setMinimumSize(QtCore.QSize(25, 25))
        self.btn_close.setObjectName("btn_close")
        self.btn_close.setStyleSheet(
            read_qss("E:\毕设\style_transfer\window\qss\\btn_qss\\btn_close_qss.qss")
        )
        self.btn_close.clicked.connect(lambda: self.close_window(MainWindow))
        self.gridLayout_.addWidget(self.btn_close, 0, 4)

        self.btn_min = QtWidgets.QPushButton(self.widget_2)
        self.btn_min.setMinimumSize(QtCore.QSize(25, 25))
        self.btn_min.setStyleSheet(
            read_qss("E:\毕设\style_transfer\window\qss\\btn_qss\\btn_min_qss.qss")
        )
        self.btn_min.clicked.connect(lambda: self.min_window(MainWindow))
        self.gridLayout_.addWidget(self.btn_min, 0, 3)

        self.widget_2.setLayout(self.gridLayout_)

        """
        navigation whose background color should not match the MainWindow's
        """
        self.widget_3 = QtWidgets.QWidget(self.widget)
        self.widget_3.setMinimumSize(QtCore.QSize(200, 466 * 2))
        self.widget_3.setMaximumSize(QtCore.QSize(400, 16777215))
        self.widget_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.widget_3.setObjectName("widget_3")

        self.scrollArea = QtWidgets.QScrollArea(self.widget_3)
        self.scrollArea.setGeometry(QtCore.QRect(10, 10, 121, 441))
        self.scrollArea.setMinimumSize(QtCore.QSize(150, 441*2))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 119, 439))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        # button init
        self.button_init()

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.widget_3, 1, 0, 1, 1)

        """
        main window whose background color should not match the MainWindow's
        """
        self.widget_5 = QtWidgets.QWidget(self.widget)
        self.widget_5.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.widget_5.setObjectName("widget_5")
        self.widget_5.setMinimumSize(QtCore.QSize(1200, 800))

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        # tab init
        self.tabWidget = QtWidgets.QTabWidget(self.widget_5)
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.tab_init()

        """
        others
        """
        self.verticalLayout_2.addWidget(self.tabWidget)
        self.gridLayout.addWidget(self.widget_5, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.widget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def button_init(self):
        """
        @ToDo: modify every button style sheet
        """
        self.btn_home_page = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.btn_home_page.setMinimumSize(QtCore.QSize(101, 51))
        self.btn_home_page.setObjectName("pushButton")
        self.verticalLayout_3.addWidget(self.btn_home_page)

        self.btn_style_page = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.btn_style_page.setMinimumSize(QtCore.QSize(101, 51))
        self.btn_style_page.setObjectName("pushButton_2")
        self.verticalLayout_3.addWidget(self.btn_style_page)

        self.btn_load_img = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.btn_load_img.setMinimumSize(QtCore.QSize(101, 51))
        self.btn_load_img.setObjectName("pushButton_3")
        self.verticalLayout_3.addWidget(self.btn_load_img)

        self.btn_history_img = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.btn_history_img.setMinimumSize(QtCore.QSize(101, 51))
        self.btn_history_img.setObjectName("pushButton_4")
        self.verticalLayout_3.addWidget(self.btn_history_img)


    def tab_init(self):
        """
        @ToDo: change every tab with customize widget
        """
        self.tab_home= QtWidgets.QWidget()
        self.tab_home.setObjectName("tab")
        self.home = ArtExhibition()
        self.home.setupUi(self.tab_home)
        self.tabWidget.addTab(self.tab_home, "")

        self.tab_style = QtWidgets.QWidget()
        self.tab_style.setObjectName("tab_2")
        self.style_page = StylePage()
        self.style_page.setupUi(self.tab_style)
        self.tabWidget.addTab(self.tab_style, "")

        self.tab_upload = QtWidgets.QWidget()
        self.tab_upload.setObjectName("tab_3")
        self.qr_wid = Upload()
        self.qr_wid.setupUi(self.tab_upload)
        self.tabWidget.addTab(self.tab_upload, "")

        self.tab_history = QtWidgets.QWidget()
        self.tab_history.setObjectName("tab_4")
        self.history_page = ImageListWidget(self.tab_history)
        self.history_page.setupUi(self.tab_history)
        self.tabWidget.addTab(self.tab_history, "")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_home_page.setText(_translate("MainWindow", "主页"))
        self.btn_style_page.setText(_translate("MainWindow", "照片艺术化"))
        self.btn_load_img.setText(_translate("MainWindow", "上传图片"))
        self.btn_history_img.setText(_translate("MainWindow", "历史图片"))
        self.btn_close.setText(_translate("MainWindow", ""))
        self.btn_min.setText(_translate("MainWindow", ""))
        self.label_icon.setText(_translate("MainWindow", ""))
        self.label_title.setText(_translate("MainWindow", "style transfer "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_home), _translate("MainWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_upload), _translate("MainWindow", "页"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_history), _translate("MainWindow", "页"))




    """
    window_moving mode
    """
    # 鼠标按下记录开始位置
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())

    # 持续更新鼠标位置并移动窗口
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._startPos is None:
            return
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)

    # 鼠标释放就取消拖动
    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = None
            self._endPos = None

    def close_window(self, window):
        """
        @Description：close the window
        @ToDo: finished
        """
        window.close()


    def min_window(self, window):
        """
        @Description：minimum the window
        @ToDo: finished
        """
        window.showMinimized()





