"""
@FileName：history_img.py
@Description：show a list of history generated image and user can click one image to download it
@Author：zuoqiumama
@Time：2023/4/6 14:49
"""
import os

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, \
    QVBoxLayout, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image


class ImageListWidget(QWidget):
    def load_images(self):
        self.clear_layout(self.image_layout)
        self.image_list = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path) if
                           f.endswith('.jpg') or f.endswith('.png')]

        row = 0
        col = 0
        for image in self.image_list:
            pixmap = QPixmap(image)
            pixmap = pixmap.scaledToHeight(150, Qt.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pixmap)
            label.mousePressEvent = self._create_click_handler(image)
            self.image_layout.addWidget(label, row, col)
            col += 1
            if col == 5:  # 每行最多显示5张图片
                col = 0
                row += 1

        # 调整可滚动区域大小
        self.scrollAreaWidgetContents.adjustSize()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

    def _create_click_handler(self, image):
        def handler(event):
            if event.button() == Qt.LeftButton and event.type() == QEvent.MouseButtonDblClick:
                self.show_img(image)

        return handler

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def show_img(self, img_path):
        print(f"[history image] {img_path} have been clicked")
        Image.open(img_path).show()

    def setupUi(self, Form):
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(10, 10, 1121, 853))
        self.widget.setObjectName("widget")

        self.image_path = '../web_window/static/res_img'
        self.image_list = []

        self.scrollArea = QtWidgets.QScrollArea(self.widget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 10, 1000, 800))
        self.scrollArea.setMinimumSize(QtCore.QSize(1000, 800))
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1000, 800))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.image_layout = QGridLayout(self.scrollAreaWidgetContents)
        self.image_layout.setHorizontalSpacing(20)  # 设置横向间距
        self.image_layout.setVerticalSpacing(20)  # 设置纵向间距
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.image_layout)
        self.load_images()

        self.widget.setLayout(self.main_layout)

"""
import os

import PIL
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, \
    QVBoxLayout, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget
from PIL import Image


class ImageListWidget(QWidget):
    def load_images(self):
        self.clear_layout(self.image_layout)
        self.image_list = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path) if
                           f.endswith('.jpg') or f.endswith('.png')]

        row = 0
        col = 0
        for image in self.image_list:
            pixmap = QPixmap(image)
            pixmap = pixmap.scaledToHeight(150, Qt.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pixmap)
            label.mousePressEvent = self._create_click_handler(image)
            self.image_layout.addWidget(label, row, col)
            col += 1
            if col == 5:  # 每行最多显示5张图片
                col = 0
                row += 1

    def _create_click_handler(self, image):
        def handler(event):
            if event.button() == Qt.LeftButton and event.type() == QEvent.MouseButtonDblClick:
                self.show_img(image)

        return handler

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def show_img(self, img_path):
        print(f"[history image] {img_path} have been clicked")
        Image.open(img_path).show()

    def setupUi(self, Form):
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(10, 10, 1121, 853))
        self.widget.setObjectName("widget")

        self.image_path = '../web_window/static/res_img'
        self.image_list = []

        self.scrollArea = QtWidgets.QScrollArea(self.widget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 10, 1000, 800))
        self.scrollArea.setMinimumSize(QtCore.QSize(1000, 800))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1000, 800))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollAreaWidgetContents.setMinimumHeight(len(self.image_list) // 5 * 170)
        self.image_layout = QGridLayout()
        self.image_layout.setHorizontalSpacing(20)  # 设置横向间距
        self.image_layout.setVerticalSpacing(20)  # 设置纵向间距
        self.main_layout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.main_layout.addLayout(self.image_layout)
        self.load_images()


        self.widget.setLayout(self.main_layout)
"""
