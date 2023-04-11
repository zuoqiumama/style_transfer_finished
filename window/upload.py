# -*- coding: utf-8 -*-
import socket

# Form implementation generated from reading ui file 'upload.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import qrcode
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib import pyplot as plt


class Upload(QWidget):
    def setupUi(self, Form):
        """
        init current host ip address
        """
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        self.url = "https://" + ip_address + ":5000"
        print(f"[upload page] qrcode url = {self.url}")

        Form.setObjectName("Form")
        Form.resize(824, 632)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(10, 10, 1121, 853))
        self.widget.setObjectName("widget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        """
        upload web page qrcode
        """
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setMinimumSize(QtCore.QSize(450, 450))
        self.widget_2.setObjectName("widget_2")
        self.widget_2.setStyleSheet("background: transparent")
        self.horizontalLayout.addWidget(self.widget_2)
        qr_path = self.generate_qrcode(self.url)

        self.qr_label = QtWidgets.QLabel(self.widget_2)
        self.qr_label.setMinimumSize(300, 300)
        self.qr_label.setPixmap(QPixmap(qr_path))
        """
        info help
        """
        self.widget_3 = QtWidgets.QWidget(self.widget)
        self.widget_3.setMinimumSize(QtCore.QSize(231, 491))
        self.widget_3.setObjectName("widget_3")
        self.widget_3.setStyleSheet("background-color: rgb(50, 100, 125)")
        self.horizontalLayout.addWidget(self.widget_3)

        self.layout = QVBoxLayout()
        self.widget_3.setLayout(self.layout)
        self.title_label = QLabel("<h1 style='text-align:center; font-weight:bold;'>帮助</h1>")
        self.layout.addWidget(self.title_label)

        self.text_label = QLabel(
            "<p style='text-align:justify;font-size:24px;'>欢迎使用图像风格迁移系统。该系统可以将一个输入图片的风格转化为另一个图片的风格。使用该系统的步骤如下：<br>1. 在上传图片页面扫描二维码进入网页进行图片上传；<br>2. 在图像风格化页面选择上传的图片；<br>3. 如果选择自定义图像风格迁移，则需要选择一张代表风格的图片；<br>4. 如果选择现成模型，则需要勾选想要使用的模型；<br>5. 点击“开始转化”按钮进行图像风格迁移。</p>")
        self.text_label.setWordWrap(True)
        self.layout.addWidget(self.text_label)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

    def generate_qrcode(self, url):
        res_path = 'webpage.png'
        import os
        if os.path.exists(res_path):
            os.remove(res_path)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=15,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        plt.imsave(res_path, img)
        return res_path
