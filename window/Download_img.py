"""
@FileName：Download_img.py
@Description：once user click result image,show this widget and give a qr code let user download the result image
@Author：zuoqiumama
@Time：2023/4/7 19:49
"""
import qrcode
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout


class QRCodeWidget(QWidget):
    def __init__(self, url):
        super().__init__()

        # Generate the QR code image
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")

        # Convert the QR code image to QPixmap for display in a QLabel
        qr_pixmap = QPixmap.fromImage(qr_image)

        # Create the QLabel and set its pixmap
        qr_label = QLabel(self)
        qr_label.setPixmap(qr_pixmap)

        # Create the main layout and add the QR label to it
        layout = QVBoxLayout()
        layout.addWidget(qr_label)

        # Set the main layout for the widget
        self.setLayout(layout)
