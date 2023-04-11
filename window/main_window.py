import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QTabBar
from ui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

    def connect_event(self):
        self.tab_bar = self.tabWidget.findChild(QTabBar)
        self.tab_bar.hide()
        self.btn_home_page.clicked.connect(lambda: self.open_tab(0))
        self.btn_style_page.clicked.connect(lambda: self.open_tab(1))
        self.btn_load_img.clicked.connect(lambda: self.open_tab(2))
        self.btn_history_img.clicked.connect(lambda: self.open_tab(3))

    def open_tab(self, idx):
        self.tabWidget.setCurrentIndex(idx)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.connect_event()
    w.show()
    sys.exit(app.exec_())
