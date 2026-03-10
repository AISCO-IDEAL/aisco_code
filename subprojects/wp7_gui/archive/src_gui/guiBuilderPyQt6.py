import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.uic import loadUi


class MainUI(QMainWindow):
    pushButton: QPushButton  # noqa: N815

    def __init__(self):
        super(MainUI, self).__init__()  # noqa: UP008
        ui_path = Path(__file__).parent / "basicGUI.ui"
        loadUi(ui_path, self)
        self.pushButton.clicked.connect(self.clickhandler)

    def clickhandler(self):
        print("Hello World")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    app.exec()
