import sys
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

loader = QUiLoader() #Set up a loader object

app = QtWidgets.QApplication(sys.argv)

ui_path = Path(__file__).parent / "basicGUI.ui"
window = loader.load(ui_path, None) #Load the
def do_something():
    print("Button clicked!")
window.pushButton.clicked.connect(do_something)
window.show()
app.exec()
