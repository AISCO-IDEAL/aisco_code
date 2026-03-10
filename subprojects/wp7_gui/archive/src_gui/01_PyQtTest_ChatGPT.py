#druefenacht

import sys

# import time  # noqa: ERA001
import numpy as np
import pyvista as pv
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor  # makes embedding PyVista in Qt easy


# Worker thread for long computation so GUI remains responsive
class MeshWorker(QThread):
    mesh_generation_finished = pyqtSignal(object)

    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def run(self):
        # generate a sample mesh based on params -> replace with euler calc
        x, y = np.mgrid[-10:10:100j, -10:10:100j]
        z = np.sin(self.param1 * np.sqrt(x**2 + y**2)) * self.param2
        surf = pv.StructuredGrid(x, y, z)
        self.mesh_generation_finished.emit(surf)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh Generator")

        # Create widgets
        self.slider1 = QSlider(Qt.Orientation.Horizontal)
        self.slider1.setRange(1, 10)
        self.slider1.setValue(5)
        self.slider2 = QSlider(Qt.Orientation.Horizontal)
        self.slider2.setRange(1, 5)
        self.slider2.setValue(2)
        self.start_button = QPushButton("Generate Mesh")
        self.status_label = QLabel("Adjust sliders and press Start")

        # PyVista widget
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.enable_shadows()
        # self.plotter.add_light("light kit")  # noqa: ERA001

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Param 1"))
        layout.addWidget(self.slider1)
        layout.addWidget(QLabel("Param 2"))
        layout.addWidget(self.slider2)
        layout.addWidget(self.start_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.plotter.interactor)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Action Listeners
        self.start_button.clicked.connect(self.let_euler_calculate)
        self.let_euler_calculate() # start by default

    def let_euler_calculate(self):
        p1 = self.slider1.value()
        p2 = self.slider2.value()
        self.status_label.setText("Generating mesh...")
        self.worker = MeshWorker(p1, p2)
        self.worker.mesh_generation_finished.connect(self.display_mesh)
        self.worker.start()

    def display_mesh(self, mesh):
        self.plotter.clear()

        # Convert to PolyData for smooth shading
        if isinstance(mesh, pv.StructuredGrid):
            mesh = mesh.extract_surface()

        # Enable renderer shadow support
        self.plotter.enable_shadows()

        # Remove default lights (optional)
        self.plotter.remove_all_lights()

        # Create a directional sunlight
        sunlight = pv.Light(
            light_type='headlight',       # makes it behave like the sun
            # direction=(-1, -1, -0.8),       # incoming light direction  # noqa: ERA001
            color='white',
            intensity=0.9,
            positional=False                # ensures uniform illumination
        )
        self.plotter.add_light(sunlight)

        # Optional: subtle ambient light to soften shadows
        ambient_light = pv.Light(
            light_type='scenelight',
            color='white',
            intensity=0.3
        )
        self.plotter.add_light(ambient_light)

        # Add shaded mesh
        self.plotter.add_mesh(
            mesh,
            cmap="coolwarm",
            smooth_shading=True,
            lighting=True,
            specular=0.4,
            specular_power=20,
            show_edges=False,
        )

        # Adjust camera and refresh
        self.plotter.reset_camera()
        self.status_label.setText("Mesh ready!")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
