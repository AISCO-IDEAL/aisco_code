import components
import cadquery as cq
from ocp_vscode import *

assy=components.CarAssembly().randomized(0.3, 0.05, 0.1)
set_port(49152)
show_object(assy.model,clear=True)