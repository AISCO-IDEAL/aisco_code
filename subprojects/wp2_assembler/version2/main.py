import cadquery as cq
from ocp_vscode import show
from utils.reg_imp import get_regulation

# testing regulation import
wheelDim = get_regulation("wheels.tire_radius")
print(wheelDim)

print(type(wheelDim))
