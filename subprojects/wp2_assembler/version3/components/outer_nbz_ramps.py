from subprojects.wp2_assembler.version3.components.nbz import call_assembly
import cadquery as cq

def outer_nbz_ramps(assembly_id):
    from ocp_vscode import show
    inner, outer = call_assembly(assembly_id)

    wheel_1 = inner["wheel_1"]
    print(type(wheel_1))
    wheel_1_compound = wheel_1.toCompound()
    wheel_1_bounding_box = wheel_1_compound.BoundingBox()
    print(wheel_1_bounding_box.xmin)
    
    print(inner["wheel_1"].loc.toTuple())
    result = ( 
        cq.Workplane("XZ")
        #.moveTo(inner.loc.toTuple()[0][0] + wheel_1_bounding_box.xmin, inner.loc.toTuple()[0][2])
        .polarLine(1000, 45)
        .lineTo(1500, 0) # End the shape with a line back to the x-axis
        .lineTo(0,0)
        .close()
    )
    show([result,inner])



if __name__ == "__main__":
    demo_id = 30
    outer_nbz_ramps(demo_id)
