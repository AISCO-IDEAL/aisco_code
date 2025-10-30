from crude_estimator import calculateCdProxy

print(
    calculateCdProxy(
        "silvretta_big_rot.stl",
        True,
        "silvretta_big_rot_viz.glb",
        slicing_direction="-",
        slicing_axis="z",
    )
)
