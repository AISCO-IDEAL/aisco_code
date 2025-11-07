from crude_estimator import calculateCdProxy

print(
    calculateCdProxy(
        "SilvrettaAeroskin.stl",
        True,
        "SilvrettaAeroskin_viz.glb",
        slicing_direction="-",
        slicing_axis="x",
    )
)
