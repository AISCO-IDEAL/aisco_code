#file containing individual baseline tests


import traceback

from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.diso.helpers.storage_functions import tensors_to_mesh
from subprojects.wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp3_solar_estimator.meshing_benchmarks.analytical_sdfs import (
    CubeSDF,
    RectangularBoxSDF,
    SphereSDF,
    load_testing_model,
)


class BaseTester:
    def __init__(self):
        self.meshing_algorithms = {
            "DISO_DiffMC": DISOExtractor(extractor_type="DiffMC"),
            "DISO_DiffDMC": DISOExtractor(extractor_type="DiffDMC"),
            "FlexiCubes": FCExtractor(),
            "MeshSDF": MeshSDFExtractor(),
        }
        self.testing_models = {
            "SphereSDF": SphereSDF(radius=2000.0),
            "CubeSDF": CubeSDF(side_length=2000.0),
            "RectangularBoxSDF": RectangularBoxSDF(dimensions=(1000.0, 2000.0, 3000.0)),
            "MLP_TestModel": load_testing_model(filename="test_model.pth"),
        }

    def set_all_grids(self, bbox: list[list[int | float]], resolution: int):
        """
        if the BaseTester or its child haven't set their grids already, this can be used to set all of them to the same bbox
        """
        for extractor in self.meshing_algorithms.values():
            extractor.set_grid(bbox=bbox, resolution=resolution)

    def startup(self, bbox: list[list[int | float]] | None = None) -> bool:
        """

        Tests that all meshing algorithms (using a default cubic bounding box) can run on a simple analytical SDF (SphereSDF) without errors and produce a non-empty mesh.
        This serves as a sanity check before running more complex benchmarks.

        """
        if bbox is None:
            bbox = [[-3500.0, 3500.0], [-3500.0, 3500.0], [-3500.0, 3500.0]]  # default cubic bbox for startup test

        for name, extractor in self.meshing_algorithms.items():
            print(f"Testing {name} on SphereSDF...")
            extractor.set_grid(bbox=bbox, resolution=64)
            model = self.testing_models["SphereSDF"]
            try:
                verts, faces = extractor.extract_mesh(model)
                if verts.numel() == 0 or faces.numel() == 0:
                    print(f"Warning: {name} produced an empty mesh for SphereSDF.")
                else:
                    print(f"{name} successfully extracted a mesh with {verts.shape[0]} vertices and {faces.shape[0]} faces.")
                # Saves the mesh for visual inspection, even if empty (to check if the issue is with extraction or saving)
                tensors_to_mesh(
                    verts,
                    faces,
                    save_mesh=True,
                    filename=f"{name}_SphereSDF.ply",
                    mesh_dir="/cluster/work/fuge/AISCO/wp3_solar_estimator/benchmarks/sanity_checks",
                )
            except Exception as e:
                # Print concise error and pinpoint final raising site
                print(f"Error testing {name} on SphereSDF: {e}")
                tb = e.__traceback__
                while tb and tb.tb_next:
                    tb = tb.tb_next
                if tb is not None:
                    frame = tb.tb_frame
                    filename = frame.f_code.co_filename
                    lineno = tb.tb_lineno
                    print(f"Raised at {filename}:{lineno}")
                # Full traceback for context
                traceback.print_exc()
            print("\n")
        return True

    def stretched_geometry_test(self):
        """
        Tests whether non-cubic bounding boxes cause issues for the meshing algorithms. Uses the RectangularBoxSDF which has different dimensions along each axis.
        This is important to check since our objects often do not fit into cubic bounding boxes.

        #Rotations and nonunifom stretching are not supported yet
        """
        # Baseline parameters for this test, as to ensure resolution is comparable when distorting bboxes
        BASELINE_RES = 70  # noqa: N806
        BBOX_CUBIC = [[-3500.0, 3500.0], [-3500.0, 3500.0], [-3500.0, 3500.0]]  # noqa: N806
        UNIT_LENGTH = 7000.0  # total length of bbox along each axis for baseline cubic box, used to define stretched bboxes with same total length  # noqa: N806
        STRETCHING_FACTOR = 1.5  # Ensure this * BASELINE_RES is still an int # noqa: N806
        TRANSLATION_OFFSET = 200.0  # Translate bbox by this amount along each axis, ensure that UNIT_LENGTH/BASELINE_RES * N can eqaual this offset for comparable grid_spacing # noqa: N806
        if TRANSLATION_OFFSET % (UNIT_LENGTH / BASELINE_RES) != 0:
            print(
                f"Warning: Translation offset {TRANSLATION_OFFSET} is not a multiple of grid spacing {UNIT_LENGTH / BASELINE_RES}. Consider adjusting these values for a more meaningful test."
            )
        # Set new generators (not member variables of tester!), once testing for  translation, stretching, and both.
        # Rotation tests not supported yet

        # create copies of the meshing algorithms to modify for this test

        meshing_algorithms_baseline = dict(self.meshing_algorithms.items())
        distorted_algorithms = distorted_bboxes = {
            "Stretched": dict(self.meshing_algorithms.items()),
            "Translated": dict(self.meshing_algorithms.items()),
            "Stretched_Translated": dict(self.meshing_algorithms.items()),
            "Arbitrary_Combination": dict(self.meshing_algorithms.items()),
        }

        # Set baseline rectangular box SDF
        box_sdf = RectangularBoxSDF(dimensions=(1000.0, 2000.0, 3000.0))

        # Run base algorithms
        baseline_results = {}
        print("\nRunning baseline algorithms on RectangularBoxSDF with cubic bbox...", end="")
        for name, extractor in meshing_algorithms_baseline.items():
            extractor.set_grid(bbox=BBOX_CUBIC, resolution=BASELINE_RES)
            baseline_results[name] = extractor.extract_mesh(box_sdf)  # stores verts, faces for later comparison
            tensors_to_mesh(
                baseline_results[name][0],
                baseline_results[name][1],
                save_mesh=True,
                filename=f"{name}_RectangularBoxSDF_Baseline.ply",
                mesh_dir="/cluster/work/fuge/AISCO/wp3_solar_estimator/benchmarks/sanity_checks",
            )
        print("finished.")
        # Define non-cubic bounding boxes that fits the RectangularBoxSDF and test
        bbox_stretched = [
            [axis_min * STRETCHING_FACTOR, axis_max * STRETCHING_FACTOR] for (axis_min, axis_max) in BBOX_CUBIC
        ]  # uniformly stretched bbox

        bbox_translated = [
            [axis_min + TRANSLATION_OFFSET, axis_max + TRANSLATION_OFFSET] for (axis_min, axis_max) in BBOX_CUBIC
        ]  # translated bbox

        bbox_stretched_translated = [
            [axis_min * STRETCHING_FACTOR + TRANSLATION_OFFSET, axis_max * STRETCHING_FACTOR + TRANSLATION_OFFSET]
            for (axis_min, axis_max) in BBOX_CUBIC
        ]  # uniformly stretched and translated bbox

        bbox_arbitrary_combination = [[-1500, 1100], [-2500, 3000], [-3300, 4150]]
        distorted_bboxes = {
            "Stretched": bbox_stretched,
            "Translated": bbox_translated,
            "Stretched_Translated": bbox_stretched_translated,
            "Arbitrary_Combination": bbox_arbitrary_combination,  # Note: keeps BASELINE_RES and results in nonuniform grid spacing
        }

        # Run distorted bboxes
        print("\nRunning algorithms on RectangularBoxSDF with distorted bboxes...", end="")
        for distortion_name, bbox in distorted_bboxes.items():
            print(f"\nTesting with {distortion_name} bbox: {bbox}")
            for name, extractor in distorted_algorithms[distortion_name].items():
                distorted_res = (
                    int(BASELINE_RES * STRETCHING_FACTOR) if "Stretched" in distortion_name else BASELINE_RES
                )  # Adjust resolution for stretched bboxes to maintain comparable grid spacing

                extractor.set_grid(bbox=bbox, resolution=distorted_res)
                try:
                    verts, faces = extractor.extract_mesh(box_sdf)
                    if verts.numel() == 0 or faces.numel() == 0:
                        print(f"Warning: {name} produced an empty mesh for {distortion_name} bbox.")
                    else:
                        print(
                            f"{name} successfully extracted a mesh with {verts.shape[0]} vertices and {faces.shape[0]} faces for {distortion_name} bbox."
                        )
                        tensors_to_mesh(
                            verts,
                            faces,
                            save_mesh=True,
                            filename=f"{name}_RectangularBoxSDF_{distortion_name}.ply",
                            mesh_dir="/cluster/work/fuge/AISCO/wp3_solar_estimator/benchmarks/sanity_checks",
                        )
                except Exception as e:
                    print(f"Error testing {name} with {distortion_name} bbox: {e}")
                    traceback.print_exc()
        print("finished.")
