from pathlib import Path

import yaml


def get_outer_bb_info()-> dict:
    """
    Gets info about the outer bb that is relevant for WP4 from assembly regulations version 4
    Reads the regulations_and_parameters.yaml file and returns the relevant info as a dictionary.
    Returns:
        dict: A dictionary containing the width, length, height, and centroid (float list) of the outer bb.
        Keys: "width", "length", "height", "centroid"


    """
    config_path = Path(__file__).parent.parent / "config" / "regulations_and_parameters.yaml"

    with config_path.open() as file:
        data = yaml.safe_load(file)

    outer_bb_dims=data["no_build_zones"]["outer_prism"]
    return_dict={}
    return_dict["width"]=outer_bb_dims["width"]
    return_dict["length"]=outer_bb_dims["length"]
    return_dict["height"]=outer_bb_dims["height"]
    return_dict["centroid"] = [
        0.0,  # x
        0.0,  # y
        outer_bb_dims["height"] / 2.0,  # z
    ]  # Assuming the centroid is at the center of the base and at half the height
    return return_dict
    # Get the relevant info for the outer bb
