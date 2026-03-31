import os
import sys
from pathlib import Path

# Add the src directory to sys.path
src_path = Path("/cluster/home/eschaerer/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/src")
sys.path.append(str(src_path))

try:
    from subprojects.wp4_aeroshell_optimizer import load_config

    # Check what load_config returns
    config = load_config(v="v1", cfg_name="general_config.yaml")
    run_locally = config.get("general_info", {}).get("run_locally")

    print(f"Detected run_locally: {run_locally}")

    # Check env vars
    is_cluster = (
        "SLURM_CLUSTER_NAME" in os.environ or "SLURM_JOB_ID" in os.environ or os.environ.get("HOSTNAME", "").startswith("eu-")
    )
    print(f"Is Cluster Environment: {is_cluster}")

    expected = "False" if is_cluster else "True"
    if run_locally == expected:
        print("SUCCESS: Detection working correctly.")
    else:
        print(f"FAILURE: Expected {expected}, got {run_locally}")

except Exception as e:
    print(f"Error: {e}")
