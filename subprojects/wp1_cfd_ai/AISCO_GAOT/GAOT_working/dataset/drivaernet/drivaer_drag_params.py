import os
import re

import pandas as pd


def process_drivaer_dataset(root_dir):
    # Physical constants from the paper
    # Velocity (v) = 38.889 m/s (approx 140 km/h)
    # Density (rho) = 1.225 kg/m^3
    rho = 1.225
    v = 38.889
    dynamic_pressure = 0.5 * rho * (v**2)  # Approx 926.3 Pa with this v

    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return

    for folder_name in sorted(os.listdir(root_dir)):
        if folder_name.startswith("run_"):
            run_id = re.findall(r"\d+", folder_name)[0]
            run_path = os.path.join(root_dir, folder_name)

            geo_ref_path = os.path.join(run_path, f"geo_ref_{run_id}.csv")
            force_mom_path = os.path.join(run_path, f"force_mom_{run_id}.csv")
            # NEW: Path for the new output file
            output_path = os.path.join(run_path, f"drag_param_{run_id}.csv")

            if os.path.exists(geo_ref_path) and os.path.exists(force_mom_path):
                try:
                    # 1. Get Local Frontal Area (aRef)
                    df_geo = pd.read_csv(geo_ref_path)
                    a_ref = df_geo["aRef"].iloc[0]

                    # 2. Get Local Drag Coefficient (Cd)
                    df_force = pd.read_csv(force_mom_path)
                    cd = df_force["Cd"].iloc[0]

                    # 3. Calculate Drag Force (Fd = q * A * Cd)
                    drag_force_n = dynamic_pressure * a_ref * cd

                    # 4. Create new DataFrame with only the requested info
                    df_output = pd.DataFrame({"Cd": [cd], "Drag_Force_N": [drag_force_n]})

                    # 5. Save to the new filename instead of overwriting
                    df_output.to_csv(output_path, index=False)

                except Exception as e:
                    print(f"Error in {folder_name}: {e}")


if __name__ == "__main__":
    # Absolute path on the cluster
    dataset_path = "/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete"
    process_drivaer_dataset(dataset_path)
    print("Force calculation complete. Created drag_param_ID.csv files.")
