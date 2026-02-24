import os
import re

import pandas as pd


def process_ahmed_body_dataset(root_dir):
    # Standard constants from the methodology
    rho = 1.225
    v = 38.889
    dynamic_pressure = 0.5 * rho * (v**2)

    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return

    for folder_name in sorted(os.listdir(root_dir)):
        if folder_name.startswith('run_'):
            run_id = re.findall(r'\d+', folder_name)[0]
            run_path = os.path.join(root_dir, folder_name)

            # File paths for Ahmed Body
            geo_param_path = os.path.join(run_path, f'geo_parameters_{run_id}.csv')
            force_mom_path = os.path.join(run_path, f'force_mom_varref_{run_id}.csv')
            # Saving back into the run folder now that permissions are fixed
            output_path = os.path.join(run_path, f'drag_param_{run_id}.csv')

            if os.path.exists(geo_param_path) and os.path.exists(force_mom_path):
                try:
                    # 1. Calculate Frontal Area from mm to m^2
                    # Area = (width * height) / 1,000,000
                    df_geo = pd.read_csv(geo_param_path)
                    width = df_geo['body-width'].iloc[0]
                    height = df_geo['body-height'].iloc[0]
                    a_ref = (width * height) / 1000000.0

                    # 2. Get lowercase 'cd' from Ahmed Body file
                    df_force = pd.read_csv(force_mom_path)
                    cd = df_force['cd'].iloc[0]

                    # 3. Calculate Drag Force (Fd = q * A * cd)
                    drag_force_n = dynamic_pressure * a_ref * cd

                    # 4. Create and save the new parameters file
                    df_output = pd.DataFrame({
                        'cd': [cd],
                        'Drag_Force_N': [drag_force_n],
                        'Frontal_Area_m2': [a_ref]
                    })
                    df_output.to_csv(output_path, index=False)

                except Exception as e:
                    print(f"Error in {folder_name}: {e}")

if __name__ == "__main__":
    # Adjust to your absolute path
    dataset_path = '/cluster/work/fuge/AISCO/wp1_cfd_ai/ahmed_complete'
    process_ahmed_body_dataset(dataset_path)
    print("Process complete. drag_param_ID.csv files created in each run folder.")
