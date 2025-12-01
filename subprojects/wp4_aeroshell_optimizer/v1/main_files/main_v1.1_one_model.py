import importlib
import torch
from pathlib import Path
from subprojects.wp4_aeroshell_optimizer.v1.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_general_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import allround_save_data

# we want to dynamically run an entire optimization process for the WIRE model.
# we need the model, config and (losses)
# we need paths to store model (aka the sdf) in the end

# use the train config file


def fetch_pretrained_model(model: BaseModel,
                        pretrained_path: str = "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_dataset",
                        verbose: bool = True) -> None | Path:
    """
    Check whether a pretrained model exists for the given model architecture in the pretrained_path.
    Returns the path to the pretrained model if found, else None.

    """
    if pretrained_path is None:
        print("No valid pretrained path provided, skipping pretrained model fetch.")
        return None
    
    model_hash = dict_hashing(model.get_model_info_dict())
    found = False
    file = None
    # Check whether there is a directory containing the hash in its name and whether theres a corresponding .pth file (also with hash in name)
    for d in Path(pretrained_path).glob(f"*{model_hash}*"):
        if d.is_dir():
            for f in d.glob(f"*{model_hash}*.pth"):
                file = f
                found = True
                break
        if found:
            break

                
    if verbose and found:
        print(f"Pretrained model for {model_hash} found: {file}")
    elif verbose and not found:
        print(f"No pretrained model found for {model_hash}.")

    if not found:
        return None
    else:
        return file
            
   
    


def main(verbose: bool = True) -> None:
    
    # Get general config

    general_config = load_general_config(v="v1", cfg_name="general_config.yaml")
    model_config = load_general_config(v="v1", cfg_name=general_config["active_configs"]["model_params"])
    if verbose:
        print("Successfully loaded general and model config")

    
    # Get locations of model definition
    model_name = general_config["model_def"]["model_name"]
    model_definition = general_config["model_def"]["model_file_path"]

    # Import model and get <class 'type'>
    ModelClass = dynamic_type_loading(file_path= model_definition, class_name = model_name)

    # Initialise model with config,important that all keys match to a keyword argument in the model __init__()
    model = ModelClass(**model_config) 
    if verbose:
        print(f"Successfully initialised model: {model_name}")

    # Move model to device
    device = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Check if there is a pretrained model already
    # Note that for now this only includes pretrained spheres

    pretrained_model = fetch_pretrained_model(model,
                                              pretrained_path=general_config["storage_paths"].get("pretrained_dir", None),
                                              verbose=verbose)
    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        if verbose:
            print(f"Loaded pretrained model from {pretrained_model}")
    else:

        if verbose:
            print("No pretrained model found, proceeding by pretraining")
        model = pretraining(model=model)
        # Save pretrained model and mesh for sanity check
        allround_save_data(
            model=model,
            storage_path=general_config["storage_paths"].get("pretrained_dir","/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_models"),
            save_model=True,
            save_mesh=True,
            verbose=verbose,
            pretraining=True
        )
    
    # Start optimisation
    training_config= load_general_config(v="v1", cfg_name=general_config["active_configs"].get("training",None))
    
    # Check if model still is on device after pretraining
    model = model.to(device)

    
    model.optimise(training_config=training_config)
    if verbose:
        print("Model optimisation completed.")

    
    # Save final model and mesh
    allround_save_data(
        model=model,
        storage_path=general_config["general_info"].get("storage_path","/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"),
        save_model=True,
        save_mesh=True,
        verbose=verbose,
        pretraining=False
    )

    print("Completed job successfully.")

if __name__ == "__main__":
    main()
