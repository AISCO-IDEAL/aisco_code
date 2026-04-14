import importlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

from torch.nn import Module

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config


def dynamic_type_loading(
    file_path: str, class_name: str
) -> Module:  # Returns <class 'type'> object, i.e. not an instance but a callable instead
    """
    Dynamically returns a class type from a given file path and corresponding class name.
    I.e. not an instance but a callable instead

    Args:
        file_path (str): The dot-separated path to the module containing the class.
        class_name (str): The name of the class to be loaded.

    Returns:
        type: The class type object.

    """

    # Accept both module notation (a.b.c) and file-system notation (a/b/c.py).
    # If a real Python file exists at the given path, import directly from it.

    # 1. Remove a trailing .py extension if present
    file_path = file_path.removesuffix(".py")

    # 2. Convert file-system path separators to module separators
    file_path = file_path.replace("\\", "/").strip("/").replace("/", ".")

    # 3. Import and return
    module = importlib.import_module(file_path)
    ObjectType: Module = getattr(module, class_name)
    return ObjectType


def load_class_from_path(path: str, loss_family: str):
    """Dynamically load a class from a file path."""
    spec = importlib.util.spec_from_file_location("module.name", path)

    # 3. Add explicit check for None to prevent AttributeError
    if spec is None:
        raise FileNotFoundError(
            f"Cannot find module file for dynamic loading. "
            f"Path provided: {path}. "
            f"Please verify the file exists and the path in the config (must use '/' separators)."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    loader = spec.loader
    if loader is None or not hasattr(loader, "exec_module"):
        raise ImportError(f"No loader available for module spec when loading from path: {path}")
    loader.exec_module(module)
    return getattr(module, loss_family)


def _load_models_inventory() -> dict:
    """
    Load the models inventory from models.json.

    Args:
        version (str): The version directory, defaults to "v2".

    Returns:
        dict: The models inventory dictionary with structure:
              {
                  "models": {
                      "ModelName": {
                          "model_class": "ClassName",
                          "parent_class": "ParentClassName",
                          "file_path": "path/to/model.py",
                          "config_file": "config.yaml"
                      },
                      ...
                  }
              }

    Raises:
        FileNotFoundError: If models.json cannot be found.
        json.JSONDecodeError: If models.json is malformed.
    """
    # Construct path to models.json
    # Go From here: .../v2/src/utils/dynamic_loading.py
    # To there: .../v2/configs/models.json
    models_json_path = Path(__file__).parent.parent.parent / "configs" / "models.json"

    if not models_json_path.exists():
        raise FileNotFoundError(
            f"Models inventory file not found at {models_json_path}. Please ensure models.json exists in the configs directory."
        )

    try:
        with open(models_json_path) as f:
            inventory = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse models.json: {e.msg}",
            e.doc,
            e.pos,
        ) from e

    return inventory


def smart_model_init(model_name: str, version: str = "v2") -> tuple[Module, str, str, dict[str, bool]]:
    """
    Dynamically initialize a model from the inventory with its config and metadata.

    Loads a model definition from models.json, retrieves its configuration, instantiates
    the model class, and returns the instance along with metadata about its class hierarchy.
    Validates that the actual parent class matches the expected parent class from the inventory.

    Args:
        model_name (str): The name of the model to initialize (e.g., "WIREV2").
        version (str): The version directory to use, defaults to "v2".

    Returns:
        - model_instance: An initialized instance of the model class (nn.Module)
        - parent_class_name: The name of the parent class (str)
        - model_class_name: The name of the model class (str)
        - settings: A dictionary of additional settings from the inventory (e.g., uses_diversity)
    Raises:
        FileNotFoundError: If models.json or model file/config cannot be found.
        KeyError: If the model_name is not in the inventory.
        ValueError: If model configuration or initialization fails.
        json.JSONDecodeError: If models.json is malformed.
        AssertionError: If the actual parent class does not match the expected parent class
                       from the inventory.

    Example:
        >>> model, parent_cls, model_cls = smart_model_init("WIREV2")
        >>> print(f"Loaded {model_cls} (parent: {parent_cls})")
        >>> # Use model for inference or training
    """
    # Load models inventory
    inventory = _load_models_inventory()

    # Check if model exists in inventory
    if model_name not in inventory.get("models", {}):
        available = ", ".join(inventory.get("models", {}).keys())
        raise KeyError(f"Model '{model_name}' not found in inventory. Available models: {available}")

    model_entry = inventory["models"][model_name]

    # Extract model metadata from inventory
    model_class_name = model_entry.get("model_class")
    expected_parent_class = model_entry.get("parent_class")
    model_file_path = model_entry.get("file_path")
    config_file_name = model_entry.get("config_file")
    settings: dict[str, bool] = {}
    settings["uses_diversity"] = model_entry.get("uses_diversity")
    settings["has_optimise"] = model_entry.get("has_optimise")
    if not all([model_class_name, expected_parent_class, model_file_path, config_file_name]):
        raise ValueError(
            f"Incomplete model entry for '{model_name}'. Required fields: model_class, parent_class, file_path, config_file"
        )

    try:
        # Load model configuration
        model_config = load_config(v=version, cfg_name=config_file_name)
    except Exception as e:
        raise ValueError(f"Failed to load model config '{config_file_name}' for model '{model_name}': {e}") from e

    try:
        # Load model class dynamically
        model_class = dynamic_type_loading(file_path=model_file_path, class_name=model_class_name)
    except Exception as e:
        raise ImportError(f"Failed to load model class '{model_class_name}' from '{model_file_path}': {e}") from e

    # Introspect parent
    if hasattr(model_class, "__bases__") and model_class.__bases__:
        actual_parent_class = model_class.__bases__[0].__name__
    else:
        warnings.warn(
            f"Model '{model_name}' has no parent class. This may indicate an incomplete model definition.",
            UserWarning,
            stacklevel=2,
        )
        actual_parent_class = None

    # Validate parent class matches expected from inventory
    assert actual_parent_class == expected_parent_class, (
        f"Parent class mismatch for model '{model_name}': "
        f"expected '{expected_parent_class}' but got '{actual_parent_class}'. "
        f"This may indicate a broken model hierarchy or outdated inventory."
    )

    try:
        # Instantiate the model with config
        model_instance = model_class(**model_config)
    except Exception as e:
        raise ValueError(f"Failed to instantiate model '{model_name}' with config: {e}") from e

    return model_instance, actual_parent_class, model_class_name, settings


def validate_inventory(version: str = "v2", verbose: bool = True) -> dict:
    """
    Validate that all models in the inventory have accessible files and configs.

    Checks that:
    - Model Python files exist and can be imported
    - Model classes exist in those files
    - Configuration files exist and are valid YAML/JSON
    - All required fields are present in each model entry
    - Actual parent class matches expected parent class from inventory

    Args:
        version (str): The version directory to use, defaults to "v2".
        verbose (bool): If True, print validation results. Defaults to True.

    Returns:
        dict: Validation report with structure:
              {
                  "valid_models": ["WIREV2", ...],
                  "invalid_models": {
                      "ModelName": "error reason"
                  },
                  "all_valid": bool
              }

    Example:
        >>> report = validate_inventory()
        >>> if not report["all_valid"]:
        ...     for model_name, error in report["invalid_models"].items():
        ...         print(f"{model_name}: {error}")
    """
    inventory = _load_models_inventory()
    valid_models = []
    invalid_models = {}

    for model_name, entry in inventory.get("models", {}).items():
        try:
            # Check required fields
            model_class_name = entry.get("model_class")
            expected_parent_class = entry.get("parent_class")
            file_path = entry.get("file_path")
            config_file = entry.get("config_file")

            if not all([model_class_name, expected_parent_class, file_path, config_file]):
                raise ValueError("Missing required fields: model_class, parent_class, file_path, or config_file")  # noqa: TRY301

            # Try to load the model class (this validates file exists and class is importable)
            model_class = dynamic_type_loading(file_path=file_path, class_name=model_class_name)

            # Validate parent class matches
            if hasattr(model_class, "__bases__") and model_class.__bases__:
                actual_parent_class = model_class.__bases__.__name__
            else:
                warnings.warn(
                    f"Model '{model_name}' has no parent class. This may indicate an incomplete model definition.",
                    UserWarning,
                    stacklevel=2,
                )
                actual_parent_class = None

            if actual_parent_class != expected_parent_class:
                raise AssertionError(f"Parent class mismatch: expected '{expected_parent_class}' but got '{actual_parent_class}'")  # noqa: TRY301

            # Try to load the config (this validates config file exists)
            _config = load_config(v=version, cfg_name=config_file)

            # If we got here, all checks passed
            valid_models.append(model_name)
            if verbose:
                print(f"✓ {model_name}: {model_class_name} (parent: {actual_parent_class})")

        except Exception as e:
            invalid_models[model_name] = str(e)
            if verbose:
                print(f"✗ {model_name}: {e}")

    report = {
        "valid_models": valid_models,
        "invalid_models": invalid_models,
        "all_valid": len(invalid_models) == 0,
    }

    return report
