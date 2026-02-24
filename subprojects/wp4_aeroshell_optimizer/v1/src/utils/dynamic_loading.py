import importlib
import importlib.util
import sys


def dynamic_type_loading(
    file_path: str, class_name: str
) -> type:  # Returns <class 'type'> object, i.e. not an instance but a callable instead
    """
    Dynamically returns a class type from a given file path and corresponding class name.
    I.e. not an instance but a callable instead

    Args:
        file_path (str): The dot-separated path to the module containing the class.
        class_name (str): The name of the class to be loaded.

    Returns:
        type: The class type object.

    """

    file_path = file_path.replace("/", ".")  # convert path to module format
    module = importlib.import_module(file_path)
    ObjectType = getattr(module, class_name)  # gets <class 'type'>
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
    spec.loader.exec_module(module)
    return getattr(module, loss_family)
