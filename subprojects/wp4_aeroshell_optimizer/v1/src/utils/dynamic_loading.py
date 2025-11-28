
import importlib


def dynamic_type_loading(file_path: str, class_name: str) -> type: # Returns <class 'type'> object, i.e. not an instance but a callable instead
    """
    Dynamically returns a class type from a given file path and corresponding class name.
    """

    file_path = file_path.replace("/", ".")  # convert path to module format
    module = importlib.import_module(file_path) 
    ObjectType = getattr(module, class_name)  # gets <class 'type'>
    return ObjectType
