import hashlib
import json


def dict_hashing(data: dict) -> str:
    """
    Generate a SHA-1 hash for the given data dictionary.

    Returns a 10 hex digit hash string.
    Args:
        data (dict): The data to hash.

        """
    canonical_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hash_value = hashlib.sha1(canonical_bytes).hexdigest()[:10]
    return hash_value

def str_hashing(data: str) -> str:
    """
    Generate a SHA-1 hash for the given string.

    Returns a 10 hex digit hash string.
    Args:
        data (str): The string to hash.

        """
    canonical_bytes = data.encode("utf-8")
    hash_value = hashlib.sha1(canonical_bytes).hexdigest()[:10]
    return hash_value