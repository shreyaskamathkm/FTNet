import json
from pathlib import Path
from typing import Any, Dict

from .conversion import to_python_float

# =============================================================================
# https://discuss.pytorch.org/t/typeerror-tensor-is-not-json-serializable/36065/3
# =============================================================================


def _to_json_dict_with_strings(dictionary: Any) -> Any:
    """Convert dict to dict with leafs only being strings. Recursively converts
    values to strings if they are not dictionaries.

    Use case:
        - saving dictionary of tensors (convert the tensors to strings!)
        - saving arguments from script (e.g. argparse) for it to be pretty
    Args:
        dictionary (Any): The input dictionary or value.

    Returns:
        Any: The converted dictionary or string value.
    """
    if not isinstance(dictionary, dict):
        return str(to_python_float(dictionary))
    return {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}


def _to_json(input: Any) -> Any:
    """Convert a dictionary to a JSON-compatible dictionary with string values.

    Args:
        dic (Any): The input dictionary.

    Returns:
        Any: The JSON-compatible dictionary.
    """
    dic = dict(input) if isinstance(input, dict) else input.__dict__
    return _to_json_dict_with_strings(dic)


def save_to_json_pretty(
    input: Dict, path: Path, mode: str = "w", indent: int = 4, sort_keys: bool = True
) -> None:
    """Save a dictionary to a JSON file with pretty formatting.

    Args:
        input (Dictionary): The input dictionary.
        path (Path): The path to the output JSON file.
        mode (str): The file mode (default is "w").
        indent (int): The indentation level for pretty printing (default is 4).
        sort_keys (bool): Whether to sort the keys in the JSON output (default is True).
    """
    with path.open(mode) as f:
        json.dump(_to_json(input), f, indent=indent, sort_keys=sort_keys)


def my_pprint(input: Dict) -> None:
    """Pretty print a dictionary with string values.

    Args:
        input (Dict): The input dictionary.
    """
    pretty_dic = json.dumps(_to_json(input), indent=4, sort_keys=True)
    print(pretty_dic)
