import json

from .utils import to_python_float

# =============================================================================
# https://discuss.pytorch.org/t/typeerror-tensor-is-not-json-serializable/36065/3
# =============================================================================


def _to_json_dict_with_strings(dictionary):
    """Convert dict to dict with leafs only being strings. So it recursively
    makes keys to strings if they are not dictionaries.

    Use case:
        - saving dictionary of tensors (convert the tensors to strins!)
        - saving arguments from script (e.g. argparse) for it to be pretty

    e.g.
    """
    if not isinstance(dictionary, dict):
        return str(to_python_float(dictionary))
    d = {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}
    return d


def to_json(dic):
    if isinstance(dic, dict):
        dic = dict(dic)
    else:
        dic = dic.__dict__
    return _to_json_dict_with_strings(dic)


def save_to_json_pretty(dic, path, mode="w", indent=4, sort_keys=True):
    with open(path, mode) as f:
        json.dump(to_json(dic), f, indent=indent, sort_keys=sort_keys)


def my_pprint(dic):
    """

    @param dic:
    @return:

    Note: this is not the same as pprint.
    """
    # make all keys strings recursively with their naitve str function
    dic = to_json(dic)
    # pretty print
    pretty_dic = json.dumps(dic, indent=4, sort_keys=True)
    print(pretty_dic)
    # print(json.dumps(dic, indent=4, sort_keys=True))
    # return pretty_dic


if __name__ == "__main__":
    # import json  # results in non serializabe errors for torch.Tensors
    from pprint import pprint

    import torch

    dic = {"x": torch.randn(1, 3), "rec": {"y": torch.randn(1, 3)}}

    my_pprint(dic)
    pprint(dic)
