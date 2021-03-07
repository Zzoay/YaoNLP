
import json
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Any, Union, Optional
from collections import namedtuple

from yaonlp import checker


# # add type Config, NameTuple actually
# Config = NamedTuple

def _load_json(config_file: str) -> dict:
    with open(config_file) as f:
        dct = json.load(f)
        return dct


def _dct_to_namespace(dct: dict) -> SimpleNamespace:
    config_nspc = SimpleNamespace(**dct)
    return config_nspc


def _dct_to_nametuple(dct: dict) -> Tuple[Any, ...]: # NamedTuple actually
    # MyTuple = namedtuple("config", list(dct.keys()))
    # config_ntpl = MyTuple(**dct)
    config_ntpl = namedtuple('config', dct.keys())(**dct)

    return config_ntpl


# TODO: Think about more efficient way to load config
def load_config(config_file: str, mode: str = "dict"): 
    # config = _dct_to_namespace(_load_json(config_path))
    config_dct = _load_json(config_file)
    
    # check path and filedir
    checker.check_dirConfig(config_dct)

    if mode == "dict":
        return config_dct
    elif mode == "namedtuple":
        return _dct_to_nametuple(config_dct)
    
    assert False, "mode not found."
