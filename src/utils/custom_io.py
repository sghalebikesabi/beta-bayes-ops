"""Utility functions for Input/Output.
Adapted from https://github.com/bayesiains/nsf/blob/eaa9377f75df1193025f6b2487524cf266874472/utils/torchutils_test.py
"""

from collections.abc import MutableMapping

import os
import omegaconf
import socket
import time
import warnings


def on_cluster():
    hostname = socket.gethostname()
    return False if (hostname == "XXXX" or hostname == "") else True


def get_timestamp():
    formatted_time = time.strftime("%d-%b-%y||%H:%M:%S")
    return formatted_time


def get_project_root():
    hostname = socket.gethostname()
    path = "XXXX"
    return path


def get_log_root():
    path = f"{get_project_root()}/log"
    return path


def get_data_root():
    path = f"{get_project_root()}/datasets"
    return path


def get_checkpoint_root(from_cluster=False):
    path = f"{get_project_root()}/checkpoints"
    return path


def get_output_root():
    path = f"{get_project_root()}/results"
    return path


def get_final_root():
    path = f"{get_project_root()}/final"
    return path


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    """Flatten a nested dictionary.

    from https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def turn_dict_to_filename(d):
    return "_".join(["{}{}".format(k[:1], v) for k, v in d.items()])


def make_run_name(config):
    sup_keys_to_keep = ["seed", "sweeps", "data", "methods_args", "attack"]
    cleaned_config = {k: v for k, v in config.items() if k in sup_keys_to_keep}

    cleaned_config = flatten_dict(cleaned_config)

    flat_keys_to_drop = []
    cleaned_config = {
        k: v for k, v in cleaned_config.items() if k not in flat_keys_to_drop
    }
    cleaned_config = {
        k: v
        if not isinstance(v, omegaconf.listconfig.ListConfig)
        else "-".join([str(x) for x in v])
        for k, v in cleaned_config.items()
    }
    if config.get("run_name", False) and config.run_name.get("model_keys_name"):
        filtered_config = {}
        for k in config.run_name.model_keys_name:
            if k in cleaned_config:
                filtered_config[k] = cleaned_config[k]
            else:
                warnings.warn(f"Warning: {k} not in config")
        cleaned_config = filtered_config

    return config.run_name.get("prefix") + turn_dict_to_filename(cleaned_config)


def create_date_model_folder(parent_folder, model_name, config=None):
    date = time.strftime("%Y-%m-%d")
    folder_path = f"{parent_folder}/{date}/{model_name}"
    os.makedirs(folder_path, exist_ok=True)
    return folder_path
