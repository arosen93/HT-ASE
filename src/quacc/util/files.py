"""
Utility functions for file and path handling
"""
from __future__ import annotations

import contextlib
import os
import socket
import warnings
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from random import randint
from shutil import copy

import yaml
from monty.io import zopen
from monty.os.path import zpath
from monty.shutil import decompress_file


def check_logfile(logfile: str, check_str: str) -> bool:
    """
    Check if a logfile has a given string (case-insensitive).

    Parameters
    ----------
    logfile : str
        Path to the logfile.
    check_str : str
        String to check for.

    Returns
    -------
    bool
        True if the string is found in the logfile, False otherwise.
    """
    zlog = zpath(logfile)
    with zopen(zlog, "r") as f:
        for line in f:
            if not isinstance(line, str):
                line = line.decode("utf-8")
            if check_str.lower() in line.lower():
                return True
    return False


def copy_decompress(source_files: list[str], destination: str) -> None:
    """
    Copy and decompress files from source to destination.

    Parameters
    ----------
    source_files
        List of files to copy and decompress.
    destination
        Destination directory.

    Returns
    -------
    None
    """
    for f in source_files:
        z_path = zpath(f)
        if os.path.exists(z_path):
            z_file = os.path.basename(z_path)
            copy(z_path, os.path.join(destination, z_file))
            decompress_file(os.path.join(destination, z_file))
        else:
            warnings.warn(f"Cannot find file: {z_path}", UserWarning)


def make_unique_dir(base_path: str | None = None) -> str:
    """
    Make a directory with a unique name. Uses the same format as Jobflow.

    Parameters
    ----------
    base_path
        Path to the base directory.

    Returns
    -------
    str
        Path to the job directory.
    """
    time_now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
    job_dir = f"quacc-{time_now}-{randint(10000, 99999)}"
    if base_path:
        job_dir = os.path.join(base_path, job_dir)
    os.makedirs(job_dir)

    return job_dir


def load_yaml_calc(yaml_path: str | Path) -> dict:
    """
    Loads a YAML file containing calculator settings.

    Parameters
    ----------
    yaml_path
        Path to the YAML file.

    Returns
    -------
    dict
        The calculator configuration (i.e. settings).
    """

    _, ext = os.path.splitext(yaml_path)
    if not ext:
        yaml_path += ".yaml"

    if not os.path.exists(yaml_path):
        raise ValueError(f"Cannot find {yaml_path}.")

    # Load YAML file
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)

    # Inherit arguments from any parent YAML files
    # but do not overwrite those in the child file.
    for config_arg in config.copy():
        if "parent" in config_arg:
            # Relative Path
            yaml_parent_path = os.path.join(
                os.path.dirname(yaml_path), config[config_arg]
            )

            # Absolute path
            if not os.path.exists(yaml_parent_path):
                if os.path.exists(Path(config[config_arg])):
                    yaml_parent_path = Path(config[config_arg])
                else:
                    # Try package data
                    pkg_name = config[config_arg].split(".")[0]
                    with contextlib.suppress(ImportError):
                        pkg_data_path = files(pkg_name)
                        y_path = Path("/".join(config[config_arg].split(".")))
                        yaml_parent_path = pkg_data_path / y_path
            parent_config = load_vasp_yaml_calc(yaml_parent_path)
            for k, v in parent_config.items():
                if k not in config:
                    config[k] = v
                else:
                    v_new = parent_config.get(k, {})
                    for kk, vv in v_new.items():
                        if kk not in config[k]:
                            config[k][kk] = vv

    return config


def load_vasp_yaml_calc(yaml_path: str | Path) -> dict:
    """
    Loads a YAML file containing calculator settings.

    Parameters
    ----------
    yaml_path
        Path to the YAML file.

    Returns
    -------
    dict
        The calculator configuration (i.e. settings).
    """

    config = load_yaml_calc(yaml_path)
    if "INCAR" in config:
        if "MAGMOM" in config["INCAR"]:
            config["inputs"]["elemental_magmoms"] = config["INCAR"]["MAGMOM"]
            del config["MAGMOM"]
        config["inputs"] = config["INCAR"]
        del config["INCAR"]
    if "POTCAR" in config:
        config["setups"] = config["POTCAR"]
        del config["POTCAR"]

    for k in config["inputs"]:
        config["inputs"][k] = k.lower()

    # Allow for either "Cu_pv" and "_pv" style setups
    if "inputs" in config and config.get(["setups"]):
        for k, v in config["inputs"]["setups"].items():
            if k in v:
                config["inputs"]["setups"][k] = v.split(k)[-1]


def find_recent_logfile(dir_name: Path | str, logfile_extensions: str | list[str]):
    """
    Find the most recent logfile in a given directory.

    Parameters
    ----------
    dir_name
        The path to the directory to search
    logfile_extensions
        The extension (or list of possible extensions) of the logfile to search for.
        For an exact match only, put in the full file name.

    Returns
    -------
    logfile
        The path to the most recent logfile with the desired extension
    """
    mod_time = 0.0
    logfile = None
    if isinstance(logfile_extensions, str):
        logfile_extensions = [logfile_extensions]
    for f in os.listdir(dir_name):
        f_path = os.path.join(dir_name, f)
        for ext in logfile_extensions:
            if ext in f and os.path.getmtime(f_path) > mod_time:
                mod_time = os.path.getmtime(f_path)
                logfile = os.path.abspath(f_path)
    return logfile


def get_uri(dir_name: str | Path) -> str:
    """
    Return the URI path for a directory.

    This allows files hosted on different file servers to have distinct locations.

    Adapted from Atomate2.

    Parameters
    ----------
    dir_name : str
        A directory name.

    Returns
    -------
    str
        Full URI path, e.g., "fileserver.host.com:/full/path/of/dir_name".
    """
    fullpath = Path(dir_name).absolute()
    hostname = socket.gethostname()
    with contextlib.suppress(socket.gaierror, socket.herror):
        hostname = socket.gethostbyaddr(hostname)[0]
    return f"{hostname}:{fullpath}"
