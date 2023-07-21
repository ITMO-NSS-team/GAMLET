from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

PathType = Union[PathLike, str]

DATA_SUBDIR = 'data'


def ensure_dir_exists(dir_: Path) -> Path:
    if dir_.is_file():
        dir_ = dir_.parent
    if not dir_.exists():
        dir_.mkdir(parents=True)
    return dir_


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parents[3]


def get_data_dir() -> Path:
    data_dir = get_project_root().joinpath(DATA_SUBDIR)
    return data_dir
