from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from meta_automl.data_preparation.file_system import PathType


class CacheType(Enum):
    file = 'file'
    directory = 'directory'


@dataclass
class CacheProperties:
    type_: Optional[CacheType] = None
    dir_: Optional[Path] = None
    template: Optional[PathType] = None
