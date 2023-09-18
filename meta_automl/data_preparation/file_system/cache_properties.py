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
    type: Optional[CacheType] = None
    dir: Optional[Path] = None
    path_template: Optional[PathType] = None
