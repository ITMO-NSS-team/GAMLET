from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import scipy as sp


@dataclass
class Dataset:
    name: str
    X: Union[np.ndarray, pd.DataFrame, sp.sparse.csr_matrix]
    y: Optional[Union[np.ndarray, pd.DataFrame]]
    path: Path
