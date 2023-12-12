import functools
from typing import Optional, Sequence, Union

from pandas import DataFrame


class DatasetMetaFeatures(DataFrame):
    _metadata = ["is_summarized", "features"]

    def __init__(self, *args, is_summarized: bool = True, features: Optional[Sequence[Union[str, int]]] = None,
                 **kwargs):
        self.is_summarized = is_summarized
        self.features = features
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return functools.partial(type(self), is_summarized=self.is_summarized)
