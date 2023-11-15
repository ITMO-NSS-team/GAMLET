from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict
from contextlib import AbstractContextManager
from functools import wraps
from inspect import getcallargs, ismethod
from pathlib import Path
from typing import Literal, Sequence

from meta_automl.data_preparation.file_system import get_cache_dir

CacheAccess = Literal['r', 're', 'ew', 'rew', 'e']

logger = logging.getLogger(__file__)

COMP_CACHE_FILE_NAME = '_comp_cache.pkl'

_COMP_CACHE: None | dict = None


class NoCache:
    def __bool__(self):
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, NoCache)


def _get_hash(objects: Sequence[object]) -> str:
    return _json_dumps(tuple(objects))


def _json_dumps(obj: object) -> str:
    return json.dumps(
        obj,
        default=_json_default,
        # force formatting-related options to known values
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
    )


def _json_default(obj: object):
    obj_class = obj.__class__
    class_path = '.'.join((obj_class.__module__, obj_class.__name__))
    vars_dict = vars(obj) if hasattr(obj, '__dict__') else {}
    return _json_dumps(dict(__class_path__=class_path, **vars_dict))


def initialize_cache(file_path: os.PathLike) -> dict:
    try:
        logger.info(f'Loading cache file "{file_path}"...')
        with open(file_path, 'rb') as f:
            cache_dict = pickle.load(f)
    except FileNotFoundError:
        logger.info(f'File "{file_path}" not found. Creating a new one...')
        cache_dict = defaultdict(NoCache)
    return cache_dict


def update_cache(cache_dict, file_path: os.PathLike):
    logger.info(f'Writing cache file "{file_path}"...')
    with open(file_path, 'wb') as f:
        pickle.dump(cache_dict, f)


def parse_key(__key: str, func, *args, **kwargs):
    return eval(__key, getcallargs(func, *args, **kwargs))


class Cache(AbstractContextManager):
    def __init__(self, file_path: os.PathLike | str | None = None, access: CacheAccess = 'rew'):
        file_path = file_path or COMP_CACHE_FILE_NAME
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = get_cache_dir() / file_path
        self.file_path = file_path
        self.access = access
        self.cache_dict = None

    def __call__(self, func, key: str | None = None):
        @wraps(func)
        def decorated(*args, **kwargs):
            with self:
                access = self.access
                cache_dict = self.cache_dict

                if access == 'e':
                    logger.info(f'Executing cache value, since no access to the cache is provided...')
                    return func(*args, **kwargs)

                if key:
                    hash_ = parse_key(key, func, *args, **kwargs)
                else:
                    hash_objects = [func.__name__, args, tuple(sorted(kwargs.items()))]

                    if ismethod(func):
                        hash_objects.insert(0, func.__self__)

                    hash_ = _get_hash(hash_objects)
                was_read = False
                if 'r' in access:
                    logger.info(f'Getting cache for the key "{hash_}"...')
                    res = cache_dict[hash_]
                else:
                    res = NoCache()

                if not isinstance(res, NoCache):
                    logger.info(f'Found the cache for the key "{hash_}": {res}...')
                    was_read = True

                if isinstance(res, NoCache) and 'e' not in access:
                    raise ValueError(
                        f'No cache found for {func.__name__}({args, kwargs}) in "{self.file_path}", '
                        f'but computation is not allowed (access="{access}").')

                if 'e' in access and not was_read:
                    logger.info(f'Executing cache value for the key "{hash_}"...')
                    res = func(*args, **kwargs)

                if 'w' in access and not was_read and not isinstance(res, NoCache):
                    logger.info(f'Writing cache for the key "{hash_}": {res}...')
                    cache_dict[hash_] = res

                return res

        return decorated

    def __enter__(self):
        if 'r' in self.access:
            self.cache_dict = initialize_cache(self.file_path)
        else:
            self.cache_dict = defaultdict(NoCache)
        return self.cache_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'w' in self.access:
            update_cache(self.cache_dict, self.file_path)
        self.cache_dict.clear()
        self.cache_dict = None
