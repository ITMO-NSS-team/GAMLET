from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict
from contextlib import AbstractContextManager
from functools import wraps
from inspect import getcallargs, ismethod, signature
from pathlib import Path
from typing import Any, Callable, Hashable, Literal, Sequence

from meta_automl.data_preparation.file_system import get_cache_dir

CacheAccess = Literal['r', 're', 'ew', 'rew', 'e']

logger = logging.getLogger(__file__)

COMP_CACHE_FILE_NAME = '_comp_cache.pkl'


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


def initialize_cache(file_path: os.PathLike) -> Any:
    try:
        logger.info(f'Loading cache file "{file_path}"...')
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
    except FileNotFoundError:
        logger.info(f'File "{file_path}" not found.')
        result = NoCache()
    return result


def initialize_cache_dict(file_path: os.PathLike) -> dict:
    cache_dict = initialize_cache(file_path)
    if isinstance(cache_dict, NoCache):
        logger.info('Creating a new cache dict...')
        cache_dict = defaultdict(NoCache)
    elif not isinstance(cache_dict, dict):
        raise ValueError(f'File "{file_path}" contains value of type "{type(cache_dict)}", not dictionary.'
                         f'Rename or delete it beforehand.')
    return cache_dict


def update_cache(cache: Any, file_path: os.PathLike):
    logger.info(f'Writing cache file "{file_path}"...')
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)


def parse_key(callable_or_code: Callable[[Any], Hashable] | str, func: Callable, *args, **kwargs) -> Hashable:
    if callable(callable_or_code):
        sign_params = signature(callable_or_code).parameters
    elif isinstance(callable_or_code, str):
        sign_params = callable_or_code
    else:
        raise ValueError(f'Inner key should be either string or callable, got {type(callable_or_code)}.')

    if 'args' in sign_params or 'kwargs' in sign_params:
        input_kwargs = dict(args=args, kwargs=kwargs)
    else:
        input_kwargs = getcallargs(func, *args, **kwargs)

    if callable(callable_or_code):
        key = callable_or_code(**input_kwargs)
    else:
        key = eval(callable_or_code, None, input_kwargs)
    return key


class CacheDict(AbstractContextManager):
    """ Decorator/context manager for caching of evaluation results.
    Creates "pickle" file at disk space on a specified path.

    If used as a context, provides a dictionary to put/read values in.
    To do so, use the syntax "with *instance*: ...".

    If used as a decorator, wraps a function and stores its execution results mapped from hash.
    To do so, use the method ``CacheDict.decorate()``

    Args:

        file_path - a path to an existing or non-existent pickle file.
            If a relative path or a filename is given, puts it into the framework cache directory.

        access - cache access indicators. The string may include the following indicators:
            - ``r`` - read - grants access to read the cache file content
            - ``e`` - execute/evaluate - grants access to evaluate the decorated function (if such is present)
            - ``w`` - write - grants access to modify the cache file content

    Examples
    --------
    Example 1:

    >>> with CacheDict('example_cache_dict.pkl') as cache_dict:
    >>>     x = np.array([[1, 2], [3, 4]])
    >>>     x_T = cache_dict['x_T']  # Read the cache first
    >>>     if x_T is None:  # If cache not found,
    >>>         x_T = x.T    #   then execute the value
    >>>     cache_dict['x_T'] = x_T  # Put the value in cache

    Example 2:

    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6], [7, 8]])
    >>>
    >>> cached_mult = CacheDict.decorate(
    >>>     np.multiply,  # Select a function to cache.
    >>>     file_path='np_multiplication.pkl',  # Select path to a pickle file.
    >>>     inner_key='tuple(map(lambda a: a.data.tobytes(), args))')  # Retrieve hashable representation of args.
    >>>
    >>> cached_mult(a, b)
    array([[ 5, 12],
       [21, 32]])

    """

    def __init__(self, file_path: os.PathLike | str | None = None, access: CacheAccess = 'rew'):
        file_path = file_path or COMP_CACHE_FILE_NAME
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = get_cache_dir() / file_path
        self.file_path = file_path
        self.access = access
        self.cache_dict = None

    @classmethod
    def decorate(cls, func: Callable,
                 file_path: os.PathLike | str | None = None,
                 access: CacheAccess = 'rew',
                 outer_key: Hashable | None = None,
                 inner_key: str | Callable[[Any], Hashable] | None = None) -> Callable:
        """ Wraps a function and stores its execution results into cache dictionary.

        Example:

        >>> import numpy as np
        >>> a = np.array([[1, 2], [3, 4]])
        >>> b = np.array([[5, 6], [7, 8]])
        >>>
        >>> cached_mult = CacheDict.decorate(
        >>>     np.multiply,
        >>>     file_path='np_multiplication.pkl',
        >>>     inner_key='tuple(map(lambda a: a.data.tobytes(), args))')
        >>>
        >>> cached_mult(a, b)
        array([[ 5, 12],
           [21, 32]])

        Params:

            func - a function to decorate.

            file_path - a path to an existing or non-existent pickle file.
                If a relative path or a filename is given, puts it into the framework cache directory.

            access - cache access indicators. The string may include the following indicators:
                - ``r`` - read - grants access to read the cache file content
                - ``e`` - execute/evaluate - grants access to evaluate the decorated function (if such is present)
                - ``w`` - write - grants access to modify the cache file content

            outer_key - a constant hashable key to store the function call's result.

            inner_key - a callable or a code expression that evaluates a hashable key to store
                the function call's result. To do so, use argument names that are used inside the function.
                Some functions do not support signatures and will throw an error.
                You may use "args" and "kwargs" in your expression instead.
        """
        if outer_key is not None and inner_key is not None:
            raise ValueError('At most one of (outer key, inner key) can be specified.')

        @wraps(func)
        def decorated(*args, **kwargs):
            with cls(file_path, access) as cache_dict:
                if access == 'e':
                    logger.info(f'Executing cache value, since no access to the cache is provided...')
                    return func(*args, **kwargs)

                if outer_key is not None:
                    hash_ = outer_key
                elif inner_key is not None:
                    hash_ = parse_key(inner_key, func, *args, **kwargs)
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
                        f'No cache found for {func.__name__}({args, kwargs}) in "{file_path}", '
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
            self.cache_dict = initialize_cache(self.file_path) or defaultdict(NoCache)
        else:
            self.cache_dict = defaultdict(NoCache)
        return self.cache_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'w' in self.access:
            update_cache(self.cache_dict, self.file_path)
        self.cache_dict.clear()
        self.cache_dict = None


class OneValueCache:
    def __init__(self, file_path: os.PathLike | str, access: CacheAccess = 'rew'):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        self.file_path = file_path
        self.access = access

    def __call__(self, func: Callable):
        @wraps(func)
        def decorated(*args, **kwargs):
            access = self.access

            if access == 'e':
                logger.info(f'Executing cache value, since no access to the cache is provided...')
                return func(*args, **kwargs)

            was_read = False
            if 'r' in access:
                logger.info(f'Getting cache from "{self.file_path}"...')
                res = initialize_cache(self.file_path)
            else:
                res = NoCache()

            if not isinstance(res, NoCache):
                logger.info(f'Found the cache for "{self.file_path}": {res}...')
                was_read = True

            if isinstance(res, NoCache) and 'e' not in access:
                raise ValueError(
                    f'No cache found in "{self.file_path}", '
                    f'but execution is not allowed (access="{access}").')

            if 'e' in access and not was_read:
                logger.info(f'Executing cache value for "{self.file_path}"...')
                res = func(*args, **kwargs)

            if 'w' in access and not was_read and not isinstance(res, NoCache):
                logger.info(f'Writing cache for "{self.file_path}": {res}...')
                update_cache(res, self.file_path)

            return res

        return decorated
