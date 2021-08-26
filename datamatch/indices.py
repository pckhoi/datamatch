"""
An index divides the data up into one or more buckets. Only records in the same bucket are matched against each other.
When used correctly this decreases the number of pairs to compare and speeds up the matching process significantly.
"""

import operator
import functools
import itertools
from typing import Type
from abc import ABC, abstractmethod

import pandas as pd


class BaseIndex(ABC):
    """Abstract base class for all index classes.

    Sub-class should implement :meth:`_key_ind_map` method.

    .. automethod:: _key_ind_map
    """
    _dfs: list[tuple[pd.DataFrame, list]]

    def __init__(self) -> None:
        self._dfs = list()

    @abstractmethod
    def _key_ind_map(self, df: pd.DataFrame) -> dict:
        """Returns a mapping between bucket keys and row indices.

        :param df: the data to index
        :type df: :class:`pandas:pandas.DataFrame`

        :return: a mapping between bucket key and all row indices that belong
            to the bucket. Key could be anything hashable but the value must always
            be a list even if there is only one row.
        :rtype: :obj:`dict`
        """
        raise NotImplementedError()

    def keys(self, df: pd.DataFrame) -> set:
        """Returns a set of keys that could be used to retrieve buckets

        :param df: the data to index
        :type df: :class:`pandas:pandas.DataFrame`

        :return: a set of bucket keys
        :rtype: :obj:`set`
        """
        key_inds = self._key_ind_map(df)
        self._dfs.append((df, key_inds))
        return set(key_inds.keys())

    def bucket(self, df: pd.DataFrame, key: any) -> pd.DataFrame:
        """Retrieves a bucket given the original data and a bucket key

        :param df: the data to index
        :type df: :class:`pandas:pandas.DataFrame`

        :param key: one of the keys returned from :meth:`BaseIndex.keys`
        :type key: any

        :return: rows in bucket
        :rtype: :class:`pandas:pandas.DataFrame`
        """
        for frame, key_inds in self._dfs:
            if frame is df:
                break
        else:
            raise ValueError(
                "frame not registered, make sure to run index.keys(df) first."
            )
        rows = df.loc[key_inds[key]]
        if len(rows.shape) == 1:
            rows = rows.to_frame().transpose()
        return rows


class NoopIndex(BaseIndex):
    """Returns all data as a single bucket.

    Using this is like using no index at all. Useful for when you don't care about
    performance (e.g. when there are not too many rows).
    """

    def _key_ind_map(self, df: pd.DataFrame) -> dict:
        return {0: df.index.tolist()}


class ColumnsIndex(BaseIndex):
    """Split data into multiple buckets based on one or more columns.
    """

    def __init__(self, cols: str or list[str]) -> None:
        """
        :param cols: single column name or list of column names to index
        :type cols: :obj:`str` or :obj:`list` of :obj:`str`
        """
        super().__init__()
        if type(cols) is str:
            self._cols = [cols]
        else:
            self._cols = cols

    def _key_ind_map(self, df: pd.DataFrame) -> dict:
        result = dict()
        for idx, row in df.iterrows():
            key = tuple(
                row[col] for col in self._cols
            )
            result.setdefault(key, list()).append(idx)
        for l in result.values():
            l.sort()
        return result


class MultiIndex(BaseIndex):
    """Creates bucket keys by combining bucket keys from 2 or more indices.

    This has 2 modes of operation:

    - When **combine_keys** is `False`: the key sets of each index are concatenated together, this is like OR-ing the keys.
    - When **combine_keys** is `True`: the final key set is the cartesian product of all key sets, this is like AND-ing the keys.
    """

    def __init__(self, indices: list[Type[BaseIndex]], combine_keys: bool = False) -> None:
        """
        :param indices: list of indices to combine
        :type indices: :obj:`list` of :class:`BaseIndex` subclass

        :param combine_keys: whether the final key set should be the cartesian product of all key sets, defaults to `False`
        :type combine_keys: :obj:`bool`
        """
        super().__init__()
        self._indices = indices
        self._combine_keys = combine_keys

    def _key_ind_map(self, df: pd.DataFrame) -> dict:
        maps = [idx._key_ind_map(df) for idx in self._indices]
        result = dict()
        if self._combine_keys:
            for prod in itertools.product(*[d.items() for d in maps]):
                rows = list(functools.reduce(
                    operator.and_, [set(vals) for _, vals in prod]
                ))
                if len(rows) == 0:
                    continue
                key = tuple(key for key, _ in prod)
                result[key] = rows
        else:
            for d in maps:
                for key, rows in d.items():
                    result[key] = result.get(key, set()) | set(rows)
            result = {key: sorted(list(rows)) for key, rows in result.items()}
        return result
