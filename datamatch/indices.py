import hashlib
import operator
import functools
import itertools
from typing import Type

import pandas as pd


class BaseIndex(object):
    """Base class for all index classes.

    This doesn't have any functionality but simple serve as an interface for all
    subclasses to implement.
    """
    _dfs: list[tuple[pd.DataFrame, list]]

    def __init__(self) -> None:
        self._dfs = list()

    def _key_ind_map(self, df: pd.DataFrame) -> dict:
        """Returns a mapping between bucket keys and row indices.

        This is the mandatory method for all subclassses to implement.

        Args:
            df (pd.DataFrame): the data to index

        Returns:
            a mapping between bucket key and all row indices that belong
            to the bucket. Key could be anything hashable but value must always
            be a list even if there are only one row.
        """
        raise NotImplementedError()

    def keys(self, df: pd.DataFrame) -> set:
        """Returns a set of keys that should be used to retrieve buckets

        Args:
            df (pd.DataFrame): the data to index

        Returns:
            a set of bucket keys
        """
        key_inds = self._key_ind_map(df)
        self._dfs.append((df, key_inds))
        return set(key_inds.keys())

    def bucket(self, df: pd.DataFrame, key: any) -> pd.DataFrame:
        """Retrieve a bucket given the original data and a bucket key

        Args:
            df (pd.DataFrame): the data to index
            key (any): key of bucket to retrieve

        Returns:
            rows in bucket
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

    Using this is like using no index at all. Useful for when there are
    not too many rows.
    """

    def _key_ind_map(self, df: pd.DataFrame) -> dict:
        return {0: df.index.tolist()}


class ColumnsIndex(BaseIndex):
    """Split data into multiple buckets based on one or more columns.
    """

    def __init__(self, cols: str or list[str]) -> None:
        """Creates a new instance of ColumnsIndex

        Args:
            cols (str or list of str): column names to index

        Returns:
            no value
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
    - if `combine_keys` is False then key lists are simply concatenated together.
    Which mean a more lax pairing outcome.
    - if `combine_keys` is True then final key lists is the product of all key lists.
    Thus mean a stricter pairing outcome.
    """

    def __init__(self, indices: list[Type[BaseIndex]], combine_keys: bool = False) -> None:
        """Creates a new instance of MultiIndex

        Args:
            indices (list of index): the indices to combine
            combine_keys (bool): whether final key list should be the catersian
                product of all key lists

        Returns:
            no value
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
