from typing import Iterator

import pandas as pd


class BaseIndex(object):
    def keys(self, df: pd.DataFrame) -> set:
        raise NotImplementedError()

    def bucket(self, df: pd.DataFrame, key: any) -> pd.DataFrame:
        raise NotImplementedError()


class NoopIndex(BaseIndex):
    """
    This pair every record with every other record. It's like not using an index.
    """

    def __init__(self):
        pass

    def keys(self, df: pd.DataFrame) -> set:
        return set([0])

    def bucket(self, df: pd.DataFrame, key: any) -> pd.DataFrame:
        if key != 0:
            raise KeyError(key)
        return df


class ColumnsIndex(BaseIndex):
    """
    This pair records with the same value in specified columns.
    """

    def __init__(self, cols):
        self._cols = cols

    def keys(self, df: pd.DataFrame) -> set:
        return set(
            df[self._cols].drop_duplicates().to_records(index=False).tolist()
        )

    def _bool_index(self, df: pd.DataFrame, key: any):
        result = None
        for ind, val in enumerate(key):
            if result is None:
                result = df[self._cols[ind]] == val
            else:
                result = result & (df[self._cols[ind]] == val)
        return result

    def bucket(self, df: pd.DataFrame, key: any) -> pd.DataFrame:
        rows = df.loc[self._bool_index(df, key)]
        if len(rows.shape) == 1:
            rows = rows.to_frame().transpose()
        return rows
