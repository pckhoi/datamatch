import pandas as pd


class BaseIndex(object):
    """Base class for all index classes.

    This doesn't have any functionality but simple serve as an interface for all
    subclasses to implement.
    """

    def keys(self, df: pd.DataFrame) -> set:
        """Returns a set of keys that should be used to retrieve buckets

        Args:
            df (pd.DataFrame): the data to index

        Returns:
            a set of bucket keys
        """
        raise NotImplementedError()

    def bucket(self, df: pd.DataFrame, key: any) -> pd.DataFrame:
        """Retrieve a bucket given the original data and a bucket key

        Args:
            df (pd.DataFrame): the data to index
            key (any): key of bucket to retrieve

        Returns:
            rows in bucket
        """
        raise NotImplementedError()


class NoopIndex(BaseIndex):
    """Returns all data as a single bucket.

    Using this is like using no index at all. Useful for when there are
    not too many rows.
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
    """Split data into multiple buckets based on one or more columns.
    """

    def __init__(self, cols: str or list(str)):
        if type(cols) is str:
            self._cols = [cols]
        else:
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
