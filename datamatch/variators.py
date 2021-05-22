from typing import Iterator

import pandas as pd


class Variator(object):
    """Base class of all variator classes

    A variator creates variations from a single record. This class
    serves as a noop variator. It simply return the record as is.
    """

    def variations(self, sr: pd.Series) -> Iterator[pd.Series]:
        """Produces variations of the same record

        Args:
            sr (pd.Series):
                the record to retun variations for

        Returns:
            an iterator of variations
        """
        yield sr


class Swap(Variator):
    """Produces variations by swapping column values"""

    def __init__(self, column_a: str, column_b: str) -> None:
        """Creates new instance of Swap

        Args:
            column_a (str):
                first column name
            column_b (str):
                last column name

        Returns:
            no value
        """
        super().__init__()
        self._col_a = column_a
        self._col_b = column_b

    def variations(self, sr: pd.Series) -> Iterator[pd.Series]:
        yield sr
        if not (pd.isna(sr[self._col_a]) and pd.isna(sr[self._col_b])) and (sr[self._col_a] != sr[self._col_b]):
            sr = sr.copy(True)
            v = sr[self._col_a]
            sr.loc[self._col_a] = sr[self._col_b]
            sr.loc[self._col_b] = v
            yield sr
