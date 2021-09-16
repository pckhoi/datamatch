"""
A variator creates variations of a single record. :class:`ThresholdMatcher` can produce different similarity
scores for different variations of the same record, discarding all but the highest score in the final result.
This is very useful in situations where values are not put into the correct columns (e.g. when a person's first
name and last name are swapped).
"""

from typing import Iterator

import pandas as pd


class Variator(object):
    """Base class of all variator classes.

    Sub-class should override method :meth:`variations`.

    This class also serves as a no-op variator. It simply returns the record as is.
    """

    def variations(self, sr: pd.Series) -> Iterator[pd.Series]:
        """Returns variations of the input record.

        :param sr: The input record.
        :type sr: :class:`pandas:pandas.Series`

        :rtype: :ref:`Iterator <python:typeiter>` of :class:`pandas:pandas.Series`
        """
        yield sr


class Swap(Variator):
    """Produces variations by swapping values between two columns"""

    def __init__(self, column_a: str, column_b: str) -> None:
        """
        :param column_a: The left column.
        :type column_a: :obj:`str`

        :param column_b: The right column.
        :type column_b: :obj:`str`
        """
        super().__init__()
        self._col_a = column_a
        self._col_b = column_b

    def variations(self, sr: pd.Series) -> Iterator[pd.Series]:
        """
        .. hiding method's docstring

        :meta private:
        """
        yield sr
        if not (pd.isna(sr[self._col_a]) and pd.isna(sr[self._col_b])) and (sr[self._col_a] != sr[self._col_b]):
            sr = sr.copy(True)
            v = sr[self._col_a]
            sr.loc[self._col_a] = sr[self._col_b]
            sr.loc[self._col_b] = v
            yield sr
