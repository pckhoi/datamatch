"""
A filter discards pairs from the matching process. An index, which dictates which pair can be compared,
does the opposite. They are both employed to increase matching performance.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseFilter(ABC):
    """Base class of all filter classes.

    Sub-class should implement the :meth:`valid` method.
    """

    @abstractmethod
    def valid(self, a: pd.Series, b: pd.Series) -> bool:
        """Returns true if a pair of records is valid (can be matched).

        :param a: the left record.
        :type a: :class:`pandas:pandas.Series`

        :param b: the right record.
        :type b: :class:`pandas:pandas.Series`

        :return: Whether these two records can be matched.
        :rtype: :obj:`bool`
        """
        raise NotImplementedError()


class DissimilarFilter(BaseFilter):
    """Eliminates pairs with the same value for a specific field.
    """

    def __init__(self, col: str, ignore_key_error: bool = False) -> None:
        """
        :param col: The column to check.
        :type col: :obj:`str`

        :param ignore_key_error: When set to True, if the column is not found, acts like a no-op
            filter instead of raising KeyError.
        :param ignore_key_error: :obj:`bool`
        """
        super().__init__()
        self._col = col
        self._ignore_key_error = ignore_key_error

    def valid(self, a: pd.Series, b: pd.Series) -> bool:
        """
        .. hiding method's docstring

        :meta private:
        """
        try:
            val_a = a[self._col]
            val_b = b[self._col]
        except KeyError:
            if self._ignore_key_error:
                return True
            raise
        if pd.isnull(val_a) or pd.isnull(val_b):
            return True
        return val_a != val_b


class NonOverlappingFilter(BaseFilter):
    """Eliminates pairs with overlapping ranges.

    This is usually used over time ranges, which ensures time exclusivity of a record.
    """

    def __init__(self, start: str, end: str) -> None:
        """
        Both start and end columns must be of the same type and must be comparable.

        e.g. `df[end] < df[start]` should produce a boolean series.

        :param start: the range start column.
        :type start: :obj:`str`

        :param end: the range end column.
        :type end: :obj:`str`
        """
        super().__init__()
        self._start = start
        self._end = end

    def valid(self, a: pd.Series, b: pd.Series) -> bool:
        """
        .. hiding method's docstring

        :meta private:
        """
        return a[self._end] < b[self._start] or a[self._start] > b[self._end]
