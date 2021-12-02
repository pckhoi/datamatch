"""
A pairer produces pairs of records for matching. Most users should never need
to touch a pairer but this is exposed anyway for the sake of customization.

There are two pairers corresponding to two strategies:

- | :class:`MatchPairer`: Takes in two datasets and produces pairs of records
  | such that each pair contain one record from one dataset and one record from
  | the other dataset. This pairer is utilized when :class:`ThresholdMatcher`
  | is given two datasets. It is useful for matching records between
  | two datasets.

- | :class:`DeduplicatePairer`: Takes in one dataset and produces pairs of records
  | each having only records from the input dataset. This pairer is utilized
  | when :class:`ThresholdMatcher` is given only one dataset. It is
  | useful for deduplication tasks.
"""

import itertools
from typing import Iterator, Type
from abc import ABC, abstractmethod

import pandas as pd

from .indices import BaseIndex


class BasePairer(ABC):
    """Abstract base class for all pairer classes.

    Sub-class must implement :meth:`frame_a`, :meth:`frame_b`, and :meth:`pairs`.

    :meth:`frame_a` should produce the left set of records, :meth:`frame_b`
    should produce the right set of records, whereas :meth:`pairs` should produce
    pairs of records (one from :meth:`frame_a`, one from :meth:`frame_b`).
    """

    @property
    @abstractmethod
    def frame_a(self) -> pd.DataFrame:
        """Returns the left set of records.

        :rtype: :class:`pandas:pandas.DataFrame`
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def frame_b(self) -> pd.DataFrame:
        """Returns the right set of records.

        :rtype: :class:`pandas:pandas.DataFrame`
        """
        raise NotImplementedError()

    @abstractmethod
    def pairs(self) -> Iterator:
        """Returns an iterator over pairs of records that should be compared.

        Each pair is a tuple of left record and right record. Each record is a tuple
        of two elements: the row index and the row data which is a :class:`pandas:pandas.Series`.

        :rtype: :ref:`Iterator <python:typeiter>`
        """
        raise NotImplementedError()


class MatchPairer(BasePairer):
    """Pairs records from two frames using the provided index.
    """

    def __init__(self, dfa: pd.DataFrame, dfb: pd.DataFrame, index: Type[BaseIndex]) -> None:
        """
        :param dfa: The left dataset.
        :type dfa: :class:`pandas:pandas.DataFrame`

        :param dfb: The right dataset.
        :type dfb: :class:`pandas:pandas.DataFrame`

        :param index: The index to divide datasets into buckets.
        :type index: sub-class of :class:`datamatch.indices.BaseIndex`
        """
        self._index = index
        if dfa.index.duplicated().any() or dfb.index.duplicated().any():
            raise ValueError(
                "Dataframe index contains duplicates. Both frames need to have index free of duplicates."
            )
        if set(dfa.columns) != set(dfb.columns):
            raise ValueError(
                "Dataframe columns are not equal."
            )

        self._dfa = dfa
        self._dfb = dfb

    @property
    def frame_a(self) -> pd.DataFrame:
        """
        .. hiding method's docstring

        :meta private:
        """
        return self._dfa

    @property
    def frame_b(self) -> pd.DataFrame:
        """
        .. hiding method's docstring

        :meta private:
        """
        return self._dfb

    def pairs(self) -> Iterator:
        """
        .. hiding method's docstring

        :meta private:
        """
        keys_a = self._index.keys(self._dfa)
        keys_b = self._index.keys(self._dfb)
        keys = list(keys_a.intersection(keys_b))
        for key in keys:
            rows_a = self._index.bucket(self._dfa, key)
            rows_b = self._index.bucket(self._dfb, key)
            for idx_a, ser_a in rows_a.iterrows():
                for idx_b, ser_b in rows_b.iterrows():
                    yield (idx_a, ser_a), (idx_b, ser_b)


class DeduplicatePairer(BasePairer):
    """Pairs records from a single frame for deduplication.

    As this class is only initialized with a single frame, both
    `frame_a` and `frame_b` returns this same frame.
    """

    def __init__(self, df: pd.DataFrame, index: Type[BaseIndex]) -> None:
        """
        :param df: The dataset to deduplicate.
        :type df: :class:`pandas:pandas.DataFrame`

        :param index: The index to divide datasets into buckets.
        :type index: sub-class of :class:`datamatch.indices.BaseIndex`
        """
        self._index = index
        if df.index.duplicated().any():
            raise ValueError(
                "Dataframe index contains duplicates."
            )
        self._df = df

    @property
    def frame_a(self) -> pd.DataFrame:
        """
        .. hiding method's docstring

        :meta private:
        """
        return self._df

    @property
    def frame_b(self) -> pd.DataFrame:
        """
        .. hiding method's docstring

        :meta private:
        """
        return self._df

    def pairs(self) -> Iterator:
        """
        .. hiding method's docstring

        :meta private:
        """
        keys = self._index.keys(self._df)
        for key in keys:
            rows = self._index.bucket(self._df, key)
            for row_a, row_b in itertools.combinations(rows.iterrows(), 2):
                yield row_a, row_b
