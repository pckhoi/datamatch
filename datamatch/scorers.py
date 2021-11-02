"""
A scorer decide how the similarity score for each pair is calculated.
"""

from abc import ABC, abstractmethod
import math
from typing import Type

import pandas as pd


class RefuseToScoreException(Exception):
    """Raise to delegate scoring to a parent object
    """
    pass


class BaseScorer(ABC):
    """Base class for all scorer classes.

    Sub-class should implement the :meth:`score` method.
    """

    @abstractmethod
    def score(self, a: pd.Series, b: pd.Series) -> float:
        """Returns similarity score (0 <= sim <= 1) for a pair of records.

        :param a: the left record.
        :type a: :class:`pandas:pandas.Series`

        :param b: the right record.
        :type b: :class:`pandas:pandas.Series`

        :return: similarity score.
        :rtype: :obj:`float`
        """
        raise NotImplementedError()


class SimSumScorer(BaseScorer):
    """Returns the sum of similarity values of all fields.
    """

    def __init__(self, fields: dict) -> None:
        """
        :param fields: The mapping between field name and :ref:`similarity class <Similarities>` to use.
        :type fields: :obj:`dict` of similarity classes
        """
        super().__init__()
        self._fields = fields

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        sim_vec = dict()
        for k, scls in self._fields.items():
            if pd.isnull(a[k]) or pd.isnull(b[k]):
                sim_vec[k] = 0
            else:
                sim_vec[k] = scls.sim(a[k], b[k])
        return math.sqrt(
            sum(v * v for v in sim_vec.values()) / len(self._fields))


class AbsoluteScorer(BaseScorer):
    """Returns an arbitrary score if both records has the same value for a column.

    If the values are not equal or is null then this scorer will raise :class:`RefuseToScoreException`.
    Therefore, this class should never be used on its own but always wrapped in either
    :class:`MaxScorer` or :class:`MinScorer`.
    """

    def __init__(self, column_name: str, score: float) -> None:
        """
        :param column_name: The column to compare.
        :type column_name: :obj:`str`

        :param score: The score to return.
        :type score: :obj:`float`
        """
        self._column = column_name
        self._default_score = score

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        if pd.isnull(a[self._column]) or pd.isnull(b[self._column]):
            raise RefuseToScoreException("one of the values is null")
        elif a[self._column] == b[self._column]:
            return self._default_score
        raise RefuseToScoreException("values are not equal")


class MaxScorer(BaseScorer):
    """Returns the max value from the scores of all child scorers.
    """

    def __init__(self, scorers: list[Type[BaseScorer]]) -> None:
        """
        :param scorers: The scorer classes to use.
        :type scorers: list of :class:`BaseScorer` subclasses.
        """
        self._scorers = scorers

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        val = None
        for scorer in self._scorers:
            try:
                n = scorer.score(a, b)
            except RefuseToScoreException:
                continue
            if val is None or n > val:
                val = n
        if val is None:
            raise RefuseToScoreException("all children refuse to score")
        return val


class MinScorer(BaseScorer):
    """Returns the min value from the scores of all child scorers.
    """

    def __init__(self, scorers: list[Type[BaseScorer]]) -> None:
        """
        :param scorers: The scorer classes to use.
        :type scorers: list of :class:`BaseScorer` subclasses.
        """
        self._scorers = scorers

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        val = None
        for scorer in self._scorers:
            try:
                n = scorer.score(a, b)
            except RefuseToScoreException:
                continue
            if val is None or n < val:
                val = n
        if val is None:
            raise RefuseToScoreException("all children refuse to score")
        return val