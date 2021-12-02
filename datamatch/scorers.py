"""
A scorer decide how the similarity score for each pair is calculated.
"""

from abc import ABC, abstractmethod
import inspect
import math
from typing import Callable, Type

import pandas as pd


class RefuseToScoreException(Exception):
    """Raise to delegate scoring to a parent object
    """
    pass


ScoreFunc = Callable[[pd.Series, pd.Series], float]


class BaseScorer(ABC):
    """Base class for all scorer classes.

    Sub-class should implement the :meth:`score` method.
    """

    @abstractmethod
    def score(self, a: pd.Series, b: pd.Series) -> float:
        """Returns similarity score (0 <= sim <= 1) for a pair of records.

        :param a: The left record.
        :type a: :class:`pandas:pandas.Series`

        :param b: The right record.
        :type b: :class:`pandas:pandas.Series`

        :return: Similarity score.
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
    """Returns an arbitrary score if both records have the same value for a column.

    If the values are not equal or one of them is null then this scorer will raise :class:`RefuseToScoreException`.
    Therefore, this class should never be used on its own but always wrapped in either
    :class:`MaxScorer` or :class:`MinScorer`.
    """

    def __init__(self, column_name: str, score: float, ignore_key_error: bool = False) -> None:
        """
        :param column_name: The column to compare.
        :type column_name: :obj:`str`

        :param score: The score to return.
        :type score: :obj:`float`

        :param ignore_key_error: When set to True, if the column does not exist in either record,
            raise RefuseToScoreException (delegate to a parent scorer) instead of KeyError.
        :type ignore_key_error: :obj:`bool`
        """
        self._column = column_name
        self._default_score = score
        self._ignore_key_error = ignore_key_error

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        try:
            if pd.isnull(a[self._column]) or pd.isnull(b[self._column]):
                raise RefuseToScoreException("one of the values is null")
            elif a[self._column] == b[self._column]:
                return self._default_score
        except KeyError:
            if self._ignore_key_error:
                raise RefuseToScoreException(
                    "column does not exist in one of the record")
            else:
                raise
        raise RefuseToScoreException("values are not equal")


class MaxScorer(BaseScorer):
    """Returns the max value from the scores of all child scorers.
    """

    def __init__(self, scorers: list[Type[BaseScorer]]) -> None:
        """
        :param scorers: The children classes.
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
        :param scorers: The children classes.
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


class AlterScorer(BaseScorer):
    """Modifies the score for pairs with the same given values
    """

    def __init__(self, scorer: Type[BaseScorer], values: pd.Series, alter: Callable[[int], int]) -> None:
        """
        :param scorer: The wrapped scorer.
        :type scorer: :class:`BaseScorer` subclass.
        :param values:
            for each pair, if both rows have index in this series and their values are the same then
            call `alter` to modify the final score.
        :type values: :class:`pandas:pandas.Series`
        :param alter: callback to modify the final score.
        :type alter: Callable[[int], int]
        """
        self._scorer = scorer
        self._values = values
        self._alter = alter

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        score = self._scorer.score(a, b)
        try:
            if self._values[a.name] == self._values[b.name]:
                return self._alter(score)
        except KeyError:
            pass
        return score


class FuncScorer(BaseScorer):
    """Scores pairs by calling the given callback
    """

    def __init__(self, cb: ScoreFunc) -> None:
        """
        :param cb: Callback to calculate score
        :type cb: Callable[[:class:`pandas:pandas.Series`, :class:`pandas:pandas.Series`], :obj:`float`]
        """
        self._cb = cb

    def score(self, a: pd.Series, b: pd.Series) -> float:
        """
        .. hiding method's docstring

        :meta private:
        """
        return self._cb(a, b)
