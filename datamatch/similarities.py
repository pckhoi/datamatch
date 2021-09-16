"""
When given a pair of values, a similarity class produces a similarity score that ranges between 0 and 1.
A similarity score of 1 means the two values are completely identical while 0 means there are no similarities.

Note that these classes only compute similarity scores between scalar values or native Python objects such as
:class:`datetime.datetime`, not the entire row (which is handled by :class:`ThresholdMatcher`).
"""

from datetime import datetime
from Levenshtein import ratio, jaro_winkler
from unidecode import unidecode


class StringSimilarity(object):
    """Computes a similarity score between two strings using Levenshtein distance.
    """

    def sim(self, a: str, b: str):
        """Returns a similarity score.

        :param a: The left string.
        :type a: :obj:`str`

        :param b: The right string.
        :type b: :obj:`str`

        :return: The similarity score.
        :rtype: :obj:`float`
        """
        return ratio(unidecode(a), unidecode(b))


class JaroWinklerSimilarity(object):
    """Similar to :class:`StringSimilarity` but gives extra weight to common prefixes.

    This class is very good at matching people's names because mistaking the first
    letter in a person's name should be a rare event.
    """

    def __init__(self, prefix_weight=0.1):
        """
        :param prefix_weight: The extra weight given to common prefixes, defaults to 0.1.
        :type prefix_weight: :obj:`float`
        """
        self._prefix_weight = prefix_weight

    def sim(self, a: str, b: str):
        """Returns a similarity score.

        :param a: The left string.
        :type a: :obj:`str`

        :param b: The right string.
        :type b: :obj:`str`

        :return: The similarity score.
        :rtype: :obj:`float`
        """
        return jaro_winkler(unidecode(a), unidecode(b), self._prefix_weight)


class AbsoluteNumericalSimilarity(object):
    """Computes similarity score between two numbers, extrapolated from a maximum absolute difference.

    Maximum absolute difference **d_max** (greater than 0) is the maximum tolerated difference
    between two numbers regardless of their actual values. If the difference between the two values
    are less than **d_max** then the similarity score between two values `a` and `b` is
    ``1.0 - abs(a - b) / d_max``. Otherwise, the score is 0.

    Implementation follows strategy for numerical values given in the Data Matching book [1]_
    """

    def __init__(self, d_max: float) -> None:
        """
        :param d_max: The maximum absolute difference.
        :type d_max: :obj:`float`
        """
        self._d_max = d_max

    def sim(self, a: float or int, b: float or int) -> float:
        """Returns a similarity score.

        :param a: The left number.
        :type a: :obj:`float` or :obj:`int`

        :param b: The right number.
        :type b: :obj:`float` or :obj:`int`

        :return: The similarity score.
        :rtype: :obj:`float`
        """
        d = abs(a - b)
        if d < self._d_max:
            return 1 - d / self._d_max
        return 0


class RelativeNumericalSimilarity(object):
    """Computes similarity score between two numbers, extrapolated from a maximum percentage difference.

    This class serves a similar purpose to :class:`AbsoluteNumericalSimilarity` but is more dependent on
    the actual values being compared.

    Percentage difference `pc` between two values `a` and `b` is defined as
    ``abs(a - b) / max(abs(a), abs(b)) * 100``.

    Maximum percentage difference **pc_max** (0 < pc_max < 100) is the maximum tolerated percentage
    difference between the two numbers. If the percentage difference `pc` is less than **pc_max** then
    the similarity score is calculated with ``1.0 - pc / pc_max``. Otherwise, the score is 0.

    Implementation follows strategy for numerical values given in the Data Matching book [1]_
    """

    def __init__(self, pc_max: int) -> None:
        """
        :param pc_max: The maximum percentage difference.
        :type pc_max: :obj:`int`
        """
        self._pc_max = pc_max

    def sim(self, a: float or int, b: float or int) -> float:
        """Returns a similarity score

        :param a: The left number.
        :type a: :obj:`float` or :obj:`int`

        :param b: The right number.
        :type b: :obj:`float` or :obj:`int`

        :return: The similarity score.
        :rtype: :obj:`float`
        """
        d = abs(a - b)
        pc = d / max(abs(a), abs(b)) * 100
        if pc < self._pc_max:
            return 1 - pc / self._pc_max
        return 0


class DateSimilarity(object):
    """Computes similarity score between two dates, extrapolated from a maximum absolute difference in days.

    Maximum absolute difference in days **d_max** is the maximum tolerated difference in days between two dates.
    Similar to :class:`AbsoluteNumericalSimilarity` if both dates `a` and `b` are less than **d_max** days
    apart then the similarity score is ``1 - (a - b) / d_max``.

    If however ``(a - b) >= d_max`` then we employs two alternative strategies to hedge against typos:

    - If the year values are the same but the month and day values are swapped, then the similarity score is 0.5.

    - | The last resort is to write each date in `YYYYMMDD` format and computes the similarity score between
      | two strings.

    Implementation follows strategy for date/time given in the Data Matching book [2]_
    """

    def __init__(self, d_max=30):
        """
        :param d_max: Dates that are less than this number of days apart will have similarity score as
            ``1 - <difference in days> / d_max``. For dates that are further apart, this class employs
            alternative methods to compute the similarity score to hedge against typos. This defaults to 30.
        :type d_max: :obj:`int`
        """
        self._d_max = d_max

    def sim(self, a: datetime, b: datetime):
        """Returns a similarity score.

        :param a: The left date.
        :type a: :obj:`datetime.datetime`

        :param b: The right date.
        :type b: :obj:`datetime.datetime`

        :return: The similarity score.
        :rtype: :obj:`float`
        """
        d = a - b
        if b > a:
            d = b - a
        if d.days < self._d_max:
            return 1 - d.days / self._d_max
        if a.year == b.year and a.month == b.day and a.day == b.month:
            return 0.5
        if a.year == b.year and a.day == b.day:
            return ratio(a.strftime("%Y%m%d"), b.strftime("%Y%m%d"))
        return 0
