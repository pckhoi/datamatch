"""
A similarity class when given a pair of values, produces a similarity score that ranges between 0 and 1.
A similarity score of 1 means the 2 values are completely identical while 0 means there are no similarities.

Note that these classes only compute similarity scores between scalar values or native Python objects such as
:class:`datetime.datetime`, not the entire row (which is handled by :class:`ThresholdMatcher`).
"""

from datetime import date, datetime
from Levenshtein import ratio, jaro_winkler
from unidecode import unidecode


class StringSimilarity(object):
    """Computes similarity score between 2 strings using Levenshtein distance"""

    def sim(self, a: str, b: str):
        """Returns a similarity score

        :param a: The left string
        :type a: :obj:`str`

        :param b: The right string
        :type b: :obj:`str`

        :return: The similarity score
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
        :param prefix_weight: The extra weight given to common prefixes, defaults to 0.1
        :type prefix_weight: :obj:`float`
        """
        self._prefix_weight = prefix_weight

    def sim(self, a: str, b: str):
        """Returns a similarity score

        :param a: The left string
        :type a: :obj:`str`

        :param b: The right string
        :type b: :obj:`str`

        :return: The similarity score
        :rtype: :obj:`float`
        """
        return jaro_winkler(unidecode(a), unidecode(b), self._prefix_weight)


class AbsoluteNumericalSimilarity(object):
    """Computes similarity score between two numbers, extrapolated from a maximum absolute difference

    Maximum absolute difference **d_max** (greater than 0) is the maximum tolerated difference
    between two numbers regardless of their actual values. If the difference between the two values
    are less than **d_max** then the similarity score between two values `a` and `b` is
    ``1.0 - abs(a - b) / d_max``. Otherwise the score is 0.
    """

    def __init__(self, d_max: float) -> None:
        """
        :param d_max: The maximum absolute difference
        :type d_max: :obj:`float`
        """
        self._d_max = d_max

    def sim(self, a: float or int, b: float or int) -> float:
        """Returns a similarity score

        :param a: The left number
        :type a: :obj:`float` or :obj:`int`

        :param b: The right number
        :type b: :obj:`float` or :obj:`int`

        :return: The similarity score
        :rtype: :obj:`float`
        """
        d = abs(a - b)
        if d < self._d_max:
            return 1 - d / self._d_max
        return 0


class RelativeNumericalSimilarity(object):
    """Computes similarity score between two numbers, extrapolated from a maximum percentage difference

    This class serves similar purpose to :class:`AbsoluteNumericalSimilarity` but is more dependent on
    the actual values being compared.

    Percentage difference `pc` between two values `a` and `b` is defined as
    ``abs(a - b) / max(abs(a), abs(b)) * 100``.

    Maximum percentage difference **pc_max** (0 < pc_max < 100) is the maximum tolerated percentage
    difference between the two numbers. If the percentage difference `pc` is less than **pc_max** then
    the similarity score is calculated with ``1.0 - pc / pc_max``. Otherwise the score is 0.
    """

    def __init__(self, pc_max: int) -> None:
        """
        :param pc_max: The maximum percentage difference
        :type pc_max: :obj:`int`
        """
        self._pc_max = pc_max

    def sim(self, a: float or int, b: float or int) -> float:
        """Returns a similarity score

        :param a: The left number
        :type a: :obj:`float` or :obj:`int`

        :param b: The right number
        :type b: :obj:`float` or :obj:`int`

        :return: The similarity score
        :rtype: :obj:`float`
        """
        d = abs(a - b)
        pc = d / max(abs(a), abs(b)) * 100
        if pc < self._pc_max:
            return 1 - pc / self._pc_max
        return 0


class DateSimilarity(object):
    """Computes similarity score between 2 dates

    This is how similarity score is computed:

    - | If both dates are less than **days_max_diff** days apart then the similarity score is
      | ``1 - <difference in days> / days_max_diff``, otherwise

    - If the year digits are the same but the month and day digits are swapped, then the similarity score is 0.5.

    - If both measures fail then the last resort is to write each date in `YYYYMMDD` format and returns Levenshtein distance between them. 
    """

    def __init__(self, days_max_diff=30):
        """
        :param days_max_diff: Dates that are less than this number of days apart will have similarity score as
            ``1 - <difference in days> / days_max_diff``. For dates that are further apart, this class employs
            alternative methods to compute the similarity score to hedge against typos. This defaults to 30.
        :type days_max_diff: :obj:`int`
        """
        self._days_max_diff = days_max_diff

    def sim(self, a: datetime, b: datetime):
        """Returns a similarity score

        :param a: The left date
        :type a: :obj:`datetime.datetime`

        :param b: The right date
        :type b: :obj:`datetime.datetime`

        :return: The similarity score
        :rtype: :obj:`float`
        """
        d = a - b
        if b > a:
            d = b - a
        if d.days < self._days_max_diff:
            return 1 - d.days / self._days_max_diff
        if a.year == b.year and a.month == b.day and a.day == b.month:
            return 0.5
        if a.year == b.year and a.day == b.day:
            return ratio(a.strftime("%Y%m%d"), b.strftime("%Y%m%d"))
        return 0
