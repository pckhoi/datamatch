from Levenshtein import ratio, jaro_winkler
from unidecode import unidecode


class StringSimilarity(object):
    """Computes similarity score between 2 strings using Levenshtein distance"""

    def __init__(self):
        pass

    def sim(self, a, b):
        return ratio(unidecode(a), unidecode(b))


class JaroWinklerSimilarity(object):
    """Similar to StringSimilarity but give extra weight to common prefix

    Makes this class very good for matching people's name because mistaking
    the first letter in a person's name should be very rare indeed.
    """

    def __init__(self, prefix_weight=0.1):
        self._prefix_weight = prefix_weight

    def sim(self, a, b):
        return jaro_winkler(unidecode(a), unidecode(b), self._prefix_weight)


class DateSimilarity(object):
    """Computes similarity score between 2 dates

    If score is 1 then 2 dates are exactly the same. Except in special
    conditions, if both days are more than `days_max_diff` (default to 30)
    days apart then score will be 0.

    Score is computed differently in special conditions to combat typo:
    - month and day digits are swapped, in which case similarity score
    will be 0.5
    - years and days are the same but month digits are different, in which
    case the date is coverted to a string and string similarity score is
    returned
    """

    def __init__(self, days_max_diff=30):
        self._days_max_diff = days_max_diff

    def sim(self, a, b):
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
