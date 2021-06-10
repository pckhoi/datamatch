import pandas as pd


class BaseFilter(object):
    """Base class of all filter classes

    A filter eliminates pairs out of matching process. It is similar
    to an index in that they are both employed to reduce matching
    time but whereas an index white-lists what records can be matched,
    a filter acts like a black-listing mechanism.
    """

    def valid(self, a: pd.Series, b: pd.Series) -> bool:
        """Returns whether a pair of record is valid (should be matched).

        Args:
            a (Series): left record
            b (Series): right record

        Returns:
            whether this pair of records should be matched
        """
        raise NotImplementedError()


class DissimilarFilter(BaseFilter):
    """Eliminates pairs with the same value for a specific field.
    """

    def __init__(self, field: str) -> None:
        """Creates a new instance of DissimilarFilter

        Args:
            field (str):
                name of field to check

        Returns:
            no value
        """
        super().__init__()
        self._field = field

    def valid(self, a: pd.Series, b: pd.Series) -> bool:
        val_a = a[self._field]
        val_b = b[self._field]
        if pd.isnull(val_a) or pd.isnull(val_b):
            return True
        return val_a != val_b


class NonOverlappingFilter(BaseFilter):
    """Eliminates pairs with overlapping number range.
    """

    def __init__(self, start: str, end: str) -> None:
        """Creates a new instance of NonOverlappingFilter

        Args:
            start (str):
                field that contain start value of range. It must never
                contain NA value.
            end (str):
                field that contain end value of range. It must never
                contain NA value.

        Returns:
            no value  
        """
        super().__init__()
        self._start = start
        self._end = end

    def valid(self, a: pd.Series, b: pd.Series) -> bool:
        return a[self._end] < b[self._start] or a[self._start] > b[self._end]
