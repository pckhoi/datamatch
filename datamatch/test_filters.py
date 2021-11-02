from unittest import TestCase

import pandas as pd

from datamatch.filters import DissimilarFilter, NonOverlappingFilter


class DissimilarFilterTestCase(TestCase):
    def test_valid(self):
        f = DissimilarFilter('agency')
        index = ['agency', 'uid']
        self.assertFalse(f.valid(
            pd.Series(['slidell pd', '123'], index=index),
            pd.Series(['slidell pd', '456'], index=index)
        ))
        self.assertTrue(f.valid(
            pd.Series(['gretna pd', '123'], index=index),
            pd.Series(['slidell pd', '456'], index=index)
        ))

    def test_ignore_key_error(self):
        index = ['agency', 'uid']
        series_a = pd.Series(['slidell pd', '123'], index=index)
        series_b = pd.Series(['slidell pd', '456'], index=index)
        self.assertRaises(
            KeyError,
            lambda: DissimilarFilter('first').valid(series_a, series_b)
        )
        self.assertTrue(DissimilarFilter(
            'first', ignore_key_error=True
        ).valid(series_a, series_b))


class NonOverlappingFilterTestCase(TestCase):
    def test_valid(self):
        f = NonOverlappingFilter('start', 'end')
        index = ['uid', 'start', 'end']
        self.assertFalse(f.valid(
            pd.Series(['123', 0, 4], index=index),
            pd.Series(['456', 3, 6], index=index)
        ))
        self.assertFalse(f.valid(
            pd.Series(['123', 10, 14], index=index),
            pd.Series(['456', 3, 16], index=index)
        ))
        self.assertFalse(f.valid(
            pd.Series(['123', 0, 4], index=index),
            pd.Series(['456', 3, 3], index=index)
        ))
        self.assertFalse(f.valid(
            pd.Series(['123', 10, 14], index=index),
            pd.Series(['456', 3, 11], index=index)
        ))
        self.assertTrue(f.valid(
            pd.Series(['123', 10, 10], index=index),
            pd.Series(['456', 3, 6], index=index)
        ))
        self.assertTrue(f.valid(
            pd.Series(['123', 10, 12], index=index),
            pd.Series(['456', 13, 16], index=index)
        ))
        self.assertTrue(f.valid(
            pd.Series(['123', 0, 10], index=index),
            pd.Series(['456', 23, 26], index=index)
        ))
