from unittest import TestCase
import itertools

import pandas as pd
from pandas.testing import assert_series_equal

from datamatch.variators import Variator, Swap


class BaseVariatorTestCase(TestCase):
    def assert_series_list_equal(self, list_a, list_b):
        self.assertEqual(len(list_a), len(list_b))
        for rec_a, rec_b in itertools.zip_longest(list_a, list_b):
            assert_series_equal(rec_a, rec_b)


class VariatorTestCase(BaseVariatorTestCase):
    def test_variations(self):
        sr = pd.Series([1, 2, 3])
        v = Variator()
        self.assert_series_list_equal(list(v.variations(sr)), [sr])


class SwapTestCase(BaseVariatorTestCase):
    def test_variations(self):
        sr1 = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
        sr2 = pd.Series([2, 2, 3], index=['x', 'y', 'z'])

        v = Swap('x', 'y')

        self.assert_series_list_equal(list(v.variations(sr1)), [
            sr1, pd.Series([2, 1, 3], index=['x', 'y', 'z'])
        ])

        self.assert_series_list_equal(list(v.variations(sr2)), [sr2])
