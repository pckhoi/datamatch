from unittest import TestCase

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from datamatch.pairers import MatchPairer, DeduplicatePairer
from datamatch.indices import ColumnsIndex


class BasePairerTestCase(TestCase):
    def assert_pair_equal(self, pair_a, pair_b):
        self.assertEqual(len(pair_a), 2)
        self.assertEqual(len(pair_b), 2)
        for i, (name, ser) in enumerate(pair_a):
            self.assertEqual(name, pair_b[i][0])
            assert_series_equal(ser, pair_b[i][1], check_names=False)


class MatchPairerTestCase(BasePairerTestCase):
    def test_pairs(self):
        columns = ['x', 'y', 'z']
        dfa = pd.DataFrame([
            [1, 'a', 'b'],
            [2, 'c', 'd'],
            [3, 'e', 'f']
        ], columns=columns)
        dfb = pd.DataFrame([
            [1, 'q', 'w'],
            [4, 'z', 'x'],
            [2, 'a', 's'],
        ], columns=columns)

        pairer = MatchPairer(dfa, dfb, ColumnsIndex(['x']))
        assert_frame_equal(pairer.frame_a, dfa)
        assert_frame_equal(pairer.frame_b, dfb)

        pairs = list(pairer.pairs())
        self.assertEqual(len(pairs), 2)
        self.assert_pair_equal(pairs[0], (
            (0, pd.Series([1, 'a', 'b'], index=columns)),
            (0, pd.Series([1, 'q', 'w'], index=columns)),
        ))
        self.assert_pair_equal(pairs[1], (
            (1, pd.Series([2, 'c', 'd'], index=columns)),
            (2, pd.Series([2, 'a', 's'], index=columns)),
        ))


class DeduplicatePairerTestCase(BasePairerTestCase):
    def test_pairs(self):
        columns = ['x', 'y', 'z']
        df = pd.DataFrame([
            [1, 'a', 'b'],
            [2, 'c', 'd'],
            [3, 'e', 'f'],
            [1, 'q', 'w'],
            [4, 'z', 'x'],
            [2, 'a', 's'],
        ], columns=columns)

        pairer = DeduplicatePairer(df, ColumnsIndex(['x']))
        assert_frame_equal(pairer.frame_a, df)
        assert_frame_equal(pairer.frame_b, df)

        pairs = list(pairer.pairs())
        self.assertEqual(len(pairs), 2)
        self.assert_pair_equal(pairs[0], (
            (0, pd.Series([1, 'a', 'b'], index=columns)),
            (3, pd.Series([1, 'q', 'w'], index=columns)),
        ))
        self.assert_pair_equal(pairs[1], (
            (1, pd.Series([2, 'c', 'd'], index=columns)),
            (5, pd.Series([2, 'a', 's'], index=columns)),
        ))
