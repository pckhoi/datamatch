import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import itertools

from .indices import MultiIndex, NoopIndex, ColumnsIndex


class BaseIndexTestCase(unittest.TestCase):
    def assert_pairs_equal(self, pair_a, pair_b):
        df1, df2 = pair_a
        df3, df4 = pair_b
        assert_frame_equal(df1, df3)
        assert_frame_equal(df2, df4)

    def assert_pairs_list_equal(self, list_a, list_b):
        self.assertEqual(len(list_a), len(list_b))
        for pair_a, pair_b in itertools.zip_longest(list_a, list_b):
            self.assert_pairs_equal(pair_a, pair_b)


class TestNoopIndex(BaseIndexTestCase):
    def test_index(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        idx = NoopIndex()
        keys = idx.keys(df)
        self.assertEqual(keys, set([0]))
        assert_frame_equal(idx.bucket(df, 0), df)


class TestColumnsIndex(BaseIndexTestCase):
    def test_index(self):
        cols = ["c", "d"]
        df = pd.DataFrame(
            [[1, 2], [2, 4], [3, 4]], index=["x", "y", "z"], columns=cols)
        idx = ColumnsIndex(["c"])
        keys = idx.keys(df)
        self.assertEqual(keys, set([(1,), (2,), (3,)]))
        assert_frame_equal(
            idx.bucket(df, (1,)),
            pd.DataFrame([[1, 2]], index=["x"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (2,)),
            pd.DataFrame([[2, 4]], index=["y"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (3,)),
            pd.DataFrame([[3, 4]], index=["z"], columns=cols)
        )

    def test_multi_columns(self):
        cols = ["c", "d"]
        df = pd.DataFrame(
            [[1, 2], [2, 4], [3, 4]], index=["z", "x", "c"], columns=cols)
        idx = ColumnsIndex(["c", "d"])
        keys = idx.keys(df)
        self.assertEqual(keys, set([(1, 2), (2, 4), (3, 4)]))
        assert_frame_equal(
            idx.bucket(df, (1, 2)),
            pd.DataFrame([[1, 2]], index=["z"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (2, 4)),
            pd.DataFrame([[2, 4]], index=["x"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (3, 4)),
            pd.DataFrame([[3, 4]], index=["c"], columns=cols)
        )


class MultiIndexTestCase(BaseIndexTestCase):
    def test_index(self):
        cols = ["c", "d"]
        df = pd.DataFrame(
            [[1, 2], [2, 4], [3, 4]], index=["x", "y", "z"], columns=cols
        )

        idx = MultiIndex([
            ColumnsIndex('c'),
            ColumnsIndex('d')
        ])
        keys = idx.keys(df)
        self.assertEqual(keys, set([(1,), (2,), (3,), (4,)]))
        assert_frame_equal(
            idx.bucket(df, (1,)),
            pd.DataFrame([[1, 2]], index=["x"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (2,)),
            pd.DataFrame([[1, 2], [2, 4]], index=["x", "y"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (3,)),
            pd.DataFrame([[3, 4]], index=["z"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, (4,)),
            pd.DataFrame([[2, 4], [3, 4]], index=["y", "z"], columns=cols)
        )

        idx = MultiIndex([
            ColumnsIndex('c'),
            ColumnsIndex('d')
        ], combine_keys=True)
        keys = idx.keys(df)
        self.assertEqual(keys, set([
            ((3,), (4,)),
            ((2,), (4,)),
            ((1,), (2,)),
        ]))
        assert_frame_equal(
            idx.bucket(df, ((1,), (2,))),
            pd.DataFrame([[1, 2]], index=["x"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, ((2,), (4,))),
            pd.DataFrame([[2, 4]], index=["y"], columns=cols)
        )
        assert_frame_equal(
            idx.bucket(df, ((3,), (4,))),
            pd.DataFrame([[3, 4]], index=["z"], columns=cols)
        )
