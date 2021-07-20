from datamatch.filters import DissimilarFilter, NonOverlappingFilter
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from .indices import NoopIndex
from .matchers import ThresholdMatcher
from .similarities import JaroWinklerSimilarity, StringSimilarity
from .variators import Swap


class TestThresholdMatcher(unittest.TestCase):
    def test_match(self):
        cols = ["a", "b"]
        dfa = pd.DataFrame(
            [
                ["ab", "cd"],
            ],
            columns=cols
        )
        dfb = pd.DataFrame(
            [
                ["ab", "cd"],
                ["ae", "vb"],
                ["rt", "qw"]
            ],
            columns=cols
        )
        matcher = ThresholdMatcher(
            NoopIndex(), {"a": StringSimilarity()}, dfa, dfb
        )

        self.assertEqual(matcher._pairs, [(1.0, 0, 0)])

        self.assertEqual(matcher.get_index_pairs_within_thresholds(), [(0, 0)])

        assert_frame_equal(
            matcher.get_sample_pairs(),
            pd.DataFrame.from_records([
                {"score_range": "1.00-0.95", "pair_idx": 0,
                    "sim_score": 1.0, "row_key": 0, "a": "ab", "b": "cd"},
                {"score_range": "1.00-0.95", "pair_idx": 0,
                    "sim_score": 1.0, "row_key": 0, "a": "ab", "b": "cd"},
            ], index=["score_range", "pair_idx", "sim_score", "row_key"]))

        assert_frame_equal(
            matcher.get_all_pairs(),
            pd.DataFrame.from_records([
                {"pair_idx": 0, "sim_score": 1.0,
                    "row_key": 0, "a": "ab", "b": "cd"},
                {"pair_idx": 0, "sim_score": 1.0,
                    "row_key": 0, "a": "ab", "b": "cd"},
            ], index=["pair_idx", "sim_score", "row_key"]))

    def test_ensure_unique_index(self):
        dfa = pd.DataFrame(
            [[1, 2], [3, 4]], index=["a", "a"]
        )
        dfb = pd.DataFrame(
            [[5, 6], [7, 8]], index=["a", "b"]
        )
        with self.assertRaisesRegex(
                ValueError,
                "Dataframe index contains duplicates. Both frames need to have index free of duplicates."):
            ThresholdMatcher(NoopIndex(), {"a": StringSimilarity()}, dfa, dfb)

    def test_ensure_same_columns(self):
        dfa = pd.DataFrame(
            [[1, 2], [3, 4]], columns=["a", "c"]
        )
        dfb = pd.DataFrame(
            [[5, 6], [7, 8]], columns=["a", "b"]
        )
        with self.assertRaisesRegex(
                ValueError,
                "Dataframe columns are not equal."):
            ThresholdMatcher(NoopIndex(), {"a": StringSimilarity()}, dfa, dfb)

    def test_deduplicate(self):
        cols = ['last', 'first']
        df = pd.DataFrame([
            ['beech', 'freddie'],
            ['beech', 'freedie'],
            ['dupas', 'demia'],
            ['dupas', 'demeia'],
            ['brown', 'latoya'],
            ['bowen', 'latoya'],
            ['rhea', 'cherri'],
            ['rhea', 'cherrie'],
            ['be', 'freedie'],
            ['du', 'demeia'],
            ['teneisha', 'green'],
            ['tyler', 'green'],
            ['te neisha', 'green'],
            ['t', 'green'],
        ], columns=cols)

        matcher = ThresholdMatcher(NoopIndex(), {
            'last': JaroWinklerSimilarity(),
            'first': JaroWinklerSimilarity()
        }, df)

        self.assertEqual(
            matcher.get_index_clusters_within_thresholds(0.83),
            [
                frozenset({6, 7}),
                frozenset({4, 5}),
                frozenset({2, 3, 9}),
                frozenset({10, 12, 13}),
                frozenset({0, 8, 1}),
            ],
        )

        self.maxDiff = None
        self.assertEqual(
            matcher.get_clusters_within_threshold(0.83).to_string(),
            '\n'.join([
                '                                             last    first',
                'cluster_idx pair_idx sim_score row_key                    ',
                '0           0        0.990522  6             rhea   cherri',
                '                               7             rhea  cherrie',
                '1           0        0.985297  10        teneisha    green',
                '                               12       te neisha    green',
                '            1        0.878609  10        teneisha    green',
                '                               13               t    green',
                '            2        0.876863  12       te neisha    green',
                '                               13               t    green',
                '2           0        0.980748  2            dupas    demia',
                '                               3            dupas   demeia',
                '            1        0.923472  3            dupas   demeia',
                '                               9               du   demeia',
                '            2        0.902589  2            dupas    demia',
                '                               9               du   demeia',
                '3           0        0.941913  4            brown   latoya',
                '                               5            bowen   latoya',
                '4           0        0.939581  0            beech  freddie',
                '                               1            beech  freedie',
                '            1        0.923472  1            beech  freedie',
                '                               8               be  freedie',
                '            2        0.857679  0            beech  freddie',
                '                               8               be  freedie',
            ]),
        )

    def test_swap_variator(self):
        cols = ['last', 'first']
        df = pd.DataFrame([
            ['blake', 'lauri'],
            ['lauri', 'blake'],
            ['robinson', 'alexis'],
            ['robertson', 'alexis'],
            ['haynes', 'terry'],
            ['terry', 'hayes']
        ], columns=cols)

        matcher = ThresholdMatcher(NoopIndex(), {
            'last': JaroWinklerSimilarity(),
            'first': JaroWinklerSimilarity()
        }, df, variator=Swap('first', 'last'))

        self.assertEqual(
            matcher.get_index_pairs_within_thresholds(),
            [(2, 3), (4, 5), (0, 1)]
        )

    def test_filters(self):
        cols = ['uid', 'first', 'agency', 'start', 'end']
        df = pd.DataFrame([
            ['1', 'john', 'slidell pd', 0, 10],
            ['2', 'john', 'slidell pd', 10, 20],
            ['3', 'john', 'slidell pd', 20, 30],
            ['4', 'john', 'gretna pd', 11, 21],
            ['5', 'john', 'gretna pd', 0, 7],
            ['6', 'john', 'gretna pd', 10, 18],
        ], columns=cols)

        matcher = ThresholdMatcher(NoopIndex(), {
            'first': JaroWinklerSimilarity()
        }, df, filters=[
            DissimilarFilter('agency'),
            NonOverlappingFilter('start', 'end')
        ])

        self.assertEqual(
            matcher.get_index_pairs_within_thresholds(),
            [(0, 3), (1, 4), (2, 4), (2, 5)]
        )
