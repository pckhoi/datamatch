from datamatch.filters import DissimilarFilter, NonOverlappingFilter
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from datamatch.indices import ColumnsIndex, NoopIndex
from datamatch.matchers import ThresholdMatcher
from datamatch.scorers import AbsoluteScorer, MaxScorer, SimSumScorer
from datamatch.similarities import JaroWinklerSimilarity, StringSimilarity
from datamatch.variators import Swap


class TestThresholdMatcher(unittest.TestCase):
    def test_match(self):
        cols = ["a", "b"]
        dfa = pd.DataFrame(
            [
                ["ab", "cd"],
                ["rtx", "qw"]
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

        self.assertEqual(matcher._pairs, [(0.8, 1, 2), (1.0, 0, 0)])

        self.assertEqual(
            matcher.get_index_pairs_within_thresholds(), [(1, 2), (0, 0)])

        assert_frame_equal(
            matcher.get_sample_pairs(),
            pd.DataFrame.from_records([
                {"score_range": "1.00-0.95", "pair_idx": 0,
                    "sim_score": 1.0, "row_key": 0, "a": "ab", "b": "cd"},
                {"score_range": "1.00-0.95", "pair_idx": 0,
                    "sim_score": 1.0, "row_key": 0, "a": "ab", "b": "cd"},
                {"score_range": "0.85-0.80", "pair_idx": 0,
                    "sim_score": 0.8, "row_key": 1, "a": "rtx", "b": "qw"},
                {"score_range": "0.85-0.80", "pair_idx": 0,
                    "sim_score": 0.8, "row_key": 2, "a": "rt", "b": "qw"},
            ], index=["score_range", "pair_idx", "sim_score", "row_key"])
        )

        assert_frame_equal(
            matcher.get_sample_pairs(include_exact_matches=False),
            pd.DataFrame.from_records([
                {"score_range": "0.85-0.80", "pair_idx": 0,
                    "sim_score": 0.8, "row_key": 1, "a": "rtx", "b": "qw"},
                {"score_range": "0.85-0.80", "pair_idx": 0,
                    "sim_score": 0.8, "row_key": 2, "a": "rt", "b": "qw"},
            ], index=["score_range", "pair_idx", "sim_score", "row_key"])
        )

        assert_frame_equal(
            matcher.get_all_pairs(),
            pd.DataFrame.from_records([
                {"pair_idx": 0, "sim_score": 1.0,
                    "row_key": 0, "a": "ab", "b": "cd"},
                {"pair_idx": 0, "sim_score": 1.0,
                    "row_key": 0, "a": "ab", "b": "cd"},
                {"pair_idx": 1, "sim_score": 0.8,
                    "row_key": 1, "a": "rtx", "b": "qw"},
                {"pair_idx": 1, "sim_score": 0.8,
                    "row_key": 2, "a": "rt", "b": "qw"},
            ], index=["pair_idx", "sim_score", "row_key"])
        )

        assert_frame_equal(
            matcher.get_all_pairs(include_exact_matches=False),
            pd.DataFrame.from_records([
                {"pair_idx": 1, "sim_score": 0.8,
                    "row_key": 1, "a": "rtx", "b": "qw"},
                {"pair_idx": 1, "sim_score": 0.8,
                    "row_key": 2, "a": "rt", "b": "qw"},
            ], index=["pair_idx", "sim_score", "row_key"])
        )

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

    def test_scorer(self):
        columns = ['first_name', 'attract_id']
        df = pd.DataFrame([
            ['john', 5],
            ['jim', 5],
            ['ted', 3],
            ['tedd', 2]
        ], columns=columns)

        matcher = ThresholdMatcher(NoopIndex(), MaxScorer([
            AbsoluteScorer('attract_id', 1),
            SimSumScorer({
                'first_name': JaroWinklerSimilarity()
            })
        ]), df)

        self.assertEqual(
            matcher.get_clusters_within_threshold().to_string(),
            '\n'.join([
                '                                       first_name  attract_id',
                'cluster_idx pair_idx sim_score row_key                       ',
                '0           0        1.000000  0             john           5',
                '                               1              jim           5',
                '1           0        0.941667  2              ted           3',
                '                               3             tedd           2',
            ]),
        )

        self.assertEqual(
            matcher.get_clusters_within_threshold(
                include_exact_matches=False).to_string(),
            '\n'.join([
                '                                       first_name  attract_id',
                'cluster_idx pair_idx sim_score row_key                       ',
                '1           0        0.941667  2              ted           3',
                '                               3             tedd           2',
            ]),
        )

    def test_func_scorer(self):
        self.maxDiff = None
        df = pd.DataFrame([
            ['j', 'john', 20],
            ['j', 'jim', 20],
            ['b', 'bill', 19],
            ['b', 'bob', 21]
        ], columns=['fc', 'name', 'age'])

        matcher = ThresholdMatcher(
            index=ColumnsIndex('fc'),
            scorer=lambda a, b: 1.0 if a.age == b.age else 0.8,
            dfa=df
        )

        self.assertEqual(
            matcher.get_clusters_within_threshold().to_string(),
            '\n'.join([
                '                                       fc  name  age',
                'cluster_idx pair_idx sim_score row_key              ',
                '0           0        1.0       0        j  john   20',
                '                               1        j   jim   20',
                '1           0        0.8       2        b  bill   19',
                '                               3        b   bob   21',
            ])
        )
