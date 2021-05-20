import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from .indices import NoopIndex
from .matchers import ThresholdMatcher
from .similarities import JaroWinklerSimilarity, StringSimilarity


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
            NoopIndex(), {"a": StringSimilarity()}, dfa, dfb)

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
            ['rhea', 'cherrie']
        ], columns=cols)

        matcher = ThresholdMatcher(NoopIndex(), {
            'last': JaroWinklerSimilarity(),
            'first': JaroWinklerSimilarity()
        }, df)

        self.assertEqual(
            matcher.get_index_pairs_within_thresholds(),
            [(0, 1), (4, 5), (2, 3), (6, 7)])
