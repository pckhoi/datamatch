from unittest import TestCase

import pandas as pd
import numpy as np

from datamatch.similarities import AbsoluteNumericalSimilarity, JaroWinklerSimilarity
from datamatch.scorers import RefuseToScoreException, SimSumScorer, AbsoluteScorer, MinScorer, MaxScorer


class SimSumScorerTestCase(TestCase):
    def test_score(self):
        scorer = SimSumScorer({
            'first_name': JaroWinklerSimilarity(),
            'age': AbsoluteNumericalSimilarity(10)
        })
        columns = ['first_name', 'age']
        self.assertEqual(
            scorer.score(
                pd.Series(["john", 41], index=columns),
                pd.Series(["john", 41], index=columns),
            ),
            1,
        )
        self.assertEqual(
            scorer.score(
                pd.Series(["jim", 41], index=columns),
                pd.Series(["jimm", 43], index=columns),
            ),
            0.8737093656105305,
        )


class AbsoluteScorerTestCase(TestCase):
    def test_score(self):
        scorer = AbsoluteScorer('attract_id', 1)
        self.assertEqual(
            scorer.score(
                pd.Series([1234], index=['attract_id']),
                pd.Series([1234], index=['attract_id']),
            ),
            1
        )
        self.assertRaises(RefuseToScoreException, lambda: scorer.score(
            pd.Series([1234], index=['attract_id']),
            pd.Series([2345], index=['attract_id']),
        ))
        self.assertRaises(RefuseToScoreException, lambda: scorer.score(
            pd.Series([1234], index=['attract_id']),
            pd.Series([np.NaN], index=['attract_id']),
        ))
        self.assertRaises(RefuseToScoreException, lambda: scorer.score(
            pd.Series([None], index=['attract_id']),
            pd.Series([1234], index=['attract_id']),
        ))

    def test_ignore_key_error(self):
        series_a = pd.Series([1], index=['a'])
        series_b = pd.Series([2], index=['a'])
        self.assertRaises(
            KeyError,
            lambda: AbsoluteScorer('b', 1).score(series_a, series_b)
        )
        self.assertRaises(
            RefuseToScoreException,
            lambda: AbsoluteScorer(
                'b', 1, ignore_key_error=True
            ).score(series_a, series_b)
        )


class MaxScorerTestCase(TestCase):
    def test_score(self):
        scorer = MaxScorer([
            AbsoluteScorer('attract_id', 1),
            SimSumScorer({
                'first_name': JaroWinklerSimilarity()
            })
        ])
        columns = ['first_name', 'attract_id']
        self.assertEqual(
            scorer.score(
                pd.Series(['john', 5], index=columns),
                pd.Series(['jim', 5], index=columns),
            ),
            1
        )
        self.assertEqual(
            scorer.score(
                pd.Series(['john', 5], index=columns),
                pd.Series(['jim', 4], index=columns),
            ),
            0.575
        )


class MinScorerTestCase(TestCase):
    def test_score(self):
        scorer = MinScorer([
            AbsoluteScorer('repell_id', 0),
            SimSumScorer({
                'first_name': JaroWinklerSimilarity()
            })
        ])
        columns = ['first_name', 'repell_id']
        self.assertEqual(
            scorer.score(
                pd.Series(['john', 5], index=columns),
                pd.Series(['jim', 5], index=columns),
            ),
            0
        )
        self.assertEqual(
            scorer.score(
                pd.Series(['john', 5], index=columns),
                pd.Series(['jim', 4], index=columns),
            ),
            0.575
        )
