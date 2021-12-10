"""
A matcher matches records from two datasets (deduplicate if it is only given one dataset). It is also the main entry for this
library. Through its constructor, you can configure which index to use, which and how each column should be compared, etc...
"""

import operator
import functools
import itertools
import math
from bisect import bisect_left, bisect
from typing import Any, Iterator, Type

import pandas as pd
import numpy as np
from tqdm import tqdm

from .scorers import BaseScorer, FuncScorer, ScoreFunc, SimSumScorer
from .indices import BaseIndex
from .pairers import DeduplicatePairer, MatchPairer
from .filters import BaseFilter
from .variators import Variator


class ContinueOuter(Exception):
    """
    :meta private:
    """
    pass


MODE_MATCH = 1
MODE_DEDUP = 2


class ThresholdMatcher(object):
    """Matchs records by computing similarity score for each pair and discard those that fall below a threshold.

    This matcher does not require any training data, therefore it is perfect for when there is not too much data or if
    training data is not available.
    """

    _mode: int

    def __init__(
        self,
        index: Type[BaseIndex],
        scorer: dict or Type[BaseScorer] or ScoreFunc,
        dfa: pd.DataFrame,
        dfb: pd.DataFrame or None = None,
        variator: Type[Variator] or None = None,
        filters: list[Type[BaseFilter]] = [],
        show_progress: bool = False
    ) -> None:
        """
        If it is given two datasets then it will try to match records
        between them. If given only one dataset then it attempts to detect
        duplicates instead.

        :param index: The index to divide the dataset into distinct buckets.
        :type index: sub-class of :class:`datamatch.indices.BaseIndex`

        :param scorer: The scorer class to score each pair. If it is a dict then create
            a :class:`datamatch.scorers.SimSumScorer` with that dict and use it.
        :type scorer: Callable[[:class:`pandas:pandas.Series`, :class:`pandas:pandas.Series`], :obj:`float`] or sub-class of :class:`datamatch.scorers.BaseScorer` or :obj:`dict` of similarity classes

        :param dfa: The left dataset to match. Its index must not contain duplicates.
        :type dfa: :class:`pandas:pandas.DataFrame`

        :param dfb: The right dataset to match. Its index must not contain duplicates and
            its column must match **dfa**'s. If this is not given then the matcher will
            attempt to deduplicate **dfa** instead.
        :type dfb: :class:`pandas:pandas.DataFrame`, optional

        :param variator: The :ref:`variator <variators>` to use.
        :type variator: sub-class of :class:`datamatch.variators.Variator`, optional

        :param filters: The list of :ref:`filters` to use.
        :type filters: :obj:`list` of sub-class of :class:`datamatch.filters.BaseFilter`, optional

        :param show_progress: Prints a `tqdm <https://github.com/tqdm/tqdm>`_ progress bar to console during matching, defaults to False.
        :type show_progress: :obj:`bool`, optional
        """
        if dfb is None:
            self._pairer = DeduplicatePairer(dfa, index)
            self._mode = MODE_DEDUP
        else:
            self._pairer = MatchPairer(dfa, dfb, index)
            self._mode = MODE_MATCH
        if type(scorer) is dict:
            self._scorer = SimSumScorer(scorer)
        elif type(scorer).__name__ == 'function':
            self._scorer = FuncScorer(scorer)
        else:
            self._scorer = scorer
        if variator is None:
            self._variator = Variator()
        else:
            self._variator = variator
        self._filters = filters
        self._show_progress = show_progress
        self._score_all_pairs()

    def _remove_lesser_matches(self):
        """If a row already have a better match then remove the other matches.
        """
        indices_a = set()
        indices_b = set()
        keep = []
        self._pairs.reverse()
        for i, tup in enumerate(self._pairs):
            _, idx_a, idx_b = tup
            if idx_a in indices_a or idx_b in indices_b:
                continue
            indices_a.add(idx_a)
            indices_b.add(idx_b)
            keep.append(i)
        self._pairs = [self._pairs[i] for i in keep]
        self._pairs.reverse()

    def _valid_pairs(self) -> Iterator:
        for (idx_a, rec_a), (idx_b, rec_b) in self._pairer.pairs():
            try:
                for f in self._filters:
                    if not f.valid(rec_a, rec_b):
                        raise ContinueOuter
            except ContinueOuter:
                continue
            yield (idx_a, rec_a), (idx_b, rec_b)

    def _score_all_pairs(self):
        """Calculate similarity value for all pairs of records.
        """
        pairs = []
        valid_pairs = self._valid_pairs()
        if self._show_progress:
            valid_pairs = tqdm(valid_pairs, desc='scoring pairs')
        for (idx_a, rec_a), (idx_b, rec_b) in valid_pairs:
            sim = max(
                self._scorer.score(ser_a, ser_b)
                for ser_a, ser_b in itertools.product(
                    self._variator.variations(rec_a),
                    self._variator.variations(rec_b)
                )
            )
            pairs.append((sim, idx_a, idx_b))
        self._pairs = sorted(pairs, key=operator.itemgetter(0))
        # in dedup mode we can group more than two records therefore we're not dropping lesser matches
        if self._mode == MODE_MATCH:
            self._remove_lesser_matches()
        self._scores = [t[0] for t in self._pairs]

    def _split_clusters(self, orig_cluster: set[tuple[float, Any, Any]]) -> list[tuple[frozenset, set]]:
        paths: dict[Any, set] = {}
        pairs: dict[frozenset, tuple[float, Any, Any]] = {}
        nodes = set()
        for sim, idx_a, idx_b in orig_cluster:
            paths.setdefault(idx_a, set()).add(idx_b)
            paths.setdefault(idx_b, set()).add(idx_a)
            nodes.add(idx_a)
            nodes.add(idx_b)
            pairs[frozenset([idx_a, idx_b])] = (sim, idx_a, idx_b)
        clusters: list[set] = []
        clustered = set()
        for node in nodes:
            if node in clustered:
                continue
            cluster = set([node])
            clustered.add(node)
            queue = [node]
            # BFS to find all members of cluster
            while len(queue) > 0:
                cur = queue.pop()
                for neighbor in paths[cur]:
                    if neighbor in clustered:
                        continue
                    if all([n in paths[neighbor] for n in cluster]):
                        clustered.add(neighbor)
                        cluster.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)
        return [
            (
                frozenset(s),
                set([
                    pairs[frozenset(y)]
                    for y in itertools.combinations(s, 2)
                ])
            )
            for s in clusters if len(s) > 1
        ]

    def _get_clusters_dict_within_thresholds(self, lower_bound=0.7, upper_bound=1) -> dict[frozenset, set]:
        pairs = self._pairs[
            bisect_left(self._scores, lower_bound):
            bisect(self._scores, upper_bound)]
        pairs.reverse()
        clusters: dict[frozenset, set] = {}

        for sim, idx_a, idx_b in pairs:
            cluster_keys: list[set] = []
            for key in clusters:
                if idx_a in key or idx_b in key:
                    cluster_keys.append(key)
            new_key = set()
            new_val = set()
            for key in cluster_keys:
                new_key = new_key.union(key)
                new_val = new_val.union(clusters[key])
                clusters.pop(key, None)

            new_key.add(idx_a)
            new_key.add(idx_b)
            new_val.add((sim, idx_a, idx_b))
            clusters.__setitem__(frozenset(new_key), new_val)

        return dict(functools.reduce(operator.add, [
            self._split_clusters(cluster) for cluster in clusters.values()
        ]))

    def get_index_clusters_within_thresholds(self, lower_bound=0.7, upper_bound=1) -> list[frozenset]:
        """Returns index clusters with similarity scores within the specified thresholds.

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param upper_bound: The upper threshold above which pairs won't be included, defaults to 1.
        :type upper_bound: :obj:`float`

        :return: A list of clusters, each cluster is a set of indices.
        :rtype: :obj:`list` of :obj:`frozenset`
        """
        return [
            idx_set for idx_set in
            self._get_clusters_dict_within_thresholds(lower_bound, upper_bound)
        ]

    def get_clusters_within_threshold(self, lower_bound=0.7, upper_bound=1, include_exact_matches: bool = True) -> pd.DataFrame:
        """Returns all clusters between a lower bound and upper bound.

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param upper_bound: The upper threshold above which pairs won't be included, defaults to 1.
        :type upper_bound: :obj:`float`

        :param include_exact_matches: Includes clusters with score = 1.0.
        :type include_exact_matches: :obj:`bool`

        :return: A multi-indexed frame that contains all matched clusters.
        :rtype: :class:`pandas:pandas.DataFrame`
        """
        records = []
        clusters = sorted([
            sorted(pairs_set, key=lambda x: x[0], reverse=True)
            for pairs_set in self._get_clusters_dict_within_thresholds(
                lower_bound, upper_bound
            ).values()
        ], key=lambda x: x[0][0], reverse=True)
        for cluster_idx, cluster in enumerate(clusters):
            if not include_exact_matches and sum(score for score, _, _ in cluster) == len(cluster):
                continue
            for pair_idx, pair in enumerate(cluster):
                sim_score, idx_a, idx_b = pair
                records.append(dict([
                    ("cluster_idx", cluster_idx), ("pair_idx", pair_idx),
                    ("sim_score", sim_score), ("row_key", idx_a)
                ] + list(self._pairer.frame_a.loc[idx_a].to_dict().items())))
                records.append(dict([
                    ("cluster_idx", cluster_idx), ("pair_idx", pair_idx),
                    ("sim_score", sim_score), ("row_key", idx_b)
                ] + list(self._pairer.frame_b.loc[idx_b].to_dict().items())))
        return pd.DataFrame.from_records(
            records,
            index=["cluster_idx", "pair_idx", "sim_score", "row_key"])

    def get_index_pairs_within_thresholds(self, lower_bound=0.7, upper_bound=1) -> list:
        """Returns index pairs with similarity scores within specified thresholds.

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param upper_bound: The upper threshold above which pairs won't be included, defaults to 1.
        :type upper_bound: :obj:`float`

        :return: A list of pairs of indices of matching records.
        :rtype: :obj:`list` of :obj:`tuple`
        """
        return [t[1:] for t in self._pairs[
            bisect_left(self._scores, lower_bound):
            bisect(self._scores, upper_bound)]]

    def get_sample_pairs(self, sample_counts=5, lower_bound=0.7, upper_bound=1, step=0.05, include_exact_matches: bool = True) -> pd.DataFrame:
        """Returns samples of record pairs for each range of similarity scores.

        :param sample_counts: The number of samples in each range.
        :type sample_counts: :obj:`int`

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param upper_bound: The upper threshold above which pairs won't be included, defaults to 1.
        :type upper_bound: :obj:`float`

        :param step: The width of each range.
        :type step: :obj:`float`

        :param include_exact_matches: Includes pairs with score = 1.0.
        :type include_exact_matches: :obj:`bool`

        :return: A multi-indexed frame that only contain samples for each range.
        :rtype: :class:`pandas:pandas.DataFrame`
        """
        sample_records = []
        ranges = list(np.arange(upper_bound, lower_bound, -step)
                      ) + [lower_bound]
        for i, upper_val in enumerate(ranges[:-1]):
            lower_val = ranges[i+1]
            score_range = "%.2f-%.2f" % (upper_val, lower_val)
            pairs = self._pairs[
                bisect(self._scores, lower_val):
                bisect(self._scores, upper_val)
            ][:sample_counts]
            pairs.reverse()
            for pair_idx, pair in enumerate(pairs):
                if not include_exact_matches and pair[0] == 1:
                    continue
                sim_score, idx_a, idx_b = pair
                sample_records.append(dict([
                    ("score_range", score_range), ("pair_idx", pair_idx),
                    ("sim_score", sim_score), ("row_key", idx_a)
                ] + list(self._pairer.frame_a.loc[idx_a].to_dict().items())))
                sample_records.append(dict([
                    ("score_range", score_range), ("pair_idx", pair_idx),
                    ("sim_score", sim_score), ("row_key", idx_b)
                ] + list(self._pairer.frame_b.loc[idx_b].to_dict().items())))
        return pd.DataFrame.from_records(
            sample_records,
            index=["score_range", "pair_idx", "sim_score", "row_key"])

    def get_all_pairs(self, lower_bound=0.7, upper_bound=1, include_exact_matches: bool = True) -> pd.DataFrame:
        """Returns all matching pairs between a lower bound and an upper bound.

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param upper_bound: The upper threshold above which pairs won't be included, defaults to 1.
        :type upper_bound: :obj:`float`

        :param include_exact_matches: Includes pairs with score = 1.0.
        :type include_exact_matches: :obj:`bool`

        :return: A multi-indexed dataframe that only contain samples for each range.
        :rtype: :class:`pandas:pandas.DataFrame`
        """
        records = []
        pairs = reversed(self._pairs[
            bisect_left(self._scores, lower_bound):
            bisect(self._scores, upper_bound)])
        for pair_idx, pair in enumerate(pairs):
            sim_score, idx_a, idx_b = pair
            if not include_exact_matches and sim_score == 1:
                continue
            records.append(dict([
                ("pair_idx", pair_idx), ("sim_score", sim_score), ("row_key", idx_a)
            ] + list(self._pairer.frame_a.loc[idx_a].to_dict().items())))
            records.append(dict([
                ("pair_idx", pair_idx), ("sim_score", sim_score), ("row_key", idx_b)
            ] + list(self._pairer.frame_b.loc[idx_b].to_dict().items())))
        return pd.DataFrame.from_records(
            records,
            index=["pair_idx", "sim_score", "row_key"])

    def save_pairs_to_excel(
        self, name: str, match_threshold: float, sample_counts: int = 5, lower_bound: float = 0.7,
        step: float = 0.05, include_exact_matches: bool = True
    ) -> None:
        """Saves matching pairs to an Excel file.

        This will create an Excel file with 3 sheets:

        - Sample pairs: sample pairs for each score range, similar to output of :meth:`get_sample_pairs`.

        - All pairs: all pairs that score higher than **lower_bound** ordered by the similarity score.

        - Decision: selected threshold and how many pairs are counted as matched.

        :param name: The excel file to save to.
        :type name: :obj:`str`

        :param match_threshold: The score above which a pair is considered a match.
        :type match_threshold: :obj:`float`

        :param sample_counts: The number of samples per score range in the "Sample pairs" sheet.
        :type sample_counts: :obj:`int`

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param step: The width of each range in the "Sample pairs" sheet.
        :type step: :obj:`float`

        :param include_exact_matches: Includes pairs with score = 1.0.
        :type include_exact_matches: :obj:`bool`

        :rtype: :obj:`None`
        """
        with pd.ExcelWriter(name) as writer:
            self.get_sample_pairs(
                sample_counts, lower_bound, 1, step, include_exact_matches
            ).to_excel(writer, sheet_name='Sample pairs')
            self.get_all_pairs(
                lower_bound, 1, include_exact_matches
            ).to_excel(writer, sheet_name='All pairs')
            self._decision_series(
                match_threshold
            ).to_excel(writer, sheet_name="Decision")

    def _decision_series(self, match_threshold: float) -> pd.Series:
        dec = {
            "match_threshold": match_threshold,
            "number_of_matched_pairs": len(self._scores) - bisect_left(self._scores, match_threshold)
        }
        dec_tups = list(itertools.zip_longest(*dec.items()))
        return pd.Series(list(dec_tups[1]), index=list(dec_tups[0]))

    def save_clusters_to_excel(self, name: str, match_threshold: float, lower_bound: float = 0.7, include_exact_matches: bool = True) -> None:
        """Save matched clusters to an Excel file.

        This will create an Excel file with two sheets:

        - All clusters: all clusters that score higher than lower bound ordered by score.

        - Decision: selected threshold and how many pairs are counted as matched.

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param name: The excel file to save to.
        :type name: :obj:`str`

        :param match_threshold: The score above which a pair is considered a match.
        :type match_threshold: :obj:`float`

        :param lower_bound: The lower threshold below which pairs won't be included, defaults to 0.7.
        :type lower_bound: :obj:`float`

        :param include_exact_matches: Includes clusters with score = 1.0.
        :type include_exact_matches: :obj:`bool`

        :rtype: :obj:`None`
        """
        with pd.ExcelWriter(name) as writer:
            self.get_clusters_within_threshold(
                lower_bound,
                include_exact_matches
            ).to_excel(writer, sheet_name='All clusters')
            self._decision_series(
                match_threshold
            ).to_excel(writer, sheet_name="Decision")

    def print_decision(self, match_threshold: float) -> None:
        """Prints number and percentage of matched pairs for selected threshold.

        :param match_threshold: The score above which a pair is considered a match.
        :type match_threshold: :obj:`float`

        :rtype: :obj:`None`
        """
        pairs = self.get_index_pairs_within_thresholds(
            lower_bound=match_threshold)
        num_pairs = len(pairs)
        print("for threshold %.3f:" % match_threshold)
        print("  %d matched pairs (%d%% of A, %d%% of B)" %
              (num_pairs, num_pairs / self._pairer.frame_a.shape[0] * 100, num_pairs / self._pairer.frame_b.shape[0] * 100))
