import itertools
import math
from bisect import bisect_left, bisect
from operator import itemgetter
from typing import Type

import pandas as pd
import numpy as np

from .indices import BaseIndex
from .pairers import DeduplicatePairer, MatchPairer


class ThresholdMatcher(object):
    """Matchs records by computing similarity score.

    Final match results can be retrieved with a similarity threshold.
    """

    def __init__(self, index: Type[BaseIndex], fields: dict, dfa: pd.DataFrame, dfb: pd.DataFrame or None = None) -> None:
        """Creates new instance of ThresholdMatcher.

        If it is given 2 dataframes at the end then it will try to match records between them.
        If given only one dataframe then it attempt to detect duplicates in this dataframe.

        Args:
            index (subclass of BaseIndex): how to index the data
            fields (dict): mapping between field name and similarity class to use
            dfa (pd.DataFrame): data to match. Its index must not contain duplicates.
            dfb (pd.DataFrame): if this is set then match this with `dfa`, otherwise
                deduplicate `dfa`. If given then its index must not contain
                duplicates and its columns must match `dfa`'s.

        Returns:
            no value
        """
        if dfb is None:
            self._pairer = DeduplicatePairer(dfa, index)
        else:
            self._pairer = MatchPairer(dfa, dfb, index)
        self._fields = fields
        self._score_all_pairs()

    def _score_pair(self, ser_a, ser_b):
        """
        Calculate similarity value(0 <= sim_val <= 1) for a pair of records
        """
        sim_vec = dict()
        for k, scls in self._fields.items():
            if pd.isnull(ser_a[k]) or pd.isnull(ser_b[k]):
                sim_vec[k] = 0
            else:
                sim_vec[k] = scls.sim(ser_a[k], ser_b[k])
        return math.sqrt(
            sum(v * v for v in sim_vec.values()) / len(self._fields))

    def _remove_lesser_matches(self):
        """
        If someone already have a better match then remove other matches
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

    def _score_all_pairs(self):
        """
        Calculate similarity value for all pairs of records
        """
        pairs = []
        for (idx_a, ser_a), (idx_b, ser_b) in self._pairer.pairs():
            sim = self._score_pair(ser_a, ser_b)
            pairs.append((sim, idx_a, idx_b))
        self._pairs = sorted(pairs, key=itemgetter(0))
        self._remove_lesser_matches()
        self._keys = [t[0] for t in self._pairs]

    def get_index_pairs_within_thresholds(self, lower_bound=0.7, upper_bound=1) -> list:
        """Returns index pairs with similarity score within specified thresholds

        Args:
            lower_bound (float): pairs that score lower than this won't be returned
            upper_bound (float): pairs that score higher than this won't be returned

        Returns:
            list of tuples of index of matching records
        """
        return [t[1:] for t in self._pairs[
            bisect_left(self._keys, lower_bound):
            bisect(self._keys, upper_bound)]]

    def get_sample_pairs(self, sample_counts=5, lower_bound=0.7, upper_bound=1, step=0.05) -> pd.DataFrame:
        """Returns samples of record pairs for each range of similarity score

        Args:
            sample_count (int): number of samples in each range
            lower_bound (float): ranges with score lower than this won't be included
            upper_bound (float): ranges with score higher than this won't be included
            step (float): width of each range.

        Returns:
            a multi-indexed dataframe that only contain samples for each range
        """
        sample_records = []
        ranges = list(np.arange(upper_bound, lower_bound, -step)
                      ) + [lower_bound]
        for i, upper_val in enumerate(ranges[:-1]):
            lower_val = ranges[i+1]
            score_range = "%.2f-%.2f" % (upper_val, lower_val)
            pairs = self._pairs[
                bisect(self._keys, lower_val):
                bisect(self._keys, upper_val)
            ][:sample_counts]
            pairs.reverse()
            for pair_idx, pair in enumerate(pairs):
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

    def get_all_pairs(self, lower_bound=0.7, upper_bound=1) -> pd.DataFrame:
        """Returns all matching pairs between a lower bound and upper bound

        Args:
            lower_bound (float): pairs that score lower than this won't be returned
            upper_bound (float): pairs that score higher than this won't be returned

        Returns:
            a multi-indexed dataframe that contains all matching pairs
        """
        records = []
        pairs = reversed(self._pairs[
            bisect_left(self._keys, lower_bound):
            bisect(self._keys, upper_bound)])
        for pair_idx, pair in enumerate(pairs):
            sim_score, idx_a, idx_b = pair
            records.append(dict([
                ("pair_idx", pair_idx), ("sim_score", sim_score), ("row_key", idx_a)
            ] + list(self._pairer.frame_a.loc[idx_a].to_dict().items())))
            records.append(dict([
                ("pair_idx", pair_idx), ("sim_score", sim_score), ("row_key", idx_b)
            ] + list(self._pairer.frame_b.loc[idx_b].to_dict().items())))
        return pd.DataFrame.from_records(
            records,
            index=["pair_idx", "sim_score", "row_key"])

    def save_pairs_to_excel(self, name: str, match_threshold: float, sample_counts: int = 5, lower_bound: float = 0.7, step: float = 0.05) -> None:
        """Save matching results to an Excel file.

        This will create an Excel file with 3 sheets:
        - Sample pairs: sample pairs for each score range, similar to
            output of `get_sample_pairs`
        - All pairs: all pairs that score higher than lower bound ordered
            by score
        - Decision: selected threshold and how many pairs are counted
            as matched

        Args:
            name (string): excel file to save to
            match_threshold (float): the score above which a pair is
                considered matched
            sample_counts (int): number of samples per score range in
                "Sample pairs" sheet.
            lower_bound (float): pairs score lower than this will be
                eliminated from both "Sample pairs" and "All pairs"
                sheet.
            step (float): width of each range in "Sample pairs" sheet

        Returns:
            no value
        """
        samples = self.get_sample_pairs(
            sample_counts, lower_bound, 1, step)
        pairs = self.get_all_pairs(lower_bound, 1)
        dec = {
            "match_threshold": match_threshold,
            "number_of_matched_pairs": len(self._keys) - bisect_left(self._keys, match_threshold)
        }
        dec_tups = list(itertools.zip_longest(*dec.items()))
        dec_sr = pd.Series(list(dec_tups[1]), index=list(dec_tups[0]))
        with pd.ExcelWriter(name) as writer:
            samples.to_excel(writer, sheet_name='Sample pairs')
            pairs.to_excel(writer, sheet_name='All pairs')
            dec_sr.to_excel(writer, sheet_name="Decision")

    def print_decision(self, match_threshold: float):
        """Print number and percentage of matched pairs for selected threshold

        Args:
            match_threshold (float): the score above which a pair is
                considered matched

        Returns:
            no value
        """
        pairs = self.get_index_pairs_within_thresholds(
            lower_bound=match_threshold)
        num_pairs = len(pairs)
        print("for threshold %.3f:" % match_threshold)
        print("  %d matched pairs (%d%% of A, %d%% of B)" %
              (num_pairs, num_pairs / self._pairer.frame_a.shape[0] * 100, num_pairs / self._pairer.frame_b.shape[0] * 100))
