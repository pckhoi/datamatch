import itertools
from typing import Iterator, Type

import pandas as pd

from .indices import BaseIndex


class BasePairer(object):
    def __init__(self, index: Type[BaseIndex]) -> None:
        self._index = index

    @property
    def frame_a(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def frame_b(self) -> pd.DataFrame:
        raise NotImplementedError()

    def pairs(self) -> Iterator:
        raise NotImplementedError()


class MatchPairer(BasePairer):
    def __init__(self, dfa: pd.DataFrame, dfb: pd.DataFrame, index: Type[BaseIndex]) -> None:
        super().__init__(index)
        if dfa.index.duplicated().any() or dfb.index.duplicated().any():
            raise ValueError(
                "Dataframe index contains duplicates. Both frames need to have index free of duplicates."
            )
        if set(dfa.columns) != set(dfb.columns):
            raise ValueError(
                "Dataframe columns are not equal."
            )

        self._dfa = dfa
        self._dfb = dfb

    @property
    def frame_a(self) -> pd.DataFrame:
        return self._dfa

    @property
    def frame_b(self) -> pd.DataFrame:
        return self._dfb

    def pairs(self) -> Iterator:
        keys_a = self._index.keys(self._dfa)
        keys_b = self._index.keys(self._dfb)
        keys = list(keys_a.intersection(keys_b))
        for key in keys:
            rows_a = self._index.bucket(self._dfa, key)
            rows_b = self._index.bucket(self._dfb, key)
            for idx_a, ser_a in rows_a.iterrows():
                for idx_b, ser_b in rows_b.iterrows():
                    yield (idx_a, ser_a), (idx_b, ser_b)


class DeduplicatePairer(BasePairer):
    def __init__(self, df: pd.DataFrame, index: Type[BaseIndex]) -> None:
        super().__init__(index)
        if df.index.duplicated().any():
            raise ValueError(
                "Dataframe index contains duplicates."
            )
        self._df = df

    @property
    def frame_a(self) -> pd.DataFrame:
        return self._df

    @property
    def frame_b(self) -> pd.DataFrame:
        return self._df

    def pairs(self) -> Iterator:
        keys = self._index.keys(self._df)
        for key in keys:
            rows = self._index.bucket(self._df, key)
            for row_a, row_b in itertools.combinations(rows.iterrows(), 2):
                yield row_a, row_b


def build_pairer(index: Type[BaseIndex], dfa: pd.DataFrame, dfb: pd.DataFrame or None = None) -> Type[BasePairer]:
    if dfb is None:
        return DeduplicatePairer(dfa, index)
    return MatchPairer(dfa, dfb, index)
