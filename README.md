# Datamatch

Data matching is the process of identifying similar entities or matches across different datasets. This package provides utilities for data matching. There are 3 components to a data matching system:

- **Index** which divides the data into buckets such that only records within a single bucket need to be matched against each other. This improves matching time substantially as records with no chance of matching are never considered.
- **Similarity function** which computes a similarity score when given 2 values.
- **Matcher** which fetches record pairs from the index and for every pair computes similarity score for each field using the similarity functions, and finally combines the similarity scores into a single score for each pair.

This package mostly interface with Pandas so using Pandas is mandatory.

## Basic usage

```python
from datamatch import ThresholdMatcher, ColumnsIndex, JaroWinklerSimiarity

matcher = ThresholdMatcher(df1, df2, ColumnsIndex(['year_of_birth']), {
    'first_name': JaroWinklerSimilarity(),
    'last_name': JaroWinklerSimilarity()
})
print(matcher.get_index_pairs_within_thresholds(0.8))
```
