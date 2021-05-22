# Datamatch

Data matching is the process of identifying similar entities or matches across different datasets. This package provides utilities for data matching. There are 3 components to a data matching system:

- **Index** which divides the data into buckets such that only records within a single bucket need to be matched against each other. This improves matching time substantially as records with no chance of matching are never considered.
- **Similarity function** which computes a similarity score when given 2 values.
- **Matcher** which fetches record pairs from the index and for every pair computes similarity score for each field using the similarity functions, and finally combines the similarity scores into a single score for each pair.

This package mostly interface with Pandas so using Pandas is mandatory.

## Basic usage

Match records between 2 frames

```python
from datamatch import (
    ThresholdMatcher, ColumnsIndex, JaroWinklerSimiarity, DateSimilarity
)

matcher = ThresholdMatcher(ColumnsIndex(['year_of_birth']), {
    'first_name': JaroWinklerSimilarity(),
    'last_name': JaroWinklerSimilarity()
}, df1, df2)

# print sample pairs within different score ranges
print(matcher.get_sample_pairs())

# print all pairs that score 0.8 or higher
print(matcher.get_all_pairs(0.8))

# save matching result to an excel file
matcher.save_pairs_to_excel('match_result.xlsx', 0.8)

# get index pairs of matched records
matches = matcher.get_index_pairs_within_thresholds(0.8)
```

To deduplicate records in a frame, pass in just one frame instead of 2

```python
matcher = ThresholdMatcher(ColumnsIndex(['year_of_birth']), {
    'first_name': JaroWinklerSimilarity(),
    'last_name': JaroWinklerSimilarity()
}, df)
```

## Choosing a version

As this package is at preproduction state, increase in minor version number represents breaking changes.
