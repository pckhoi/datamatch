:github_url: https://github.com/pckhoi/datamatch/blob/main/doc/filters.rst

Filtering
=========

Sometimes it is easier to express what pairs should be matched in terms of
what conditions can be used to mark a pair as unmatchable instead of what
conditions can be used to mark a pair as matchable, which is the approach
of indexing. Filtering is a technique that also aims to improve matching
performance, but using conditions to reject a pair instead. You can employ
both filtering and indexing or just one of them.

This example demonstrates how filtering work:

.. ipython::

    In [0]: import pandas as pd
       ...: from datamatch import (
       ...:     ThresholdMatcher, JaroWinklerSimilarity, DissimilarFilter, NonOverlappingFilter
       ...: )

    In [0]: df = pd.DataFrame([
       ...:     ['1', 'john', 'slidell pd', 0, 10],
       ...:     ['2', 'john', 'slidell pd', 10, 20],
       ...:     ['3', 'john', 'slidell pd', 20, 30],
       ...:     ['4', 'john', 'gretna pd', 11, 21],
       ...:     ['5', 'john', 'gretna pd', 0, 7],
       ...:     ['6', 'john', 'gretna pd', 10, 18],
       ...: ], columns=['uid', 'first', 'agency', 'start', 'end'])
       ...: df

    In [0]: # we can use multiple filters as demonstrated here
       ...: matcher = ThresholdMatcher(NoopIndex(), {
       ...:     'first': JaroWinklerSimilarity()
       ...: }, df, filters=[
       ...:     DissimilarFilter('agency'),
       ...:     NonOverlappingFilter('start', 'end')
       ...: ])
       ...: matcher.get_all_pairs()

See :ref:`Filters` to find out more.
