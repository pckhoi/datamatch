:github_url: https://github.com/pckhoi/datamatch/blob/main/doc/tutorial.rst

Tutorial
========

First, download the :download:`DBLP-ACM dataset <DBLP-ACM.zip>` [1]_ to your
working directory and unzip it.

.. code-block:: bash

    unzip DBLP-ACM.zip -d DBLP-ACM

.. unzip:: DBLP-ACM.zip DBLP-ACM

This dataset contains article names, authors, and years from two different
sources, and the perfect matching between them. Our job is to match articles
from those two sources such that we can produce a result as close to the
provided perfect matching as possible.

Load and clean data
-------------------

Preceding the matching step is the data cleaning and standardization step,
which is of great importance. We're keeping this step as simple as possible:

.. ipython::

    In [0]: import pandas as pd
       ...: from datamatch import (
       ...:     ThresholdMatcher, StringSimilarity, NoopIndex, ColumnsIndex
       ...: )

    @suppress
    In [0]: pd.set_option("display.width", 300)
       ...: pd.set_option("display.max_columns", 20)

.. ipython::

    In [0]: dfa = pd.read_csv('DBLP-ACM/ACM.csv')
       ...: # set `id` as the index for this frame, this is important because the library
       ...: # relies on the frame's index to tell which row is which.
       ...: dfa = dfa.set_index('id', drop=True)
       ...: # make sure titles are all in lower case
       ...: dfa.loc[:, 'title'] = dfa.title.str.strip().str.lower()
       ...: # split author names by comma and sort them before joining them back with comma
       ...: dfa.loc[dfa.authors.notna(), 'authors'] = dfa.loc[dfa.authors.notna(), 'authors']\
       ...:     .map(lambda x: ', '.join(sorted(x.split(', '))))
       ...: dfa

.. ipython::

    In [0]: dfb = pd.read_csv('DBLP-ACM/DBLP2.csv')
       ...: # here we do the same cleaning step
       ...: dfb = dfb.set_index('id', drop=True)
       ...: dfb.loc[:, 'title'] = dfb.title.str.strip().str.lower()
       ...: dfb.loc[dfb.authors.notna(), 'authors'] = dfb.loc[dfb.authors.notna(), 'authors']\
       ...:     .map(lambda x: ', '.join(sorted(x.split(', '))))
       ...: dfb

Try matching for the first time
-------------------------------

Here's a quick primer on how threshold-based classification work. For each pair
of records, produce a similarity score (ranges from 0 to 1) between each
corresponding field, then combine to produce a final similarity score (also
ranges from 0 to 1). You can select different similarity functions for each
field depending on their characteristics (see more similarity functions
:ref:`here <Similarities>`). Finally which pairs count as matches depends on an
arbitrary threshold (for similarity score) that you specify. While this
classification method is not the gold standard in any way, it is simple and
does not require any training data, which makes it a great fit for many
problems. To learn in-depth details, see [2]_.

You can now start matching data using
:class:`ThresholdMatcher <datamatch.matchers.ThresholdMatcher>`. Notice how
simple it all is, you just need to specify the datasets to match and which
similarity function to use for each field:

.. ipython::

    @verbatim
    In [0]: matcher = ThresholdMatcher(NoopIndex(), {
       ...:     'title': StringSimilarity(),
       ...:     'authors': StringSimilarity(),
       ...: }, dfa, dfb)

And let's wait... Actually, if you have been waiting for like 5 minutes you
can stop it now. We're comparing 6 million pairs of records so it would help
tremendously if only there are some ways to increase performance.

Introducing the index
---------------------

The index (not to be confused with Pandas Index) is a data structure that
helps to reduce the number of pairs to be compared. It does this by deriving
an indexing key from each record and only attempt to match records that have
the same key. Without this technique, matching two datasets with `n` and `m`
records, respectively, would take `n x m` detailed comparisons, which is
probably infeasible for most non-trivial use cases. To learn more about
indexing, see [3]_. Another technique to reduce the number of pairs but
works the opposite way of indexing is :ref:`filtering <Filters>`.

We have been using :class:`NoopIndex <datamatch.indices.NoopIndex>` which
is the same as using no index whatsoever. We can do better. Notice how the
`year` column in both datasets denote the year in which the article was
published. It is very unlikely then that two articles within different years
could be the same. Let's employ this `year` column with
:class:`ColumnsIndex <datamatch.indices.ColumnsIndex>`:

.. ipython::

    In [0]: matcher = ThresholdMatcher(ColumnsIndex('year'), {
       ...:     'title': StringSimilarity(),
       ...:     'authors': StringSimilarity(),
       ...: }, dfa, dfb)

Now, this should run for under 1 or 2 minutes. This is not the best performance
that we can wring out of this dataset but very good for how little effort it
requires.

Select a threshold
------------------

The :class:`ThresholdMatcher <datamatch.matchers.ThresholdMatcher>` class does
not require a threshold up-front because usually, it is useful to be able to
experiment with different thresholds after the matching is done. Let's see what
the pairs look like:

.. ipython::

    In [0]: matcher.get_sample_pairs()

This returns a multi-index frame that shows five pairs under each threshold
range, ordered by descending similarity score. The purpose is to give you an
overview of what the matching records are like under different thresholds.
The returned frame has four index levels:

- **score_range**: the range of score. By default each range has a width of
  0.05. You can tweak this value with the ``step`` argument.
- **pair_idx**: the index of each pair within the range. By default it shows
  maximum 5 pairs within each range. You can tweak this value with the
  ``sample_counts`` argument.
- **sim_score**: the similarity score of this pair.
- **row_key**: the row index from the input datasets. Usually the desired
  output of the matching process is a list of matching pairs, each
  represented by a tuple of indices from the input datasets. You can get
  this list with
  :meth:`get_index_pairs_within_thresholds <datamatch.matchers.ThresholdMatcher.get_index_pairs_within_thresholds>`
  as will be demonstrated below.

The columns of this frame are the same columns as the input datasets
regardless of whether they were used to compute similarity score.

For the purpose of choosing the correct threshold, there are more tools
at our disposal:

- :meth:`get_all_pairs <datamatch.matchers.ThresholdMatcher.get_all_pairs>`:
  Returns matching pairs as a multi-index frame. It has the following levels:

  * **pair_idx**: the pair number.
  * **sim_score**: the similarity score.
  * **row_key**: the row index from input dataset.

.. ipython::

    In [0]: matcher.get_all_pairs(0.577).head(30)

- :meth:`save_pairs_to_excel <datamatch.matchers.ThresholdMatcher.save_pairs_to_excel>`:
  Save matching pairs to Excel for reviewing.

After a bit of experimentation, I selected `0.577` as my threshold. Let's
see the result:

.. ipython::

    In [0]: # this will return each pair as a tuple of index from both datasets
       ...: pairs = matcher.get_index_pairs_within_thresholds(0.577)

    In [1]: # we can construct a dataframe out of it with similar column names
       ...: # to this dataset's perfect mapping CSV.
       ...: res = pd.DataFrame(pairs, columns=['idACM', 'idDBLP'])\
       ...:     .set_index(['idACM', 'idDBLP'], drop=False)

    In [2]: # load the perfect mapping
       ...: pm = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')\
       ...:     .set_index(['idACM', 'idDBLP'], drop=False)

    @doctest
    In [3]: total = len(dfa) * len(dfb)
       ...: total
    Out[3]: 6001104

    @doctest
    In [4]: sensitivity = len(pm[pm.index.isin(res.index)]) / len(pm)
       ...: sensitivity
    Out[4]: 0.9937050359712231

    @doctest
    In [5]: specificity = 1 - len(res[~res.index.isin(pm.index)]) / (total - len(pm))
       ...: specificity
    Out[5]: 0.9999978329288134

The `sensitivity` and `specificity` are not perfect but they're still great
considering how simple this matching script is.

.. [1] `DBLP-ACM dataset <https://dbs.uni-leipzig.de/de/research/projects/object_matching/benchmark_datasets_for_entity_resolution>`_
   by the database group of Prof. Erhard Rahm under the `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_

.. [2] Peter Christen. "6.2 Threshold-Based Classification" In `Data Matching:
    Concepts and Techniques for Record Linkage, Entity Resolution, and
    Duplicate Detection`, 131-133. Springer, 2012.

.. [3] Peter Christen. "4.1 Why Indexing?" In `Data Matching: Concepts and
    Techniques for Record Linkage, Entity Resolution, and Duplicate Detection`,
    1.  Springer, 2012.
