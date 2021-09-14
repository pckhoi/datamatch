:github_url: https://github.com/pckhoi/datamatch/blob/main/doc/deduplication.rst

Deduplication
=============

Datamatch can not only match entities from different datasets but also
deduplicate records from a single dataset. We can reuse the
:download:`DBLP-ACM dataset <DBLP-ACM.zip>` [1]_ from :ref:`Tutorial` to
demonstrate this capability. Again download and unzip if you haven't:

.. code-block:: bash

    unzip DBLP-ACM.zip -d DBLP-ACM

.. unzip:: DBLP-ACM.zip DBLP-ACM

Load data
---------

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

    In [0]: dfb = pd.read_csv('DBLP-ACM/DBLP2.csv', encoding='latin_1')
       ...: # here we do the same cleaning step
       ...: dfb = dfb.set_index('id', drop=True)
       ...: dfb.loc[:, 'title'] = dfb.title.str.strip().str.lower()
       ...: dfb.loc[dfb.authors.notna(), 'authors'] = dfb.loc[dfb.authors.notna(), 'authors']\
       ...:     .map(lambda x: ', '.join(sorted(x.split(', '))))
       ...: dfb

Next let's combine two sources into one and pretend that we have a single
dataset with records to deduplicate:

.. ipython::

    In [0]: # The ACM dataset happens to use numbers as id. We need to convert it to
       ...: # str to ensure the final dataset's index has consistent type.
       ...: dfa.index = dfa.index.astype(str)
       ...: df = pd.concat([dfa, dfb])
       ...: df

Deduplicate
-----------

When `ThresholdMatcher` is given one dataset instead of two, it tries
to deduplicate instead of matching:

.. ipython::

    In [0]: matcher = ThresholdMatcher(ColumnsIndex('year'), {
       ...:     'title': StringSimilarity(),
       ...:     'authors': StringSimilarity(),
       ...: }, df)

All other concepts such as the index and similarity functions apply equally.
:meth:`get_sample_pairs <datamatch.matchers.ThresholdMatcher.get_sample_pairs>`
should still be used to review how pairs within certain thresholds look like.
But in general, the desired result is clusters instead of pairs because unlike
when matching two different datasets, there can be more than two rows that are
identified as the same entity. Methods that return clusters are:

- :meth:`get_clusters_within_threshold <datamatch.matchers.ThresholdMatcher.get_clusters_within_threshold>`:
  returns matching clusters as a multi-index frame. It has the following index
  levels:

  * **cluster_idx**: cluster number.
  * **pair_idx**: pair number within the cluster. All rows within a cluster
    are paired up, each pair are then ordered by descending similarity score.
  * **sim_score**: the similarity score.
  * **row_key**: the row index from the input dataset.

.. ipython::

    In [0]: matcher.get_clusters_within_threshold(0.7).head(30)

- :meth:`save_clusters_to_excel <datamatch.matchers.ThresholdMatcher.save_clusters_to_excel>`:
  saves matching clusters to an Excel file for review. Output is similar to
  :meth:`get_clusters_within_threshold <datamatch.matchers.ThresholdMatcher.get_clusters_within_threshold>`.
- :meth:`get_index_clusters_within_thresholds <datamatch.matchers.ThresholdMatcher.get_index_clusters_within_thresholds>`:
  returns matching clusters as a list. Each cluster is represented by a frozenset
  of row indices. You'll want to use this instead of
  :meth:`get_index_pairs_within_thresholds <datamatch.matchers.ThresholdMatcher.get_index_pairs_within_thresholds>`.

.. ipython::

    In [0]: matcher.get_index_clusters_within_thresholds(0.7)[:10]


.. [1] `DBLP-ACM dataset <https://dbs.uni-leipzig.de/de/research/projects/object_matching/benchmark_datasets_for_entity_resolution>`_
   by the database group of Prof. Erhard Rahm under the `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_
