Introduction
============

Datamatch is a library that facilitates data matching (also known as entity resolution) and deduplication process.
One of the core design goals of this library is to be as extensible as possible, therefore each sub-task is defined
as a separate class, which makes it easy to swap components of the same type and even to write your component
that fit your purpose.

For now, the only classification method supported is threshold-based classification (implemented with
:class:`ThresholdMatcher <datamatch.matchers.ThresholdMatcher>`). However, no matter what methods of classification
are eventually added to this library, concepts such as :ref:`Indices` and :ref:`Filters` will still apply.
Therefore this library is reasonably prepared to be extended to eventually support most data matching use cases.

Installation
------------

.. code-block:: bash

    pip install datamatch

Tutorial
--------

First, download the :download:`DBLP-ACM dataset <DBLP-ACM.zip>` to your working directory and unzip it. (this dataset was
made available by the database group of Prof. Erhard Rahm, see more
`here <https://dbs.uni-leipzig.de/de/research/projects/object_matching/benchmark_datasets_for_entity_resolution>`_)

.. code-block:: bash

    unzip DBLP-ACM.zip -d DBLP-ACM

This dataset contains article names, authors, and years from two different sources, and the perfect matching between
them. Our job is to match articles from those two sources such that we can produce a result as close to the provided
perfect matching as possible.

Load and clean data
~~~~~~~~~~~~~~~~~~~

Preceding the matching step is the data cleaning and standardization step, which is of great importance. We're keeping
this step as simple as possible:

.. code-block:: python

    # example.py
    import pandas as pd
    from datamatch import (
        ThresholdMatcher, StringSimilarity, NoopIndex, ColumnsIndex
    )

    def load_and_clean_acm():
        df = pd.read_csv('DBLP-ACM/ACM.csv')
        # set `id` as the index for this frame, this is important because the library
        # relies on the frame's index to tell which row is which.
        df = df.set_index('id', drop=True)
        # make sure titles are all in lower case
        df.loc[:, 'title'] = df.title.str.strip().str.lower()
        # split author names by comma and sort them before joining them back with comma
        df.loc[df.authors.notna(), 'authors'] = df.loc[df.authors.notna(), 'authors']\
            .map(lambda x: ', '.join(sorted(x.split(', '))))
        return df


    def load_and_clean_dblp():
        df = pd.read_csv('DBLP-ACM/DBLP2.csv')
        # here we do the same cleaning step
        df = df.set_index('id', drop=True)
        df.loc[:, 'title'] = df.title.str.strip().str.lower()
        df.loc[df.authors.notna(), 'authors'] = df.loc[df.authors.notna(), 'authors']\
            .map(lambda x: ', '.join(sorted(x.split(', '))))
        return df

Try matching for the first time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a quick primer on how threshold-based classification work. For each pair of records, produce a similarity score
(ranges from 0 to 1) between each corresponding field, then combine to produce a final similarity score (also ranges
from 0 to 1). You can select different similarity functions for each field depending on their characteristics (see more
similarity functions :ref:`here <Similarities>`). Finally which pairs count as matches depends on an arbitrary threshold
(for similarity score) that you specify. While this classification method is not the gold standard in any way, it is
simple and does not require any training data, which makes it a great fit for many problems. To learn in-depth details,
see [1]_.

You can now start matching data using :class:`ThresholdMatcher <datamatch.matchers.ThresholdMatcher>`. Notice how simple it all is, you just need to specify
the datasets to match and which similarity function to use for each field:

.. code-block:: python

    # example.py

    ...

    if __name__ == '__main__':
        dfa = load_and_clean_acm()
        dfb = load_and_clean_dblp()

        matcher = ThresholdMatcher(NoopIndex(), {
            'title': StringSimilarity(),
            'authors': StringSimilarity(),
        }, dfa, dfb)

And let's wait... Actually, if you have been waiting for like 5 minutes you can stop it now. We're comparing 6 million
pairs of records so it would help tremendously if only there are some ways to increase performance.

Introducing the index
~~~~~~~~~~~~~~~~~~~~~

The index (not to be confused with Pandas Index) is a data structure that helps to reduce the number of pairs to be
compared. It does this by deriving an indexing key from each record and only attempt to match records that have the
same key. Without this technique, matching two datasets with `n` and `m` records, respectively, would take `n x m`
detailed comparisons, which is probably infeasible for most non-trivial use cases. To learn more about indexing, see
[2]_. Another technique to reduce the number of pairs but works the opposite way of indexing is :ref:`filtering <Filters>`.

We have been using :class:`NoopIndex <datamatch.indices.NoopIndex>` which is the same as using no index whatsoever.
We can do better. Notice how the `year` column in both datasets denote the year in which the article was published.
It is very unlikely then that two articles within different years could be the same. Let's employ this `year` column
with :class:`ColumnsIndex <datamatch.indices.ColumnsIndex>`:

.. code-block:: python

    # example.py

    ...

    if __name__ == '__main__':
        ...

        matcher = ThresholdMatcher(ColumnsIndex('year'), {
            'title': StringSimilarity(),
            'authors': StringSimilarity(),
        }, dfa, dfb)

Now, this should run for under 1 or 2 minutes. This is not the best performance that we can wring out of this dataset but
very good for how little effort it requires.

Select a threshold
~~~~~~~~~~~~~~~~~~

The :class:`ThresholdMatcher <datamatch.matchers.ThresholdMatcher>` class does not require a threshold up-front because
usually, it is useful to be able to experiment with different thresholds after the matching is done. Let's see what the
pairs look like:

.. code-block:: python

    # example.py

    ...

    if __name__ == '__main__':
        ...
        print(matcher.get_sample_pairs())

This will print a multi-index frame that shows 5 pairs under each threshold ranges (by defaults: 1.00-0.95, 0.95-0.90,
0.90-0.85, 0.85-0.80, 0.80-0.75, and 0.75-0.70). This should give you an idea of what threshold to use. But there are
more tools at our disposal. If you want to see all pairs, use :meth:`get_all_pairs <datamatch.matchers.ThresholdMatcher.get_all_pairs>`.
If you want to save to Excel for reviewing, use :meth:`save_pairs_to_excel <datamatch.matchers.ThresholdMatcher.save_pairs_to_excel>`.

After a bit of experimentation, I selected `0.577` as my threshold. Let's see the result:

.. code-block:: python

    # example.py

    ...

    if __name__ == '__main__':
        ...

        # this will return each pair as a tuple of index from both datasets
        pairs = matcher.get_index_pairs_within_thresholds(0.577)
        # we can construct a dataframe out of it with similar column names
        # to this dataset's perfect mapping CSV.
        res = pd.DataFrame(pairs, columns=['idACM', 'idDBLP'])\
            .set_index(['idACM', 'idDBLP'], drop=False)

        # load the perfect mapping
        pm = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')\
            .set_index(['idACM', 'idDBLP'], drop=False)

        total = len(dfa) * len(dfb)
        print("total:", total)
        # total: 6001104

        sensitivity = len(pm[pm.index.isin(res.index)]) / len(pm)
        print("sensitivity:", sensitivity)
        # sensitivity: 0.9937050359712231

        specificity = 1 - len(res[~res.index.isin(pm.index)]) / (total - len(pm))
        print("specificity:", specificity)
        # specificity: 0.9999978329288134

The `sensitivity` and `specificity` are not perfect but they're still great considering how simple this matching script
is.

.. [1] Peter Christen. "6.2 Threshold-Based Classification" In `Data Matching: Concepts and Techniques
    for Record Linkage, Entity Resolution, and Duplicate Detection`, 131-133. Springer, 2012.

.. [2] Peter Christen. "4.1 Why Indexing?" In `Data Matching: Concepts and Techniques
    for Record Linkage, Entity Resolution, and Duplicate Detection`, 69. Springer, 2012.