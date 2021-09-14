:github_url: https://github.com/pckhoi/datamatch/blob/main/doc/using-variations.rst

Using variations
================

Sometimes it is useful to derive multiple variations from a single record
and try each variation during matching while retaining the highest
similarity score. An example of how this might be useful is when a person's
first name and last name are swapped due to clerical mistakes. You might
want to produce one additional variation for each record where the name
columns are swapped before matching.

The :ref:`variator classes <variators>` do just that while saving you the
extra hassle of adding and rearranging the records. This contrived example
demonstrates how to use a variator:

.. ipython::

    In [0]: import pandas as pd
       ...: from datamatch import ThresholdMatcher, JaroWinklerSimilarity, Swap

    In [0]: df = pd.DataFrame([
       ...:     ['blake', 'lauri'],
       ...:     ['lauri', 'blake'],
       ...:     ['robinson', 'alexis'],
       ...:     ['robertson', 'alexis'],
       ...:     ['haynes', 'terry'],
       ...:     ['terry', 'hayes']
       ...: ], columns=['last', 'first'])
       ...: df

    In [0]: # here we uses Swap to produce a variation that has first and last swapped
       ...: matcher = ThresholdMatcher(NoopIndex(), {
       ...:     'last': JaroWinklerSimilarity(),
       ...:     'first': JaroWinklerSimilarity()
       ...: }, df, variator=Swap('first', 'last'))
       ...: matcher.get_all_pairs()
