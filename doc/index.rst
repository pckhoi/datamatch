.. Datamatch documentation master file, created by
   sphinx-quickstart on Tue Aug 24 15:29:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pckhoi/datamatch/blob/main/doc/index.rst

Welcome to Datamatch's documentation!
=====================================

Datamatch is a library that facilitates data matching (also known as entity
resolution) and deduplication process. One of the core design goals of this
library is to be as extensible as possible, therefore each sub-task is defined
as a separate class, which makes it easy to swap components of the same type
and even to write your component that fit your purpose.

For now, the only classification method supported is threshold-based
classification (implemented with
:class:`ThresholdMatcher <datamatch.matchers.ThresholdMatcher>`). However,
no matter what methods of classification are eventually added to this library,
concepts such as :ref:`Indices` and :ref:`Filters` will still apply. Therefore
this library is reasonably prepared to be extended to eventually support
most data matching use cases.


.. toctree::
    :caption: Guides

    installation
    tutorial
    deduplication
    filtering
    using-variations

.. toctree::
    :caption: Reference

    reference/matchers
    reference/indices
    reference/similarities
    reference/filters
    reference/variators
    reference/pairers
