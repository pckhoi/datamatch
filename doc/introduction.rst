Introduction
============

Datamatch is a library that facilitates data matching (also known as entity resolution) and deduplication process.
One of the core design goal of this library is to be as extensible as possible, therefore each sub-task are defined
as a separate class, which makes it easy to swap components of the same type and even to write your own component
that fit your purpose.

Installation
------------

.. code-block:: bash

    pip install datamatch

First taste
-----------

First download the :download:`DBLP-ACM dataset <DBLP-ACM.zip>` to current folder and unzip. (this dataset was made
available by the database group of Prof. Erhard Rahm, see more
`here <https://dbs.uni-leipzig.de/de/research/projects/object_matching/benchmark_datasets_for_entity_resolution>`_)

.. code-block:: bash

    unzip DBLP-ACM.zip -d DBLP-ACM

Utilize the index
-----------------

Better similarity
-----------------