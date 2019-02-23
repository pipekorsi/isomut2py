Technical (formatting, IO, etc.) functions
******************************************

Most of these functions can be called as class methods for `MutationCaller objects <code_mutation_calling.rst>`_ and/or `PloidyEstimator objects <code_ploidy_estimation.rst>`_. They are described here individually, so that subtasks can be more easily managed.

Formatting functions
====================

.. automodule:: isomut2py.format
   :members:

IO functions
============

.. automodule:: isomut2py.io
   :members:

Processing functions
====================

(These functions mainly take care of parallelization.)

.. automodule:: isomut2py.process
   :members:

Bayesian inference functions
============================

(These functions are called by the `PloidyEstimator object <code_ploidy_estimation.rst>`_ to fit different theoretical distributions to the actual coverage distribution calculated from the data.)

.. automodule:: isomut2py.bayesian
   :members:

Functions for ploidy comparison
===============================

(These functions perform the comparison of ploidy estimates of two samples for different file formats.)

.. automodule:: isomut2py.compare
   :members:

Functions for loading example parameter settings
================================================

(These functions download example datasets and help load the settings for processing these in a concise way.)

.. automodule:: isomut2py.examples
   :members:


