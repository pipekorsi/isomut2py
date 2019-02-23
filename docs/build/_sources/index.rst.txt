Welcome to isomut2py's documentation!
=====================================

IsoMut2py is an easy-to-use tool for the detection and postprocessing of mutations from raw NGS sequencing data. It takes sets of aligned short reads (BAM files) as its input and can explore and compare the karyotypes of different samples, detect single nucleotide variations (SNVs), insertions and deletions (indels) in single or multiple samples, optimize the identified mutations whenever provided with a list of control samples, plot mutation counts and spectra on readily interpretable charts and decompose them to predefined reference signatures.

IsoMut2py is an updated version of the original `IsoMut <http://github.com/genomicshu/isomut/>`_ software, mainly implemented in python. The most time-consuming parts of the workflow are however written in C, but also accessible through python wrapper functions.


Contents:

.. toctree::
   :maxdepth: 2
    
   introduction
   getting_started
   use_cases
   external_mutations
   PE_advanced
   postprocessing
   code

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

