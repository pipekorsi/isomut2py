# IsoMut2py: a python module for the detection and analysis of short genomic mutations in NGS data

IsoMut2py is an updated version of the original [IsoMut](https://github.com/genomicshu/isomut) 
software, mainly implemented in python. The most time-consuming parts of the workflow are 
however written in C. 

It can be used in all cases when either the generic karyotype of a sample is sought to be
explored, or short genomic mutations (SNVs and indels) are aimed to be detected. 


## Features:

- easy installation with dependencies using ``pip``
- **karyotype exploration** for a single sample, using a Bayesian approach
- **karyotype comparison** between sample pairs, for a naive identification of CNVs
- **karyotype plots**, **coverage histograms**
- **SNV** and **indel** detection in **single or multiple samples**
- detection of both **unique and shared mutations**
- refined mutation detection based on local ploidy information
- **automatic optimization** based on the user-defined list of control samples 
with easily interpretable figures as sanity checks
-  option for loading and filtering a preexisting set of mutations
- **basic hierarchical clustering** of samples based on the number of shared mutations
- **plots of [SBS, DBS and ID spectra](https://www.biorxiv.org/content/early/2018/05/15/322859)**
- **decomposition of SBS, DBS and ID spectra** to a mixture of reference signatures
using expectation maximization
- **signature composition plots**
- straightforward querying of details of samples in mutated positions

## Dependencies:

1. **samtools**: In order to use the functions for mutation calling or ploidy estimation, 
samtools needs to be installed. However, plotting and filtering of mutations is available 
without samtools.
2. pandas
3. numpy
4. scipy
5. matplotlib 
6. pymc3
7. theano 
8. seaborn 
9. biopython

Other than **samtools**, all dependencies can be automatically installed using ``pip``.

## Installation:

``pip install isomut2py``

## Documentation:

https://isomut2py.readthedocs.io/

## Authors:

Most of the code has been written by Orsolya Pipek, although the C code directly 
inherited from the original [IsoMut](https://github.com/genomicshu/isomut) software 
has been written by Dezso Ribli.

The whole project was done in collaboration of:

- Department of Physics of Complex Systems, Eotvos Lorand University 
(Orsolya Pipek, Dezso Ribli, Istvan Csabai)
- Institute of Enzymology, Research Centre for Natural Sciences, Hungarian Academy of 
Sciences (Adam Poti, David Szuts)
- Center for Biological Sequence Analysis, Department of Systems Biology, 
Technical University of Denmark (Zoltan Szallasi)

Implementation of the Fisher's exact test in C was borrowed from 
[Christopher Chang](https://github.com/chrchang/stats).

## How to cite:

Coming soon.

