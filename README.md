# IsoMut2
Unofficial repo of IsoMut2, an updated version of the original [IsoMut](https://github.com/genomicshu/isomut)

Fast and accurate detection of both unique and shared mutations in isogenic samples

---
## New features:
  - a unique ploidy value can be set for any genomic region of any of the investigated samples
  - non-unique mutations can also be detected
  - automatic optimisation of the *S* score threshold using unique mutations only
  - HTML report with figures to illustrate the optimisation process (`examples/HTML_report_example.html`)
  
---
## How to:
### Step 1: prerequisites
  - samtools
  - python 3
  - bam files of the investigated samples, indexed with samtools
  - fasta file of the reference genome, indexed with samtools

### Step 2: compiling IsoMut2

```
git clone https://github.com/pipekorsi/IsoMut2.git
cd IsoMut2/src

gcc -c -O3 isomut2_lib.c fisher.c  -W -Wall
gcc -O3 -o isomut2 isomut2.c isomut2_lib.o  fisher.o -lm -W -Wall

cd ../..
```

### Step 3: modifying parameters in `examples/isomut2_example_script.py`

#### Data: *must be modified*
  - `ref_fasta`: path to reference fasta file (the fai file should be in the same directory)
  - `input_dir`: path to directory with all the investigated bam files
  - `output_dir`: path to results directory
  - `bam_filenames`: list of all the names of the investigated bam files
  
#### Parameters with valid default values:  

##### Use local realignment:
  - `use_local_realignment`: (*default*: False)
    - By default, mutation detection is run only once, with the [samtools mpileup](http://www.htslib.org/doc/samtools-1.2.html) command with option `-B`. This turns of the probabilistic realignment of reads while creating the temporary pileup file, which might result in false positives due to alignment error. To filter out these mutations, setting the above parameter to `True` runs the whole mutation detection pipeline again for possibly mutated positions without the `-B` option as well, and only those mutations are kept that are still present with the probabilistic realignment turned on. Setting `use_local_realignment = True` increases runtime.

##### Investigated chromosomes:
  - `chromosomes`: (*default*: 1, 2, ... 22, X, Y)
    - list of chromosomes to include 
    - if the chromosomes are referred to as "chrN" in your files, make sure to modify the list to exactly match them

##### Ploidy:
  - Investigating non-diploid genomes with constant ploidy:
    - `constant_ploidy`: (*default*: 2)
      - set to the desired value (1 for haploid, 3 for triploid genomes, etc.)
  - Investigating either samples of different ploidies or non-constant ploidies.
    - `ploidy_info_filepath`: (*default*: no_ploidy_info)
      - path to the file containing ploidy information (see below for details)
    
##### Mutation type:
  - `unique_mutations_only`: (*default*: True)
    - to search for non-unique mutations as well, set to `False` (this takes longer!)
  
##### HTML report with optimisation:
  - `HTML_report`: (*default*: False)
    - if you would like to optimise the *S* score and get a HTML report, set to `True` (if `True`, make sure to set the parameters below)
    - To produce a reliable set of final mutations, setting `HTML_report = True` is strongly suggested. However, if further manual filtering of the original results is desired, setting it to `False` results in decreased runtime.
  - `control_samples`: (*default*: None)
    - list of control samples (samples with presumably no unique mutations)
    - only has effect when `HTML_report = True`
  - `FPs_per_genome`: (*default*: None)
    - maximum number of false positive mutations per a single genome
    - only has effect when `HTML_report = True`
    
##### Paralellisation:
  - The genome will be broken up into separate segments (blocks) that can be processed in parallel.
  - `n_min_block`: (*default*: 200)
    - an approximate number of blocks to create (usually there will be slightly more)
  - `n_conc_blocks`: (*default*: 4)
    - number of concurrent blocks to run 
  
##### Mutation calling parameters:
  - `min_sample_freq`: (*default*: 0.21)
    - minimum alternate allele frequency in mutated samples
  - `min_other_ref_freq`: (*default*: 0.95)
    - minimum reference allele frequency in non-mutated samples
  - `cov_limit`: (*default*: 5)
    - minimum sequencing depth in mutated samples
  - `base_quality_limit`: (*default*: 30)
    - minimum base quality
  - `min_gap_dist_snv`: (*default*: 0)
    - minimum genomic distance from an identified SNV
  - `min_gap_dist_indel`: (*default*: 20)
    - minimum genomic distance from an identified indel
  
### Step 4: run the script

---

## Ploidy info file:

To investigate samples of greatly varying ploidies, a bed file of regions of constant ploidies should be created. The basic structure of such a bed file is the following (TAB delim):

```
chrom   chromStart      chromEnd        ploidy
1       0       2161075 3
1       2161076 2610522 4
1       2610523 2760113 5
1       2760114 3210122 4
```

Make sure to include the header. This should be available for every investigated sample with non-constant or non-diploid ploidy, but if a group of samples have the same karyotype, a single bed file is sufficient for the group.

The ploidy info file (set by the `ploidy_info_filepath` parameter) should contain the path to all these bed files with the list of samples belonging to the given karyotype group. For example (TAB delim, sample list comma+space delim):

```
#file_path   sample_names_list
/full/path/to/first_group.bed	S1.bam, S2.bam, S3.bam, S4.bam, S5.bam
/full/path/to/second_group.bed	S6.bam, S7.bam, S8.bam, S9.bam, S10.bam
```

Make sure to include the header.

---

## Coming soon:

- option to automatically create ploidy bed files described above


