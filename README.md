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
git clone https://github.com/genomicshu/IsoMut2.git
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
  - `chromosomes`: list of chromosomes to include

#### Ploidy: *default is assuming diploid genomes for all regions of all samples*
  - Investigating non-diploid genomes with constant ploidy:
    - `constant_ploidy`: set to the desired value (1 for haploid, 3 for triploid genomes, etc.)
  - Investigating either samples of different ploidies or non-constant ploidies.
    - `ploidy_info_filepath`: path to the file containing ploidy information (see below for details)
    
#### Mutation type: *default is unique mutations only*
  - `unique_mutations_only`: to search for non-unique mutations as well, set to `False` (this takes longer!)
  
#### HTML report with optimisation: *by default, S score optimisation is not carried out*
  - `HTML_report`: if you would like to optimise the *S* score and get a HTML report, set to `True` (if `True`, make sure to set the parameters below)
  - `control_samples`: list of control samples (samples with presumably no unique mutations)
  - `FPs_per_genome`: maximum number of false positive mutations per a single genome
 
#### Paralellisation: *default values can be kept*
  - The genome will be broken up into separate segments (blocks) that can be processed in parallel.
  - `n_min_block`: an approximate number of blocks to create (usually there will be slightly more)
  - `n_conc_blocks`: number of concurrent block to run 
  
#### Mutation calling parameters: *default values can be kept*
  - `min_sample_freq`: minimum alternate allele frequency in mutated samples
  - `min_other_ref_freq`: minimum reference allele frequency in non-mutated samples
  - `cov_limit`: minimum sequencing depth in mutated samples
  - `base_quality_limit`: minimum base quality
  - `min_gap_dist_snv`: minimum genomic distance from an identified SNV
  - `min_gap_dist_indel`: minimum genomic distance from an identified indel
  
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


