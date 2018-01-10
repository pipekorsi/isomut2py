#!/usr/bin/env python3

#################################################
# importing the wrapper
#################################################
import sys,os
#add path for isomut_wrappers.py
#	if not running it from the isomut directory
#	change os.getcwd for the path to it
sys.path.append(os.getcwd()+'/src')

#load the parallel wrapper function
from isomut2_wrappers import run_isomut2

#add path for isomut, if its in the path comment/delete this line
#	if not running it from the isomut directory
#	change os.getcwd for the path to it
os.environ["PATH"] += os.pathsep + os.getcwd() +'/src'


#using parameter dictionary, beacause there are awful lot of parameters
params=dict()


#################################################
# DATA: make sure to modify
#################################################
#reference genome
params['ref_fasta']="/nagyvinyok/adat86/sotejedlik/orsi/tesaro_human_ref/GRCh38pa_EBV.fa"
#input dir output dir
params['input_dir']='/nagyvinyok/adat86/sotejedlik/orsi/tesaro_data/'
params['output_dir']='/nagyvinyok/adat87/home/orsi/IsoMutv2.0/testGitHub/IsoMut2/results'
params['bam_filenames']=['TE01_realign.bam',
                         'TE02_realign.bam',
                         'TE03_realign.bam',
                         'TE04_realign.bam',
                         'TE05_realign.bam',
                         'TE06_realign.bam',
                         'TE07_realign.bam',
                         'TE08_realign.bam',
                         'TE09_realign.bam',
                         'TE10_realign.bam',
                         'TE11_realign.bam',
                         'TE22_realign.bam',
                         'TE23_realign.bam',
                         'TE24_realign.bam',
                         'TE26_realign.bam',
                         'TE27_realign.bam',
                         'TE28_realign.bam',
                         'TE29_realign.bam',
                         'TE30_realign.bam',
                         'TE31_realign.bam',
                         'TE32_realign.bam',
                         'TE33_realign.bam']
# chromosomes to include
params['chromosomes'] = ['20']

#################################################
# PLOIDY: make sure to modify 
# (default ploidy is 2, chr20 is triploid in 
# this case)
#################################################
params['constant_ploidy'] = 3

#################################################
# MUTATION TYPE
# default: searching for unique mutations only
# (if False, running IsoMut takes much longer!)
#################################################
params['unique_mutations_only']=False

#################################################
# OPTIMISATION + HTML REPORT
# default: False
# if True, make sure to set
#     - params['control_samples']
#     - params['FPs_per_genome']
#################################################
params['HTML_report']=True
params['control_samples'] = ['TE01_realign.bam', 'TE23_realign.bam']
params['FPs_per_genome'] = 5

#################################################
# PARALELLISATION
# default values can be kept
#################################################
# minimum number of blocks to run
# usually there will be 10-20 more blocks
params['n_min_block']=100
#number of concurrent processes to run
params['n_conc_blocks']=4

#################################################
# MUTATION CALLING PARAMETERS
# default values can be kept
#################################################
params['min_sample_freq']=0.21
params['min_other_ref_freq']=0.93
params['cov_limit']=5
params['base_quality_limit']=30
params['min_gap_dist_snv']=0
params['min_gap_dist_indel']=20


#################################################
# RUNNING IsoMut2
#################################################
run_isomut2(params)