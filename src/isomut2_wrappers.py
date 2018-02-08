from __future__ import print_function
from __future__ import division

import multiprocessing
import sys
import time
from datetime import datetime
import subprocess
import os
import glob
import pandas as pd
import numpy as np

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io import BytesIO
import base64

import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")

import pymc3 as pm
import theano.tensor as tt
import scipy as sp
import scipy.stats as stats
from scipy.stats import exponnorm


# command line parameter "-d" for samtools
SAMTOOLS_MAX_DEPTH=1000

def define_parallel_blocks(ref_genome, min_block_no, chrom_list, params=None, level=0):
    """
    Calculate blocks of parallelization on the reference genome.


    The number of returned blocks, are higher than min_block_no.
    No block overlap chromosomes, because samtools does not accept
    region calls overlapping chromosomes.

    Returns a list of tuples, a tuple is a block: [(chr,posmin,posmax),...]
    """

    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Defining parallel blocks ...')
    sys.stdout.flush() #jupyter notebook needs this to have constant feedback

    #check faidx (it has to be there because mpileup needs it too)
    if(glob.glob(ref_genome+'.fai')==[]):
        print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Error please create faidx file for the reference genome')
        exit(1)

    #collect the length of chromosomes from the faidx file
    chroms,lens=[],[]
    with open(ref_genome+'.fai') as f_h:
        for line in f_h:
            chrom,leng=line.split('\t')[0],line.split('\t')[1]
            if (chrom_list == None or chrom in set(chrom_list)):
                chroms.append(chrom)
                lens.append(int(leng))

    #set maximum block size
    BLOCKSIZE=int(np.floor(sum(lens)/min_block_no))

    if (params != None):
        params['genome_length'] = sum(lens)

    #calculate blocks
    blocks=[]
    #loop over chroms
    for chrom,leng in zip(chroms,lens):
        pointer=0
        #until chrom is chopped into pieces
        while (pointer < leng):
            block_size=min(BLOCKSIZE,leng-pointer)
            blocks.append([chrom,pointer,pointer+block_size])
            pointer += block_size

    #return chr, posmin, posmax of blocks
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Done\n')
    sys.stdout.flush()
    return blocks

def temp_file_from_block(chrom,from_pos,to_pos,
                         input_dir,bam_file,output_dir,
                         ref_genome,
                         windowsize,
                         shiftsize,
                         min_noise,
                         base_quality_limit,
                         bedfile,
                         samtools_flags):
    """
    Run the samtools + the PEprep C application in the system shell.

    Creating temporary files for ploidy estimating by filtering positions
    based on their reference nucleotide frequency.

    One run is only on a section of a chromosome.
    With bedfile, only the positions of the bedfile are analyzed.
    Automatically logs samtools stderr output to output_dir/samtools.log
    Results are saved to a file which name is created from the block chr,posform,posto
    """

    cov_est_filename = output_dir + '/' + bam_file.split('.bam')[0] + '_tmp_HCE.txt'

    #build the command
    cmd=' samtools  mpileup'
    if (samtools_flags != None):
        cmd += ' ' + samtools_flags
    cmd+=' -f ' +ref_genome
    cmd+=' -r '+chrom+':'+str(from_pos)+'-'+str(to_pos)+' '
    if(bedfile!=None):
        cmd+=' -l '+bedfile+' '
    cmd+=input_dir+'/'+bam_file +' '
    cmd+=' 2>> '+output_dir+'/samtools.log | PEprep '
    cmd+=' '.join(map(str,[windowsize,shiftsize,min_noise,
                           base_quality_limit, cov_est_filename,bam_file])) +' '
    cmd+=' > ' +output_dir+'/PEtmp_blockfile_'+ chrom+'_'+str(from_pos)+'_'+str(to_pos)+'.csv'

    return subprocess.check_call(cmd,shell=True)


def PE_on_chrom(PEParams, chrom, prior_dict):

    """
    Run ploidy estimation on a given chromosome.

    """
    df = pd.read_csv(PEParams['output_dir']+'/' + 'PEtmp_fullchrom_' + chrom + '.txt', sep='\t', names=['chrom', 'pos', 'cov', 'mut_freq']).sort_values(by='pos')

    df['chrom'] = df['chrom'].apply(str)
    pos_all = np.array(list(df['pos']))
    total_ploidy_a = np.array([0]*len(pos_all))
    total_loh_a = np.array([0]*len(pos_all))
    est_num_a = np.array([0]*len(pos_all))
    posstart = df['pos'].min()
    posmax = df['pos'].max()
    while (posstart < posmax):
        p, loh = PE_on_range(df, posstart, posstart+PEParams['windowsize_PE'], prior_dict['mu'], prior_dict['sigma'], prior_dict['p'], PEParams)
        if (p > 0):
            total_ploidy_a += p*(pos_all>=posstart)*(pos_all<=posstart+PEParams['windowsize_PE'])
            total_loh_a += loh*(pos_all>=posstart)*(pos_all<=posstart+PEParams['windowsize_PE'])
            est_num_a += 1*(pos_all>=posstart)*(pos_all<=posstart+PEParams['windowsize_PE'])
        posstart += PEParams['shiftsize_PE']
    df['total_ploidy'] = total_ploidy_a
    df['total_loh'] = total_loh_a
    df['est_num'] = est_num_a

    df['ploidy'] = (df['total_ploidy']/df['est_num']).round()
    df['LOH'] = (df['total_loh']/df['est_num']).round()

    df = df[(~df['ploidy'].isnull()) & (~df['LOH'].isnull())]
    df['ploidy'] = df['ploidy'].astype(int)
    df['LOH'] = df['LOH'].astype(int)

    df[['chrom', 'pos', 'cov', 'mut_freq', 'ploidy', 'LOH']].to_csv(PEParams['output_dir'] + '/PE_fullchrom_' + chrom + '.txt', sep='\t', index=False)
    # return df[['chrom', 'pos', 'cov', 'mut_freq', 'ploidy', 'LOH', 'total_ploidy', 'est_num']]

def PE_on_range(dataframe, rmin, rmax, all_mu, all_sigma, prior, PEParams):

    def px0(mean_diff):
        k = -1*np.log(0.5)/1000
        return np.exp(-1*k*mean_diff)

    temp = dataframe[(dataframe['pos']>= rmin) & (dataframe['pos']<= rmax)
                     & (dataframe['cov']>=PEParams['cov_min']) & (dataframe['cov']<=PEParams['cov_max'])]

    if (temp.shape[0] == 0 or temp['pos'].max() == temp['pos'].min()):
        return 0, 0

    #get cov mean
    cov_mean = temp['cov'].mean()

    posteriors = []
    for ploidy in range(len(mu)):
        likelihood = stats.norm.pdf(cov_mean, loc=all_mu[ploidy], scale=all_sigma[ploidy])
        posteriors.append(likelihood*prior[ploidy])
    posteriors = np.array(posteriors)
    most_probable_ploidy = np.argmax(posteriors) + 1


    new_mean_dip = 0.5428682460756673

    mf_list = np.array(temp[temp['mut_freq'] >= 0.5]['mut_freq'])
    mf_mean = np.mean(mf_list)
    max_num_of_minority_alleles = int(np.floor(most_probable_ploidy/2))

    # get average difference between measurement points

    pos_all = np.array(temp.sort_values(by=['pos'])['pos'])
    pos_diff = np.diff(pos_all)
    prob_of_no_minority_allele = 1-px0(pos_diff.mean())
    prob_of_other_minority_allele = (1-prob_of_no_minority_allele)/max_num_of_minority_alleles

    minority_alleles_prior = [prob_of_no_minority_allele]+[prob_of_other_minority_allele]*max_num_of_minority_alleles
    minority_alleles_posterior = []
    number_of_unsure_bases = 2

    likelihoods_all = []
    for minority_alleles in range(max_num_of_minority_alleles+1):
        if (minority_alleles == 0):
            likelihood_ma = 1
        elif (most_probable_ploidy%2 == 0 and minority_alleles == most_probable_ploidy/2):
            likelihood_ma = stats.norm.pdf(mf_mean,
                                        loc=new_mean_dip,
                                        scale=(number_of_unsure_bases/all_mu[most_probable_ploidy-1])*(2/3))/I[0]
        else:
            likelihood_ma = stats.norm.pdf(mf_mean,
                                        loc=1-(minority_alleles/most_probable_ploidy),
                                        scale=(number_of_unsure_bases/all_mu[most_probable_ploidy-1])*(2/3))
        likelihoods_all.append(likelihood_ma)
        minority_alleles_posterior.append(likelihood_ma*minority_alleles_prior[minority_alleles])
    minority_alleles_posterior = np.array(minority_alleles_posterior)

    most_probable_ma_num = np.argmax(minority_alleles_posterior)
    if (most_probable_ma_num != 1 and most_probable_ploidy > 1):
        most_probable_loh = 1
    else:
        most_probable_loh = 0

    return most_probable_ploidy, most_probable_loh

# def PE_on_range(dataframe, rmin, rmax, haploid_cov, snp_density):
#
#     """
#     Run ploidy estimation on a given range of a chromosome.
#
#     """
#
#     temp = dataframe[(dataframe['pos']>= rmin) & (dataframe['pos']<= rmax)
#                      & (dataframe['cov']>=5) & (dataframe['cov']<=200)]
#     if (temp.shape[0] == 0 or temp['pos'].max() == temp['pos'].min()):
#         return 0, 0
#
#     mf_list = np.array(temp['mut_freq'])
#     if (float(temp[(temp['mut_freq'] > 0.45) & (temp['mut_freq'] < 0.55)].shape[0])/temp.shape[0] < 0.2): # if values are not centralised around 0.5, transform to [0,0.5]
#         mf_list = 1-mf_list*(mf_list>0.5)+(mf_list-1)*(mf_list<=0.5)
#     mc = temp['cov'].mean()
#     d_c = temp['cov'].std()
#     # mrnf = 0.5-np.abs(0.5-temp['mut_freq'].mean())
#     mrnf = np.mean(mf_list)
#     d_rnf = np.std(mf_list)
#     p_c = mc/haploid_cov
#     p_rnf = float(1)/mrnf
#     d_prnf = d_rnf/(mrnf)**2
#     d_pc = d_c/haploid_cov
#     error_pc = d_pc/p_c
#     error_prnf = d_prnf/p_rnf
#     if (error_pc+error_prnf != 0 and not np.isnan(error_pc) and not np.isnan(error_prnf)):
#         error_pc_corr = error_pc/(error_pc+error_pc+error_prnf)
#         error_prnf_corr = error_prnf/(error_pc+error_pc+error_prnf)
#         if (error_prnf != 0 and error_pc != 0):
#             s = error_prnf/error_pc
#             x = 1/(2+1/s)
#         else:
#             s = 1
#             x = 1/3
#     else:
#         error_pc_corr = 0
#         error_prnf_corr = 0
#         s = 1
#         x = 1/3
#     ##### LOH regions are not detected for ploidies higher than 4
#
#     # what to do when rnf suggests p = 1:
#     if (round(p_rnf) == 1 and (np.abs(2-p_c) < np.abs(1-p_c))):
#         return 2, 1 # diploid, but LOH
#     elif (round(p_rnf) == 1 and (np.abs(2-p_c) > np.abs(1-p_c))):
#         return 1, 0 # haploid, no LOH
#     # what to do when rnf suggests p = 2:
#     elif (round(p_rnf) == 2 and (np.abs(4-p_c) < np.abs(2-p_c))):
#         return 4, 1 # tetraploid with two identical alleles (LOH)
#     elif (round(p_rnf) == 2 and (np.abs(4-p_c) > np.abs(2-p_c))):
#         p = 2
#         loh = 0
#         # if (float(temp.shape[0])/(temp['pos'].max()-temp['pos'].min())< snp_density):
#         #     loh = 1
#         return p, loh
#     # what to do when rnf and cov suggest very different p values:
#     elif (np.abs(mc/haploid_cov-1/mrnf) >= 2.5):
#         # p = np.min([int(round(p_c)), int(round(p_rnf))]) # use the smaller one
#         p = int(round(p_c)) # or use the one suggested by coverage (this should be the most reliable if the mapping is correct, if not, rnf is worhtless as well)
#         loh = 0
#         # if (p > 1 and p < 5 and float(temp.shape[0])/(temp['pos'].max()-temp['pos'].min())< 0.0015 and float(temp[temp['mut_freq']>0.5].shape[0])/temp[temp['mut_freq']<0.5].shape[0] < 0.3):
#         if (p > 1 and p < 5 and float(temp.shape[0])/(temp['pos'].max()-temp['pos'].min())< snp_density):
#             loh = 1
#         return p, loh # something strange, LOH questionable
#     # what to do in general
#     else:
#         # if (round(p_c*(1-error_pc_corr)+p_rnf*(1-error_prnf_corr)) == round(p_c*(1-error_pc_corr)+p_rnf*(1-error_prnf_corr))):
#         if (not np.isnan(p_c) and not np.isnan(p_rnf)):
#             # what if the relative error of any of them is larger than 1? (very unreliable)
#             if (error_pc > 1 and error_prnf < 1): # if rnf is more reliable, let that win - but should we?
#                 p = int(round(p_rnf))
#             elif (error_pc < 1 and error_prnf > 1): # if rnf is unreliable, use coverage
#                 p = int(round(p_c))
#             else: # both of them are reliable, calculate accurately
#                 p = int(round(p_c*(1-error_pc_corr)+p_rnf*(1-error_prnf_corr)))
#                 # p = int(round(p_c*x*2+p_rnf*x/s))
#             loh = 0
#             # if (p > 1 and p < 5 and float(temp.shape[0])/(temp['pos'].max()-temp['pos'].min())< 0.0015 and float(temp[temp['mut_freq']>0.5].shape[0])/temp[temp['mut_freq']<0.5].shape[0] < 0.3):
#             if (p > 1 and p < 5 and float(temp.shape[0])/(temp['pos'].max()-temp['pos'].min())< snp_density and round(p_c) != round(p_rnf)):
#                 loh = 1
#             return p, loh
#         else:
#             return 0, 0


# def PE_on_chrom(chrom,
#                 output_dir,
#                 windowsize_PE,
#                 shiftsize_PE,
#                 haploid_cov,
#                 snp_density):
#     """
#     Run ploidy estimation on a given chromosome.
#
#     """
#
#     df = pd.read_csv(output_dir + '/PEtmp_fullchrom_' + chrom + '.txt', sep='\t', names=['chrom', 'pos', 'cov', 'mut_freq']).sort_values(by='pos')
#     df['chrom'] = df['chrom'].apply(str)
#     pos_all = np.array(list(df['pos']))
#     total_ploidy_a = np.array([0]*len(pos_all))
#     total_loh_a = np.array([0]*len(pos_all))
#     est_num_a = np.array([0]*len(pos_all))
#     posstart = df['pos'].min()
#     posmax = df['pos'].max()
#     while (posstart < posmax):
#         p, loh = PE_on_range(df, posstart, posstart+windowsize_PE, haploid_cov, snp_density)
#         if (p > 0):
#             total_ploidy_a += p*(pos_all>=posstart)*(pos_all<=posstart+windowsize_PE)
#             total_loh_a += loh*(pos_all>=posstart)*(pos_all<=posstart+windowsize_PE)
#             est_num_a += 1*(pos_all>=posstart)*(pos_all<=posstart+windowsize_PE)
#         posstart += shiftsize_PE
#     df['total_ploidy'] = total_ploidy_a
#     df['total_loh'] = total_loh_a
#     df['est_num'] = est_num_a
#
#     df['ploidy'] = (df['total_ploidy']/df['est_num']).round()
#     df['LOH'] = (df['total_loh']/df['est_num']).round()
#
#     df = df[(~df['ploidy'].isnull()) & (~df['LOH'].isnull())]
#     df['ploidy'] = df['ploidy'].astype(int)
#     df['LOH'] = df['LOH'].astype(int)
#
#     df[['chrom', 'pos', 'cov', 'mut_freq', 'ploidy', 'LOH']].to_csv(output_dir + '/PE_fullchrom_' + chrom + '.txt', sep='\t', index=False)

    # remove old file
    # return subprocess.check_call('rm ' + output_dir + '/PEtmp_fullchrom_' + chrom + '.txt',shell=True)


def PE_prepare_temp_files(PEParams, level=0):

    """ Prepare temporary files for ploidy estimation, by averaging coverage in moving windows for the whole genome and collecting
    positions with reference allele frequencies in the [min_noise, 1-min_noise] range."""

    starting_time = datetime.now()
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Ploidy estimation on sample')

    #check for bedfile argument:
    if ('bedfile' not in PEParams):
        PEParams['bedfile']=None
    if ('samtools_flags' not in PEParams):
        PEParams['samtools_flags']=' -B -d '+ str(SAMTOOLS_MAX_DEPTH) + ' '
    if ('chromosomes' not in PEParams):
        PEParams['chromosomes']=[str(i) for i in range(1,23)]+['X', 'Y']

    #define blocks and create args
    blocks=define_parallel_blocks(PEParams['ref_fasta'],PEParams['n_min_block'],PEParams['chromosomes'], params=None, level=level+1)
    args=[]
    for block in blocks:
        args.append([ block[0],block[1],block[2],
                     PEParams['input_dir'],PEParams['bam_filename'],
                     PEParams['output_dir'],PEParams['ref_fasta'],
                     PEParams['windowsize'],PEParams['shiftsize'],
                     PEParams['min_noise'], PEParams['base_quality_limit'],
                     PEParams['bedfile'],PEParams['samtools_flags']])

    #create dir
    print('\t'*level +  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - (All output files will be written to ' + PEParams['output_dir'] + ')\n')
    if(glob.glob(PEParams['output_dir'])==[]):
        subprocess.call(['mkdir',PEParams['output_dir']])

    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Generating temporary files, number of blocks to run: '+ str(len(args)))
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +  ' - Currently running: ', end=' ')
    #start first n concurrent block
    procs=[]
    for i in range(PEParams['n_conc_blocks']):
        procs.append(multiprocessing.Process(target=temp_file_from_block, args=args[len(procs)]))
        procs[-1].start()
        print(str(len(procs)), end=" ")
        sys.stdout.flush()

    # when one finished restart the next
    while (len(procs) != len(args)):
        for i in range(1,PEParams['n_conc_blocks']+1):
            #one finished start another one
            if(procs[-i].is_alive() == False and len(procs) != len(args)):
                procs.append(multiprocessing.Process(target=temp_file_from_block,args=args[len(procs)] ))
                procs[-1].start()
                print(str(len(procs)), end=" ")
                sys.stdout.flush()
        time.sleep(0.1)

    #wait now only the last ones running
    finished = False
    while( not finished):
        finished = True
        for proc in procs:
            if (proc.is_alive() == True):
                finished = False
        time.sleep(0.1)

    print('\n')
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Temporary files created, merging, cleaning up...\n')

    # collecting temp position files to single files for each chromosome
    for c in PEParams['chromosomes']:
        cmd = 'cat ' + PEParams['output_dir'] + '/PEtmp_blockfile_' + c + '_* > ' + PEParams['output_dir'] + '/PEtmp_fullchrom_' + c + '.txt'
        subprocess.check_call(cmd,shell=True)
        cmd = 'rm ' + PEParams['output_dir'] + '/PEtmp_blockfile_' + c + '_*'
        subprocess.check_call(cmd,shell=True)

    # print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Estimating haploid coverage by fitting an infinite mixture model to the coverage distribution...\n')
    # raw_hc = estimate_hapcov_infmix()
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Raw estimate for the haploid coverage: ' + str(raw_hc) + '\n')
    # print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Fitting equidistant Gaussians to the coverage distribution using the raw estimate as prior...\n')
    # distribution_dict = fit_gaussians()
    # print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Estimating local ploidy using the previously determined Gaussians as priors...\n')
    # ploidy_estimation_on_genome()
    #
    # finish_time = datetime.now()
    # total_time = finish_time-starting_time
    # total_time_h = int(total_time.seconds/3600)
    # total_time_m = int((total_time.seconds%3600)/60)
    # total_time_s = (total_time.seconds%3600)%60
    # print('\n'+'\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Ploidy estimation finished. (' + str(total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), ' + str(total_time_m) + ' min(s), ' + str(total_time_s) + ' sec(s).)')


    # estimating haploid coverage
    # cmd = 'cat ' + PEParams['output_dir'] + '/*_tmp_HCE.txt > ' + PEParams['output_dir'] + '/raw_est_hapcov_file.txt'
    # subprocess.check_call(cmd,shell=True)
    # cmd = 'rm ' + PEParams['output_dir'] + '/*_tmp_HCE.txt'
    # subprocess.check_call(cmd,shell=True)
    # df = pd.read_csv(PEParams['output_dir'] + '/raw_est_hapcov_file.txt', sep='\t', names=['dip_cov_total', 'dip_count', 'trip_cov_total', 'trip_count'])
    # avg_dip_cov = float(df['dip_cov_total'].sum())/df['dip_count'].sum()
    # avg_trip_cov = float(df['trip_cov_total'].sum())/df['trip_count'].sum()
    # AVG_HAP_COV = avg_dip_cov/2
    # if (avg_trip_cov < avg_dip_cov): # "diploid" regions probably tetraploid
    #     AVG_HAP_COV = avg_trip_cov/3
    # elif ((avg_trip_cov-avg_dip_cov)/avg_dip_cov < 0.3): # "triploid" regions probably diploid ones with skewed mut_freqs
    #     AVG_HAP_COV = avg_dip_cov/2
    # else:
    #     AVG_HAP_COV = np.mean([avg_dip_cov/2, avg_trip_cov/3])

    # if ('avg_hap_cov' not in PEParams):
    #     print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Raw estimate of haploid coverage: ' + str(AVG_HAP_COV))
    #     return AVG_HAP_COV
    # else:
    #     print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - User-defined haploid coverage: ' + str(PEParams['avg_hap_cov']))
    #     return PEParams['avg_hap_cov']

def get_ploidy_ranges_for_chrom(outputdir, chrom, chr_list, before_list, after_list, pl_list, loh_list, chrom_len_dict):
    """
        Gets ranges of constant ploidies from a file of positional ploidy data.
        At breakpoints, the rounded average of the two bordering positions are taken as the breakpoint position.
    """

    df = pd.read_csv(outputdir + '/PE_fullchrom_' + chrom + '.txt', sep='\t').sort_values(by='pos')
    p = np.array(list(df['pos']))
    pl = np.array(list(df['ploidy']))
    loh = np.array(list(df['LOH']))
    pl_loh = np.array([str(pl_c)+','+str(loh_c) for pl_c,loh_c in zip(pl, loh)])
    pl_loh_change = np.where(pl_loh[:-1] != pl_loh[1:])[0]

    before_pos = 0
    after_pos = 0
    for i in range(len(pl_loh_change)):
        after_idx = pl_loh_change[i]
        after_pos = int(round(np.mean([p[after_idx], p[after_idx+1]])))
        chr_list.append(chrom)
        before_list.append(before_pos)
        after_list.append(after_pos)
        pl_list.append(pl[after_idx])
        loh_list.append(loh[after_idx])
        before_pos = after_pos+1
    chr_list.append(chrom)
    before_list.append(before_pos)
    after_list.append(chrom_len_dict[chrom])
    pl_list.append(pl[-1])
    loh_list.append(loh[-1])

    return chr_list, before_list, after_list, pl_list, loh_list

def get_bed_format_for_sample(PEParams):
    """
        Creates bed file of ploidies for a given sample from a file of positional ploidy data.
    """

    chrom_len_dict = {c:l for c,l in zip(PEParams['chromosomes'], PEParams['chrom_length'])}
    chr_list = []
    before_list = []
    after_list = []
    pl_list = []
    loh_list = []
    for c in PEParams['chromosomes']:
        chr_list, before_list, after_list, pl_list, loh_list = get_ploidy_ranges_for_chrom(PEParams['output_dir'], c, chr_list, before_list, after_list, pl_list, loh_list, chrom_len_dict)
    df = pd.DataFrame()
    df['chrom'] = chr_list
    df['chromStart'] = before_list
    df['chromEnd'] = after_list
    df['ploidy'] = pl_list
    df['LOH'] = loh_list
    df.to_csv(PEParams['output_dir']+'/'+PEParams['bam_filename'].split('.bam')[0]+'_ploidy.bed', index=False)

def estimate_hapcov_infmix(PEParams, level=0):

    # coverage distribution from temporary files (the number of points is decreased for faster inference)
    covs = np.array([])
    for c in PEParams['chromosomes']:
        df = pd.read_csv(PEParams['output_dir']+'/' + 'PEtmp_fullchrom_' + c + '.txt', sep='\t', names=['chrom', 'pos', 'cov', 'rnf'])
        covs = np.append(covs, np.array(list(df['cov'])))
    covs = covs[(covs<PEParams['max_cov'])*(covs>PEParams['min_cov'])]
    covs_few = np.random.choice(covs, 2000)

    K = 20
    number_of_chains = 10
    iterations = 20000
    burn_beginning = 15000

    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    def fit_infinite_mixture_model(coverage_dist, K, number_of_chains_to_use, number_of_iterations, burn_period):
        covdist_standard = (coverage_dist - coverage_dist.mean()) / coverage_dist.std()
        N = len(covdist_standard)
        with pm.Model() as model:
            alpha = pm.Gamma('alpha', 1., 1.)
            beta = pm.Beta('beta', 1., alpha, shape=K)
            w = pm.Deterministic('w', stick_breaking(beta))

            tau = pm.Gamma('tau', 1., 1., shape=K)
            lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
            mu = pm.ExGaussian('mu', mu=-4, sigma=np.sqrt(1/(lambda_ * tau)), nu=5, shape=K)
            obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                                   observed=covdist_standard)
        with model:
            step1 = pm.Metropolis(vars=[alpha, beta, tau, lambda_, mu])
            tr = pm.sample(number_of_iterations, step=[step1], njobs=number_of_chains_to_use, progressbar=False)

        trace = tr[burn_period:]
        return trace

    trace = fit_infinite_mixture_model(coverage_dist=covs_few,
                                  K = K,
                                  number_of_chains_to_use = number_of_chains,
                                  number_of_iterations = iterations,
                                  burn_period = burn_beginning)
    chains_to_use = [c for c in range(number_of_chains)]
    hc_all_ests  = []
    for c in chains_to_use:
        comp_weights = trace.get_values('w', chains=[c]).mean(axis=0)
        comp_weights_sortidx = np.argsort(comp_weights)
        standard_means = trace.get_values('mu', chains=[c]).mean(axis=0)[comp_weights_sortidx]
        true_means = standard_means * covdist.std() + covdist.mean()
        hc_maxweight = true_means[0]/(np.round(true_means[0]/true_means[true_means>PEParams['cov_min']].min()))
        hc_all_ests.append(hc_maxweight)
    hc_all_ests = np.array(hc_all_ests)
    hc_maxweight = np.percentile(a=hc_all_ests[np.isfinite(hc_all_ests)], q=PEParams['hc_percentile'])
    return hc_maxweight, covs_few

def fit_gaussians(PEParams, estimated_hapcov, coverage_distribution, level=0):

    def get_samples(coverage_distribution, estimated_haploid_cov, number_of_iterations, burn_period):
        K = 7
        halfwidth_of_uniform = 0.2
        model = pm.Model()
        with model:
            p = pm.Dirichlet('p', a=np.array([1., 1., 1., 1., 1., 1., 1.]), shape=K)
            c1 = pm.Uniform('c1', (1-halfwidth_of_uniform)*estimated_haploid_cov, (1+halfwidth_of_uniform)*estimated_haploid_cov)
            means = tt.stack([c1, c1*2, c1*3, c1*4, c1*5, c1*6, c1*7])
            order_means_potential = pm.Potential('order_means_potential',
                                                 tt.switch(means[1]-means[0] < 0, -np.inf, 0)
                                                 + tt.switch(means[2]-means[1] < 0, -np.inf, 0))
            sds = pm.Uniform('sds', lower=0, upper=estimated_haploid_cov/2, shape=K)
            category = pm.Categorical('category',
                                      p=p,
                                      shape=len(coverage_distribution))
            points = pm.Normal('obs',
                               mu=means[category],
                               sd=sds[category],
                               observed=coverage_distribution)
        with model:
            step1 = pm.Metropolis(vars=[p, sds, means])
            step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2, 3, 4, 5, 6])
            tr = pm.sample(number_of_iterations, step=[step1, step2], progressbar=False)
        trace = tr[burn_period:]
        return trace

    iterations2 = 20000
    burn_beginning2 = 15000

    trace2 = get_samples(coverage_distribution=coverage_distribution,
                       estimated_haploid_cov = estimated_hapcov,
                       number_of_iterations = iterations2,
                       burn_period = burn_beginning2)

    std_trace = trace2.get_values('sds', chains=[0])
    p_trace = trace2.get_values('p', chains=[0])
    sigma = std_trace.mean(axis=0)
    p = p_trace.mean(axis=0)
    mu = np.array([trace2.get_values('c1', chains=[0]).mean()*(i+1) for i in range(7)])

    prior_dict = {'mu': mu, 'sigma': sigma, 'p': p}
    return prior_dict



def ploidy_estimation(PEParams, level=0):

    """
        Runs ploidy estimation pipeline for the input BAM file.

        In the first step, temporary files are generated by scanning the BAM file with a window and calculating average coverages. Positions where the reference
        allele frequency is in the range [min_noise, 1-min_noise] are kept for later use to reduce data.
        In the second step, an infinite mixture model is fitted to the coverage distribution of the remaining positions with Bayesian inference. A raw
        estimate of the haploid coverage is determined from the model.
        In the third step, seven equidistant Gaussians are fitted to the coverage distribution, using the previously determined raw haploid coverage as a prior for
        the center of the first Gaussian.
        In the forth step, the genome is scanned again with a window and local ploidies are calculated using the above Gaussians as priors and applying
        a maximum likelihood technique.
        In the fifth step, a final bed file and a HTML report of the results are generated.
    """

    starting_time = datetime.now()
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Ploidy estimation for file ' + PEParams['bam_filename'])
    print('\n')



    PE_prepare_temp_files(PEParams, level=level+1)

    print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Estimating haploid coverage by fitting an infinite mixture model to the coverage distribution...\n')
    raw_hc, covdist = estimate_hapcov_infmix(PEParams, level=level+2)
    print('\t'*(level+2) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Raw estimate for the haploid coverage: ' + str(raw_hc) + '\n')
    print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Fitting equidistant Gaussians to the coverage distribution using the raw estimate as prior...\n')
    distribution_dict = fit_gaussians(PEParams, raw_hc, covdist, level=level+2)
    print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Estimating local ploidy using the previously determined Gaussians as priors on chromosomes: ', end=' ')
    for c in PEParams['chromosomes']:
        print(c,end=" ")
        sys.stdout.flush()
        PE_on_chrom(PEParams=PEParams, chrom=c, prior_dict=distribution_dict)

    print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Generating final bed file... \n')
    get_bed_format_for_sample(PEParams)

    finish_time = datetime.now()
    total_time = finish_time-starting_time
    total_time_h = int(total_time.seconds/3600)
    total_time_m = int((total_time.seconds%3600)/60)
    total_time_s = (total_time.seconds%3600)%60
    print('\n'+'\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Ploidy estimation finished. (' + str(total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), ' + str(total_time_m) + ' min(s), ' + str(total_time_s) + ' sec(s).)')

    # print('\n')
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Preliminary ploidy estimation on chromosomes: ', end=' ')
    # for c in PEParams['chromosomes']:
    #     print(c,end=" ")
    #     sys.stdout.flush()
    #     PE_on_chrom(chrom=c, haploid_cov=avg_hap_cov, output_dir=PEParams['output_dir'], shiftsize_PE=PEParams['shiftsize_PE'], windowsize_PE=PEParams['windowsize_PE'], snp_density = 0.0015)

    # print('\n')
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Re-estimating average haploid coverage, SNP density... ')
    # dipcovmean=[]
    # snp_dens = []
    # for c in PEParams['chromosomes']:
    #     df = pd.read_csv(PEParams['output_dir'] + '/PE_fullchrom_' + c + '.txt', sep='\t')
    #     if (df[df['ploidy']==2].shape[0] > 0):
    #         dipcovmean.append(df[df['ploidy']==2]['cov'].mean())
    #     snp_dens.append(float(df.shape[0])/(df['pos'].max()-df['pos'].min()))
    # snp_density_corr = np.mean(snp_dens)*0.4
    # avg_hap_cov = np.mean(dipcovmean)/2
    # print('\n')
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - New estimation of average haploid coverage: ' + str(avg_hap_cov))
    # print('\n')
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Estimated maximum SNP density in LOH regions: ' + str(snp_density_corr))
    #
    # print('\n')
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Final ploidy estimation on chromosomes: ', end=' ')
    # for c in PEParams['chromosomes']:
    #     print(c, end=" ")
    #     sys.stdout.flush()
    #     PE_on_chrom(chrom=c, haploid_cov=avg_hap_cov, output_dir=PEParams['output_dir'], shiftsize_PE=PEParams['shiftsize_PE'], windowsize_PE=PEParams['windowsize_PE'], snp_density=snp_density_corr)
    #
    # print('\n')
    # print('\t'*(level+1) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Generating final bed file... \n')
    # get_bed_format_for_sample(PEParams)
    #
    # print('\n')
    # print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Done')

# def plot_karyotype_detailed(PEParams):
#     for c in PEParams['chromosomes']:
#         df = pd.read_csv(PEParams['output_dir'] + '/PE_fullchrom_' + c + '.txt', sep='\t')
#
#         covcolor = '#FFD700'
#         rnfcolor = (209 / 255., 10 / 255., 124 / 255.)
#         rnfalpha = 0.4
#         guidelinecolor = '#E6E6FA'
#
#         p = list(df.sort_values(by='pos')['pos'])
#         cov = list(df.sort_values(by='pos')['cov'])
#         rf = list(df.sort_values(by='pos')['mut_freq'])
#         pl = list(df.sort_values(by='pos')['ploidy'])
#         pl_i = list(1/np.array(pl))
#         loh = np.array(list(df.sort_values(by='pos')['LOH']))
#         loh_change = np.where(loh[:-1] != loh[1:])[0]
#
#         f, ax1 = plt.subplots()
#         f.set_size_inches(20,10)
#         ax2 = ax1.twinx()
#         for i in range(len(loh_change)):
#             if (i==0 and loh[loh_change[i]] == 1):
#                 w = p[loh_change[i]]-p[0]
#                 k = Rectangle((p[0], 0), w, 1, alpha=0.1, facecolor='black', edgecolor='none')
#                 ax2.add_patch(k)
#             if (loh[loh_change[i]] == 0):
#                 if (i == len(loh_change)-1):
#                     w = max(p)-p[loh_change[i]]
#                 else:
#                     w = p[loh_change[i+1]]-p[loh_change[i]]
#                 k = Rectangle((p[loh_change[i]], 0), w, 1, alpha=0.1, facecolor='black', edgecolor='none')
#                 ax2.add_patch(k)
#
#         ax1.plot(p, cov, c=covcolor)
#
#         for i in range(2,10):
#             ax2.plot(p, [1-float(1)/i]*len(p), c=guidelinecolor)
#             ax2.plot(p, [float(1)/i]*len(p), c=guidelinecolor)
#         ax2.scatter(p, rf, c='none', edgecolor=rnfcolor, alpha=rnfalpha)
#         ax2.scatter(p, pl_i, c='none', edgecolor='black', alpha=1)
#
#         if ('vert_lines' in kwargs):
#             for l in kwargs['vert_lines']:
#                 ax2.plot([l, l], [0, 1], c='black', lw=4)
#
#         ax2.set_ylabel('reference base frequency\n', size=15, color=rnfcolor)
#         ax1.set_xlabel('\n\ngenomic position', size=15)
#         ax2.yaxis.set_tick_params(labelsize=15, colors=rnfcolor)
#         ax2.xaxis.set_tick_params(labelsize=15)
#         ax1.xaxis.set_tick_params(labelsize=15)
#         ax1.set_ylabel('coverage\n', size=15, color=covcolor)
#         ax1.yaxis.set_tick_params(labelsize=15, colors=covcolor)
#         ax1.set_ylim([0,1000])
#         ax2.set_ylim([0,1])
#         ax2.set_xlim([min(p),max(p)])
#         ax1.spines['bottom'].set_color('lightgrey')
#         ax1.spines['top'].set_color('lightgrey')
#         ax1.spines['left'].set_color('lightgrey')
#         ax1.spines['right'].set_color('lightgrey')
#         ax2.spines['bottom'].set_color('lightgrey')
#         ax2.spines['top'].set_color('lightgrey')
#         ax2.spines['left'].set_color('lightgrey')
#         ax2.spines['right'].set_color('lightgrey')
#         plt.title('Chromosome ' + c + '\n\n', size=20)
#
#         print('')
#         f.tight_layout()
#         plt.show()
#         plt.close(f)


def run_isomut2_on_block(chrom,from_pos,to_pos,
                         input_dir,bam_files,output_dir,
                         ref_genome,
                         min_sample_freq,
                         min_other_ref_freq,
                         cov_limit,
                         base_quality_limit,
                         min_gap_dist_snv,
                         min_gap_dist_indel,
                         bedfile,
                         samtools_flags,
                         ploidy_info_filepath,
                         unique_only,
                         constant_ploidy):
    """
    Run the samtools + the isomut2 C application in the system shell.

    One run is only on a section of a chromosome.
    With bedfile, only the positions of the bedfile are analyzed.
    Automatically logs samtools stderr output to output_dir/samtools.log
    Results are saved to a file which name is created from the block chr,posform,posto
    """

    #build the command
    cmd=' samtools  mpileup ' + samtools_flags
    cmd+=' -f ' +ref_genome
    cmd+=' -r '+chrom+':'+str(from_pos)+'-'+str(to_pos)+' '
    if(bedfile!=None):
        cmd+=' -l '+bedfile+' '
    for bam_file in bam_files:
        cmd+=input_dir+bam_file +' '
    cmd+=' 2>> '+output_dir+'/samtools.log | isomut2 '
    cmd+=' '.join(map(str,[min_sample_freq,min_other_ref_freq,cov_limit,
                           base_quality_limit,min_gap_dist_snv,min_gap_dist_indel, constant_ploidy, ploidy_info_filepath, unique_only]))
    for bam_file in bam_files:
        cmd+=' '+ os.path.basename(bam_file)
    cmd+=' > ' +output_dir+'/tmp_isomut2_'+ chrom+'_'+str(from_pos)+'_'+str(to_pos)+'_mut.csv  '

    return subprocess.check_call(cmd,shell=True)

def run_isomut2_in_parallel(params, level=0):
    """ Run an isomut2 subcommand in parallel on the whole genome."""
    #check for bedfile argument:
    if ('ploidy_info_filepath' not in params):
        params['ploidy_info_filepath']='no_ploidy_info'
    if ('bedfile' not in params):
        params['bedfile']=None
    if ('samtools_flags' not in params):
        params['samtools_flags']=' -B -d '+ str(SAMTOOLS_MAX_DEPTH) + ' '
    if ('chromosomes' not in params):
        params['chromosomes']=None
    if ('unique_mutations_only' not in params):
        params['unique_mutations_only']=True
    if ('constant_ploidy' not in params):
        params['constant_ploidy']=2

    #define blocks and create args
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Preparations for parallelization:\n')
    blocks=define_parallel_blocks(params['ref_fasta'],params['n_min_block'],params['chromosomes'], params=params, level=level+1)
    args=[]
    for block in blocks:
        args.append([ block[0],block[1],block[2],
                     params['input_dir'],params['bam_filenames'],
                     params['output_dir'],params['ref_fasta'],
                     params['min_sample_freq'],params['min_other_ref_freq'],
                     params['cov_limit'], params['base_quality_limit'],
                     params['min_gap_dist_snv'],params['min_gap_dist_indel'],
                     params['bedfile'],params['samtools_flags'], params['ploidy_info_filepath'], int(params['unique_mutations_only']),
                     params['constant_ploidy']])

    #create dir
    if(glob.glob(params['output_dir'])==[]):
        subprocess.call(['mkdir',params['output_dir']])

    print('\t'*level + 'Total number of blocks to run: ' + str(len(args)) + '\n')
    print('\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Currently running:', end = ' ')
    #start first n concurrent block
    procs=[]
    for i in range(params['n_conc_blocks']):
        procs.append(multiprocessing.Process(target=run_isomut2_on_block, args=args[len(procs)]))
        procs[-1].start()
        print(str(len(procs)), end=" ")
        sys.stdout.flush()

    # when one finished restart teh next
    while (len(procs) != len(args)):
        for i in range(1,params['n_conc_blocks']+1):
            #one finished start another one
            if(procs[-i].is_alive() == False and len(procs) != len(args)):
                procs.append(multiprocessing.Process(target=run_isomut2_on_block,args=args[len(procs)] ))
                procs[-1].start()
                print(str(len(procs)), end=" ")
        time.sleep(0.1)

    #wait now only the last ones running
    finished = False
    while( not finished):
        finished = True
        for proc in procs:
            if (proc.is_alive() == True):
                finished = False
        time.sleep(0.1)

    print('\n' + '\t'*level + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Finished this round.\n')

# def run_ploidy_est(params):
#     """
#     Run IsoMutPE on the bam file specified in params dict, with params specified in params dict.
#
#     """
#
#     print 'Current time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
#     print '\n' + '-'*100 + '\n'
#
#     ##########################################
#     # get rough estimation of average haploid coverage
#     print time.strftime("%H:%M:%S", time.gmtime()) + ' - Estimating average haploid coverage throughout the genome based on reference base frequencies...'
#     raw_estimate_of_hapcov(params)
#
#     ##########################################
#     # running ploidy estimation with estimated haploid_cov
#     print time.strftime("%H:%M:%S", time.gmtime()) + ' - Running ploidy estimation with raw estimate of haploid coverage...'
#     run_ploidy_est_in_parallel(params)
#
#     ##########################################
#     # refining estimation if necesarry
#     if (params['rerun_hapcov_estimation']):
#         print time.strftime("%H:%M:%S", time.gmtime()) + ' - Estimating average haploid coverage based on previously identified regions...'
#         new_estimate_of_hapcov(params)
#         print time.strftime("%H:%M:%S", time.gmtime()) + ' - Running ploidy estimation with the new estimate of haploid coverage...'
#         run_ploidy_est_in_parallel(params)
#
#     ##########################################
#     # finalizing results
#     print time.strftime("%H:%M:%S", time.gmtime()) + ' - Finalizing results...'
#     print time.strftime("%H:%M:%S", time.gmtime()) + ' - Cleaning up...'
#     print '\n'
#     print time.strftime("%H:%M:%S", time.gmtime()) + ' - Done.'

def plot_tuning_curve(dataframe, params):

    '''
        Plots tuning curves for all mutations types (SNV, INS, DEL) and all ploidies for each available sample.
    '''

    unique_samples = params['bam_filenames']

    ymax_SNV = dataframe[dataframe['type']=='SNV'].groupby(['sample_name']).count().max()['chr']
    ymax_INS = dataframe[dataframe['type']=='INS'].groupby(['sample_name']).count().max()['chr']
    ymax_DEL = dataframe[dataframe['type']=='DEL'].groupby(['sample_name']).count().max()['chr']
    ymax = [ymax_SNV, ymax_INS, ymax_DEL]
    ymax_all = [10**(len(str(ym))) for ym in ymax]

    mut_types_all = ['SNV', 'INS', 'DEL']

    unique_ploidies = sorted([int(i) for i in list(dataframe['ploidy'].unique())])

    color_dict_base = {'control': '#008B8B',
                  'treated': '#8B008B'}

    color_list = [color_dict_base['control'] if s in params['control_samples']
                  else color_dict_base['treated'] for s in unique_samples]

    fig, axes = plt.subplots(len(unique_ploidies), 3)
    fig.set_size_inches(21, 5*len(unique_ploidies))
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.4, wspace=0.2)
    for m in range(len(mut_types_all)):
        if (len(unique_ploidies) == 1):
            ymax = 10**len(str(dataframe[(dataframe['type']==mut_types_all[m]) & (dataframe['ploidy']==unique_ploidies[0])].groupby(['sample_name']).count().max()['chr']))
            for s, c in zip(unique_samples, color_list):
                l = 'control samples' if s in params['control_samples'] else 'treated samples'
                score = dataframe[(dataframe['type'] == mut_types_all[m]) & (dataframe['sample_name'] == s) & (dataframe['ploidy'] == unique_ploidies[0])].sort_values(by='score')['score']
                axes[m].plot(score, len(score)-np.arange(len(score)), c=c, label=l)
            axes[m].set_xlabel(r'score threshold',fontsize=12)
            axes[m].set_title(mut_types_all[m] + ' (ploidy: ' + str(unique_ploidies[0]) + ')\n',fontsize=14)
            axes[m].set_ylabel(r'Mutations found',fontsize=12)
            axes[m].set_ylim(1,ymax)
            axes[m].set_yscale('log')
            axes[m].set_xlim(0,dataframe['score'].max())
            # axes[i][m].grid()
            handles, labels = axes[m].get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[k] for k in ids]
            axes[m].legend(handles, labels, loc='upper right', fancybox=True)
        else:
            for i in range(len(unique_ploidies)):
                ymax = 10**len(str(dataframe[(dataframe['type']==mut_types_all[m]) & (dataframe['ploidy']==unique_ploidies[i])].groupby(['sample_name']).count().max()['chr']))
                for s, c in zip(unique_samples, color_list):
                    l = 'control samples' if s in params['control_samples'] else 'treated samples'
                    score = dataframe[(dataframe['type'] == mut_types_all[m]) & (dataframe['sample_name'] == s) & (dataframe['ploidy'] == unique_ploidies[i])].sort_values(by='score')['score']
                    axes[i][m].plot(score, len(score)-np.arange(len(score)), c=c, label=l)
                axes[i][m].set_xlabel(r'score threshold',fontsize=12)
                axes[i][m].set_title(mut_types_all[m] + ' (ploidy: ' + str(unique_ploidies[i]) + ')\n',fontsize=14)
                axes[i][m].set_ylabel(r'Mutations found',fontsize=12)
                axes[i][m].set_ylim(1,ymax)
                axes[i][m].set_yscale('log')
                axes[i][m].set_xlim(0,dataframe['score'].max())
                # axes[i][m].grid()
                handles, labels = axes[i][m].get_legend_handles_labels()
                labels, ids = np.unique(labels, return_index=True)
                handles = [handles[k] for k in ids]
                axes[i][m].legend(handles, labels, loc='upper right', fancybox=True)

    figfile = BytesIO()
    plt.savefig(figfile, bbox_inches='tight', format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_roc(dataframe, params, score0=0):

    '''
        Plots ROC curves for all mutations types (SNV, INS, DEL) and all ploidies.
    '''

    steps = 50
    # unique_samples = sorted(list(dataframe['sample_name'].unique()))
    unique_samples = params['bam_filenames']
    total_num_of_FPs_per_genome = params['FPs_per_genome']
    genome_length = params['genome_length']

    unique_ploidies = sorted(list(dataframe['ploidy'].unique()))

    mut_types_all = ['SNV', 'INS', 'DEL']

    score_lim_dict = {'SNV': [], 'INS': [], 'DEL': []}

    control_idx = []
    treated_idx = []
    for i in range(len(unique_samples)):
        if (unique_samples[i] in params['control_samples']):
            control_idx.append(i)
        else:
            treated_idx.append(i)

    control_idx = np.array(control_idx)
    treated_idx = np.array(treated_idx)

    if (total_num_of_FPs_per_genome is not None):
        FPs_per_ploidy = dict()
        for m in mut_types_all:
            FPs_per_ploidy[m] = dict()
            for pl in unique_ploidies:
                totalmuts_per_ploidy = dataframe[(dataframe['ploidy'] == pl) & (dataframe['type'] == m)].shape[0]
                totalmuts = dataframe[(dataframe['type'] == m)].shape[0]
                if (totalmuts == 0):
                    FPs_per_ploidy[m][pl] = total_num_of_FPs_per_genome
                else:
                    FPs_per_ploidy[m][pl] = int(round((float(totalmuts_per_ploidy)/totalmuts)*total_num_of_FPs_per_genome))

    fig, axes = plt.subplots(len(unique_ploidies), 3)
    fig.set_size_inches(21, 5*len(unique_ploidies))
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.4, wspace=0.2)

    for m in range(len(mut_types_all)):
        if (len(unique_ploidies) == 1):
            fp, tp = [0  for k in range(steps)],[0  for k in range(steps)]
            fp_real, tp_real = [0  for k in range(steps)],[0  for k in range(steps)]
            for score_lim,j in zip(np.linspace(score0,dataframe[dataframe['type'] == mut_types_all[m]]['score'].max(),steps),range(steps)):
                muts=[]
                for s in unique_samples:
                    muts.append(dataframe[(dataframe['ploidy'] == unique_ploidies[0]) &
                                          (dataframe['sample_name'] == s) &
                                          (dataframe['score'] > score_lim) &
                                          (dataframe['type'] == mut_types_all[m])].shape[0])
                muts=np.array(muts)
                fp[j] ,tp[j]=1e-6*np.max(muts[control_idx]),1e-6*np.mean(muts[treated_idx])
                fp_real[j] ,tp_real[j]=np.max(muts[control_idx]),np.mean(muts[treated_idx])
            axes[m].step(fp,tp,c='#DB7093',lw=3)
            axes[m].set_title(mut_types_all[m] + ' (ploidy: ' + str(int(unique_ploidies[0])) + ')\n', fontsize=14)

            if (total_num_of_FPs_per_genome is not None):
                fp_real = np.array(fp_real)
                tp_real = np.array(tp_real)
                if (len(tp_real[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]]) > 0):
                    tps = tp_real[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]][0]
                    fps = fp_real[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]][0]
                    score_lim = np.linspace(score0, dataframe[dataframe['type'] == mut_types_all[m]]['score'].max(), steps)[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]][0]
                    axes[m].plot(fps*1e-6,tps*1e-6,'o',mec='#C71585',mfc='#C71585',ms=15,mew=3, label = 'score limit: ' + str(score_lim))
                    axes[m].text(0.95, 0.06, 'score limit: ' + str(score_lim),
                            bbox={'facecolor':'white', 'pad':10}, verticalalignment='bottom', horizontalalignment='right', transform=axes[m].transAxes)
                else:
                    score_lim = 10000
                    axes[m].text(0.95, 0.06, 'score limit: inf',
                            bbox={'facecolor':'white', 'pad':10}, verticalalignment='bottom', horizontalalignment='right', transform=axes[m].transAxes)
                score_lim_dict[mut_types_all[m]].append(score_lim)
            axes[m].set_ylim(ymin=0)
            axes[m].set_xlim(xmin=0)
            axes[m].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            axes[m].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
            axes[m].set_xlabel('false positive rate 1/Mbp ',fontsize=12)
            axes[m].set_ylabel('mutation rate  1/Mbp ',fontsize=12)

        else:
            for i in range(len(unique_ploidies)):
                fp, tp = [0  for k in range(steps)],[0  for k in range(steps)]
                fp_real, tp_real = [0  for k in range(steps)],[0  for k in range(steps)]
                for score_lim,j in zip(np.linspace(score0,dataframe[dataframe['type'] == mut_types_all[m]]['score'].max(),steps),range(steps)):
                    muts=[]
                    for s in unique_samples:
                        muts.append(dataframe[(dataframe['ploidy'] == unique_ploidies[i]) &
                                              (dataframe['sample_name'] == s) &
                                              (dataframe['score'] > score_lim) &
                                              (dataframe['type'] == mut_types_all[m])].shape[0])
                    muts=np.array(muts)
                    fp[j] ,tp[j]=1e-6*np.max(muts[control_idx]),1e-6*np.mean(muts[treated_idx])
                    fp_real[j] ,tp_real[j]=np.max(muts[control_idx]),np.mean(muts[treated_idx])
                axes[i][m].step(fp,tp,c='#DB7093',lw=3)
                axes[i][m].set_title(mut_types_all[m] + ' (ploidy: ' + str(int(unique_ploidies[i])) + ')\n', fontsize=14)

                if (total_num_of_FPs_per_genome is not None):
                    fp_real = np.array(fp_real)
                    tp_real = np.array(tp_real)
                    if (len(tp_real[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]]) > 0):
                        tps = tp_real[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        fps = fp_real[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        score_lim = np.linspace(score0, dataframe[dataframe['type'] == mut_types_all[m]]['score'].max(), steps)[fp_real<=FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        axes[i][m].plot(fps*1e-6,tps*1e-6,'o',mec='#C71585',mfc='#C71585',ms=15,mew=3, label = 'score limit: ' + str(score_lim))
                        axes[i][m].text(0.95, 0.06, 'score limit: ' + str(score_lim),
                                bbox={'facecolor':'white', 'pad':10}, verticalalignment='bottom', horizontalalignment='right', transform=axes[i][m].transAxes)
                    else:
                        score_lim = 10000
                        axes[i][m].text(0.95, 0.06, 'score limit: inf',
                                bbox={'facecolor':'white', 'pad':10}, verticalalignment='bottom', horizontalalignment='right', transform=axes[i][m].transAxes)
                    score_lim_dict[mut_types_all[m]].append(score_lim)
                axes[i][m].set_ylim(ymin=0)
                axes[i][m].set_xlim(xmin=0)
                axes[i][m].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
                axes[i][m].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
                axes[i][m].set_xlabel('false positive rate 1/Mbp ',fontsize=12)
                axes[i][m].set_ylabel('mutation rate  1/Mbp ',fontsize=12)
            # axes[i][m].grid()

    figfile = BytesIO()
    plt.savefig(figfile, bbox_inches='tight', format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    return figdata_png, score_lim_dict


def plot_filtered_muts(dataframe, score_lims_dict, params):

    '''
        Plots the number of unique mutations found in all the samples in different ploidy regions.
    '''

    unique_samples = sorted(params['bam_filenames'])

    list_of_filtered_dataframes = []
    samples = unique_samples
    pos = list(range(len(samples)))
    unique_ploidies = sorted(list(dataframe['ploidy'].unique()))
    width = 1./(len(unique_ploidies)+1)

    color_dict_base = {'control': '#008B8B',
                  'treated': '#8B008B'}

    color_list = [color_dict_base['control'] if s in params['control_samples']
                  else color_dict_base['treated'] for s in samples]

    mut_types_all = ['SNV', 'INS', 'DEL']

    # Plotting the bars
    fig, axes = plt.subplots(len(mut_types_all), 1)
    fig.set_size_inches(14, 6*len(mut_types_all))
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.4, wspace=0.2)


    for m in range(len(mut_types_all)):
        for i in range(len(unique_ploidies)):
            filtered_table = dataframe[(dataframe['type'] == mut_types_all[m]) & (dataframe['score'] > score_lims_dict[mut_types_all[m]][i]) &
                                       (dataframe['ploidy'] == unique_ploidies[i])]
            list_of_filtered_dataframes.append(filtered_table)
            sample_counts=filtered_table[filtered_table['ploidy']==unique_ploidies[i]].groupby(['sample_name']).count().reset_index()[['sample_name','cov']]
            sample_counts.columns=['sample','count']

            #add zeroes if a sample is missing
            for s in samples:
                if (s not in set(sample_counts['sample'])):
                    sample_counts=pd.concat([sample_counts,pd.DataFrame({'sample':s, 'count' : [0]})])
                    sample_counts=sample_counts.sort_values(by='sample').reset_index()[['sample','count']]

            sample_counts.sort_values(by='sample')

            if (len(unique_ploidies) == 1):
                a = 1
            else:
                a = 1-(i+1)*(1./len(unique_ploidies))
            barlist = axes[m].bar([p + i*width for p in pos], sample_counts['count'], width, alpha=a, color="violet", label = str(int(unique_ploidies[i])))
            for j in range(len(barlist)):
                barlist[j].set_color(color_list[j])

        # Set the y axis label
        axes[m].set_ylabel('Unique mutation count', fontsize=12)

        # Set the chart's title
        if (len(unique_ploidies) == 1):
            axes[m].set_title('\nFiltered unique ' + mut_types_all[m] + ' counts\n', fontsize=14)
        else:
            axes[m].set_title('\nFiltered unique ' + mut_types_all[m] + ' counts grouped by ploidy\n', fontsize=14)

        # Set the position of the x ticks
        axes[m].set_xticks([p -0.5 + len(unique_ploidies) * 1 * width for p in pos])
        # Set the labels for the x ticks
        sample_labels = ['\n'.join([s[i:i+10] for i in range(0, len(s), 10)]) for s in samples]
        fs = 8
        axes[m].set_xticklabels(sample_labels, rotation=90, fontsize=fs)


        # Setting the x-axis and y-axis limits
        axes[m].set_xlim(min([p -1 + len(unique_ploidies) * 0.5 * width for p in pos]),
                         max(pos)+width*(len(unique_ploidies)+1))
        axes[m].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        axes[m].set_yscale('log')
        # axes[m].grid()

    figfile = BytesIO()
    plt.savefig(figfile, bbox_inches='tight', format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    return figdata_png, pd.concat(list_of_filtered_dataframes)

def get_filtered_results(dataframe, score_lims_dict):

    '''
        Filters the dataframe based on score values in the score_lims_dict.
    '''

    list_of_filtered_dataframes = []

    unique_ploidies = sorted(list(dataframe['ploidy'].unique()))
    mut_types_all = ['SNV', 'INS', 'DEL']

    for m in range(len(mut_types_all)):
        for i in range(len(unique_ploidies)):
            filtered_table = dataframe[(dataframe['type'] == mut_types_all[m]) &
                                       (dataframe['score'] > score_lims_dict[mut_types_all[m]][i]) &
                                       (dataframe['ploidy'] == unique_ploidies[i])]
            list_of_filtered_dataframes.append(filtered_table)

    return pd.concat(list_of_filtered_dataframes)


def plot_heatmap(dataframe, params):

    '''
        Generates heatmap of the number of filtered mutations found in all possible sample pairs.
        A dendrogram is also added that is the result of hierarchical clustering of the samples.
    '''

    sample_names = params['bam_filenames']

    c = np.zeros((len(sample_names), len(sample_names)))
    for i in range(len(sample_names)):
        for j in range(i+1):
            c[i][j] = dataframe[(dataframe["sample_name"].str.contains(sample_names[i])) & (dataframe["sample_name"].str.contains(sample_names[j]))].shape[0]
            c[j][i] = c[i][j]
        # c[i][i] = dataframe[dataframe['sample_name'] == sample_names[i]].shape[0]

    d = pd.DataFrame(c)
    d.columns = sample_names
    d.index = sample_names

    g = sns.clustermap(d, method='average', cmap="viridis", robust=True);
    plt.close()

    figfile = BytesIO()
    g.savefig(figfile, bbox_inches='tight', format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    return figdata_png

def print_filtered_results(filtered_table, params):

    '''
        Prints filtered mutation results to the original IsoMut2 output directory with the name filtered_results.csv.
    '''

    IsoMut2_results_dir = params['output_dir'] + '/'
    control_samples = params['control_samples']
    total_num_of_FPs_per_genome = params['FPs_per_genome']

    with open(IsoMut2_results_dir + 'filtered_results.csv', 'a') as f:
        f.write('# IsoMut2 filtered results - ' + str(datetime.now()).split('.')[0] + '\n')
        f.write('# Original results:\n')
        f.write('#\t'+ IsoMut2_results_dir + 'all_SNVs.isomut2\n')
        f.write('#\t'+ IsoMut2_results_dir + 'all_indels.isomut2\n')
        f.write('# Control samples:\n')
        for s in control_samples:
            f.write('#\t'+ s + '\n')
        f.write('# Total allowed number of false positives per genome: ' + str(total_num_of_FPs_per_genome) + '\n')
        f.write('#\n')
        f.write('#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n')
        filtered_table.to_csv(f, sep='\t', index=False, header=False)



def generate_HTML_report(params, score0=0):

    '''
        - Filters results so that a minimal number of unique mutations are left in control samples, while keeping the number of unique mutations
        as high as possible in other samples.
        - Generates HTML report of the filtering process and the filtered results.
        - Saves filtered results to original IsoMut2 output directory.
    '''

    # first we generate two temporary files with only unique mutations, so that the score optimisation does not need to be run on such huge data files

    subprocess.check_call('cat '+params['output_dir']+'/all_SNVs.isomut2 | awk \'BEGIN{FS="\t"; OFS="\t";}{if($1 !~ /,/) print $0;}\' > ' + params['output_dir'] + '/unique_SNVs.isomut2',shell=True)
    subprocess.check_call('cat '+params['output_dir']+'/all_indels.isomut2 | awk \'BEGIN{FS="\t"; OFS="\t";}{if($1 !~ /,/) print $0;}\' > ' + params['output_dir'] + '/unique_indels.isomut2',shell=True)

    IsoMut2_results_dir = params['output_dir'] + '/'
    control_samples = params['control_samples']
    sample_names = params['bam_filenames']
    genome_length = params['genome_length']
    total_num_of_FPs_per_genome = params['FPs_per_genome']

    df_SNV = pd.read_csv(IsoMut2_results_dir + 'unique_SNVs.isomut2',
                     names = ['sample_name', 'chr', 'pos', 'type', 'score',
                            'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                     sep='\t',
                     low_memory=False)
    df_indel = pd.read_csv(IsoMut2_results_dir + 'unique_indels.isomut2', header=0,
                         names = ['sample_name', 'chr', 'pos', 'type', 'score',
                                'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                         sep='\t',
                         low_memory=False)
    df = pd.concat([df_SNV, df_indel])

    # df_somatic = df[~(df['sample_name'].str.contains(','))]
    df_somatic = df

    unique_ploidies = sorted(list(df_somatic['ploidy'].unique()))

    # all image codes:
    FIG_tuning_curve = plot_tuning_curve(df_somatic, params=params)
    FIG_ROC, score_lim_dict = plot_roc(dataframe=df_somatic, params=params, score0=score0)
    FIG_filtered_muts, df_somatic_filt = plot_filtered_muts(dataframe=df_somatic, score_lims_dict=score_lim_dict,
                                                           params=params)

    # generate files with the filtered results
    with open(params['output_dir'] + '/filtered_results.csv', 'a') as f:
        f.write('# IsoMut2 filtered results - ' + str(datetime.now()).split('.')[0] + '\n')
        f.write('# Original results:\n')
        f.write('#\t'+ IsoMut2_results_dir + 'all_SNVs.isomut2\n')
        f.write('#\t'+ IsoMut2_results_dir + 'all_indels.isomut2\n')
        f.write('# Control samples:\n')
        for s in control_samples:
            f.write('#\t'+ s + '\n')
        f.write('# Total allowed number of false positives per genome: ' + str(total_num_of_FPs_per_genome) + '\n')
        f.write('#\n')
        f.write('#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n')

    # print(score_lim_dict)
    filter_cmd = 'cat ' + IsoMut2_results_dir + 'all_SNVs.isomut2 | awk \'BEGIN{FS=\"\t\"; OFS=\"\t\";}{'
    for i in range(len(unique_ploidies)):
        filter_cmd += 'if ($11 == ' + str(unique_ploidies[i]) + ' && $5 > ' + str(score_lim_dict['SNV'][i]) + ') print $0; '
    filter_cmd += '}\' >> ' + IsoMut2_results_dir + 'filtered_results.csv'

    subprocess.check_call(filter_cmd,shell=True)
    subprocess.check_call('rm ' + IsoMut2_results_dir + 'unique_SNVs.isomut2',shell=True)

    filter_cmd = 'cat ' + IsoMut2_results_dir + 'all_indels.isomut2 | awk \'BEGIN{FS=\"\t\"; OFS=\"\t\";}{'
    for i in range(len(unique_ploidies)):
        filter_cmd += 'if ($4 == "INS" && $11 == ' + str(unique_ploidies[i]) + ' && $5 > ' + str(score_lim_dict['INS'][i]) + ') print $0; '
        filter_cmd += 'if ($4 == "DEL" && $11 == ' + str(unique_ploidies[i]) + ' && $5 > ' + str(score_lim_dict['DEL'][i]) + ') print $0; '
    filter_cmd += '}\' >> ' + IsoMut2_results_dir + 'filtered_results.csv'

    subprocess.check_call(filter_cmd,shell=True)
    subprocess.check_call('rm ' + IsoMut2_results_dir + 'unique_indels.isomut2',shell=True)

    # read filtered Results

    df = pd.read_csv(IsoMut2_results_dir + 'filtered_results.csv', comment = '#',
                         names = ['sample_name', 'chr', 'pos', 'type', 'score',
                                'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                         sep='\t',
                         low_memory=False)
    df_somatic = df[~(df['sample_name'].str.contains(','))]

    # filtered results for the whole datatable
    # df_filtered = get_filtered_results(dataframe=df, score_lims_dict=score_lim_dict)

    # if non-unique mutations are also detected
    if (df_somatic.shape[0] != df.shape[0]):
        FIG_heatmap = plot_heatmap(dataframe=df, params=params)

    # printing filtered results
    # print_filtered_results(filtered_table=df_filtered, params=params)

    # generating HTML report

    html_string = '''
    <html>
    <style>
        @import url(https://fonts.googleapis.com/css?family=Lora:400,700,400italic,700italic);
        @import url(https://fonts.googleapis.com/css?family=Open+Sans:800)
        @import url(http://fonts.googleapis.com/css?family=Lato|Source+Code+Pro|Montserrat:400,700);
        @import url(https://fonts.googleapis.com/css?family=Raleway);
        @import "font-awesome-sprockets";
        @import "font-awesome";

        body {
            font-family: 'Lora', 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 145%;}

        p {
          text-align: justify;}

        h1,h2,h3,h4,h5,h6 {
          font-family: 'Open Sans', sans-serif;
          font-weight: 800;
          line-height: 145%;}

        h1 {
          font-size: 4rem;}
        h2 {
          font-size: 3.5rem;}

        .MathJax{
            font-size: 7pt;}

        img {
            text-align:center;
            display:block;}

        </style>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <script>
            MathJax.Hub.Config({
            tex2jax: {inlineMath: [['$','$']]}
            });
        </script>

        <head>
            <meta charset="utf-8">
        </head>
        <body>
            <h2>IsoMut2 results - $S$ score optimisation</h2>
            <br>
            Date and time of analysis: ''' + str(datetime.now()).split('.')[0] + ''' <br><br>
            Original IsoMut2 results used: <br>
            ''' + IsoMut2_results_dir + '''all_SNVs.isomut2 <br>
            ''' + IsoMut2_results_dir + '''all_indels.isomut2 <br><br>
            Control samples: ''' + ', '.join(params['control_samples']) + ''' <br><br>
            Maximum number of false positives per genome: ''' + str(total_num_of_FPs_per_genome) + ''' <br><br>
            Filtered results are saved to: ''' + IsoMut2_results_dir + '''filtered_results.csv<br><br>
            <h3>Tuning curves:</h3>
            The figures below show the number of <i>unique</i> mutations found in each sample grouped by mutation types and ploidies.
            Control samples are represented with differently colored curves for easier identification. The expected result is
            that control samples tend to have steeper curves than other samples, that reach zero at lower score thresholds. If
            that is not the case, consider the following possibilities: a) control samples have been chosen incorrectly, b)
            there is reason to expect mutations in the control samples, c) the effect of mutations is negligible in
            (at least some of the) treated samples.
            <br><br>
            <img src="data:image/png;base64,''' + FIG_tuning_curve.decode('utf-8')+ '''" alt="tuning_curves.png"><br>
            <h3>ROC curves:</h3>
            The figures below show ROC curves for the number of false positive and true positive mutations grouped by mutation
            types and ploidies. The score threshold is varied along the curves. False positives are defined as the maximum
            number of <i>unique</i> mutations found in any of the control samples with the given score threshold (for the given mutation
            type and ploidy). The average number of <i>unique</i> mutations in the treated samples for the given score threshold
            is defined as the number of true positives. (Strictly speaking, this is an inaccurate definition, but it
            allows for rapid optimisation.)
            <br>
            The large dot at a specific point of the curve represents the score threshold
            that fits the user defined value of the maximum number of false positives per genome. As false positives
            are expected to occur randomly, when regions of multiple
            ploidies are present in the investigated genomes, the allowed false positives are divided between these based on
            the ratio of the original mutations (without score filter) present in the given ploidy region.
            (For example, if most of the genome is
            diploid, it follows that most of the original mutations arise from diploid regions as well, thus most of the
            false positives should be in diploid regions also.)
            Thus $FP_{p} = \lfloor \\frac{M_p}{M} \cdot FP \\rfloor$,
            where $p$ is the ploidy, $FP_p$ is the integer number of false positives in regions with ploidy $p$, $M_p$ is the
            number of non-filtered <i>unique</i> mutations in regions with ploidy $p$, $M$ is the total number of non-filtered
            <i>unique</i> mutations
            and $FP$ is the
            user-defined maximum number of false positives per genome. The following hold true:
            $M = \sum_p{M_p}$ and $FP \geq \sum_p{FP_p}$.
            <br><br>
            <img src="data:image/png;base64,''' + FIG_ROC.decode('utf-8') + '''" alt="ROC_curves.png"><br>
            <h3>Filtered results:</h3>
            The figures below show the number of filtered <i>unique</i> mutations found in the samples, grouped by mutation type.
            Bars with different hues for a single sample represent mutations arising from regions with different ploidies.
            Control samples are marked with different color for straightforward interpretation.
            The expected result is small bars for control samples and higher bars for the others. If this is not the case,
            check the tuning curves for clues.
            <br>
            <img src="data:image/png;base64,''' + FIG_filtered_muts.decode('utf-8') + '''" alt="filtered_muts.png"><br>'''

    if (df_somatic.shape[0] != df.shape[0]):
        html_string += '''
            <h3>Number of mutations found in all samples:</h3>
            The heatmap below shows the number of filtered mutations found in all sample pairs, grouped by mutation type.
            The dendrogram represents the result of hierarchical clustering carried out
            on the filtered mutation counts. Samples are reordered so that "more similar" samples tend to cluster
            together. Diagonal elements represent the total number of mutations found in a given sample.
            <br><br>
            <img src="data:image/png;base64,''' + FIG_heatmap.decode('utf-8') + '''" alt="heatmap.png"><br>'''

    html_string += '''
            </body>
        </html>'''

    with open(IsoMut2_results_dir+'report.html','w') as f:
        f.write(html_string)


def run_isomut2(params):
    """
    Run IsoMut2 on the bam files specified in params dict, with params specified in params dict.

    """

    starting_time = datetime.now()

    if (params['HTML_report']):
        if(('FPs_per_genome' not in params) or ('control_samples' not in params)):
            params['HTML_report'] = False
            print('ERROR: Parameter dictionary does not contain parameters "FPs_per_genome" and/or "control_samples". HTML report will not be created.\n\n')

    ##########################################
    #run first round
    print('Running IsoMut2: round 1/2\n\n')
    run_isomut2_in_parallel(params, level=1)

    ##########################################
    #prepare for 2nd round
    print('Running IsoMut2: round 2/2\n\n')
    print('\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Preparing files for post-processing...')

    # collect indels
    header="#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n"
    with open(params['output_dir']+'/all_indels.isomut2','w') as indel_f  :
        #write header
        indel_f.write(header)
    #copy all indel lines except the header and sort them by chr, pos
    subprocess.check_call(
        'tail -q -n+2 ' +params['output_dir']+'/tmp_isomut2_*_mut.csv | \
        awk \'$4=="INS" || $4=="DEL" {print}\' | \
        sort -n -k2,2 -k3,3 >> '+params['output_dir']+'/all_indels.isomut2',shell=True)

    # create bedfile for SNVs for post processing
    subprocess.check_call(
        'tail -q -n+2 ' +params['output_dir']+'/tmp_isomut2_*_mut.csv |\
        awk \'$4=="SNV" {print}\' | \
        sort -n -k2,2 -k3,3 | cut -f 2,3 > ' +params['output_dir']+'/tmp_isomut2.bed',shell=True)

    # saving original results for later
    for init_res in glob.glob(params['output_dir']+'/tmp_isomut2_*_mut.csv'):
        new_file_name = params['output_dir'] + '/' + init_res.split('.')[-2].split('/')[-1] + '_orig.csv'
        subprocess.check_call('mv ' + init_res + ' ' + new_file_name, shell=True)

    ##########################################
    #2nd round

    # change params for postprocessing
    params['base_quality_limit']= 13
    params['min_other_ref_freq']= 0
    params['samtools_flags'] = ' -d '+ str(SAMTOOLS_MAX_DEPTH)  + ' '
    params['bedfile']=params['output_dir']+'/tmp_isomut2.bed'

    #run it
    run_isomut2_in_parallel(params, level=1)

    ##########################################
    #finalize output
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Finalizing output...')

    # if the second round did not yield any mutations, keep the original ones, but issue warning
    number_of_files = subprocess.check_output('ls '+params['output_dir']+'/tmp_isomut2_*_mut.csv | wc -l', shell=True)
    number_of_files = int(number_of_files.strip())

    number_of_lines = subprocess.check_output('cat '+params['output_dir']+'/tmp_isomut2_*_mut.csv | wc -l', shell=True)
    number_of_lines = int(number_of_lines.strip())

    if (number_of_files == number_of_lines):
        print('\tWARNING: Second round did not result in any mutations, using original results. (It is advised you take a closer look at the BAM files.)')
        subprocess.check_call(
            'tail -q -n+2 '+params['output_dir']+'/tmp_isomut2_*_mut_orig.csv | \
            awk \'$4=="SNV" {print}\' | \
            sort -n -k2,2 -k3,3 >> '+params['output_dir']+'/all_SNVs.isomut2',shell=True)

    else:
        for new_file_name in glob.glob(params['output_dir']+'/tmp_isomut2_*_mut.csv'):
            with open(new_file_name) as f_new:
                # let's collect original cleanliness into a dict first
                cleanliness_dict=dict()
                old_file_name = params['output_dir'] + '/' + new_file_name.split('.')[-2].split('/')[-1] + '_orig.csv'
                with open(old_file_name) as f_old:
                    f_old.readline() #skip header
                    for line in f_old:
                        line_list=line.strip().split('\t')
                        if (len(line_list) == 11):
                            key='_'.join(line_list[1:4]+line_list[5:7])
                            cleanliness_dict[key]=[line_list[9], line_list[0]]
                # now it's okay to remove the old file
                subprocess.check_call('rm ' + old_file_name, shell=True)
                # now we move on to the new files
                f_new.readline() #skip header
                final_file_name = params['output_dir'] + '/' + new_file_name.split('.')[-2].split('/')[-1] + '_final.csv'
                with open(final_file_name,'w') as f_final:
                    f_final.write(header)
                    for line in f_new:
                        line_list=line.strip().split('\t')
                        key='_'.join(line_list[1:4]+line_list[5:7])
                        if key in cleanliness_dict:
                            f_final.write('\t'.join([cleanliness_dict[key][1]]+line_list[1:9]+[cleanliness_dict[key][0]]+[line_list[10]])+'\n')

        # now we collect SNVs
        subprocess.check_call(
            'tail -q -n+2 '+params['output_dir']+'/tmp_isomut2_*_mut_final.csv | \
            awk \'$4=="SNV" {print}\' | \
            sort -n -k2,2 -k3,3 >> '+params['output_dir']+'/all_SNVs.isomut2',shell=True)

    ##########################################
    #clean up
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Cleaning up temporary files...')

    #clean up
    subprocess.check_call(['rm',params['bedfile']])
    # and now we can also remove the new files and the final files too
    subprocess.check_call('rm '+params['output_dir']+'/tmp_isomut2_*_mut.csv',shell=True)
    subprocess.check_call('rm '+params['output_dir']+'/tmp_isomut2_*_mut_final.csv',shell=True)
    subprocess.check_call('rm '+params['output_dir']+'/tmp_isomut2_*_mut_orig.csv',shell=True)
    # subprocess.check_call('rm '+params['output_dir']+'/tmp_all_SNVs.isomut',shell=True)

    #HTML report
    if (params['HTML_report']):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - Generating HTML report...')
        generate_HTML_report(params, score0=0)

    finish_time = datetime.now()
    total_time = finish_time-starting_time
    total_time_h = int(total_time.seconds/3600)
    total_time_m = int((total_time.seconds%3600)/60)
    total_time_s = (total_time.seconds%3600)%60
    print('\n' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +' - IsoMut2 finished. (' + str(total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), ' + str(total_time_m) + ' min(s), ' + str(total_time_s) + ' sec(s).)')
