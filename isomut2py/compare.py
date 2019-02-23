try:
    from isomut2py import ploidyestimation

    import pandas as __pd
    import sqlite3 as __sqlite3
    import os as __os
    import subprocess as __subprocess
    from scipy import stats as __stats
    import numpy as __np
    import time as __time
    import sys as __sys
except ImportError:
    print('ImportError in isomut2py.compare, comparison of sample ploidies will not work.')


def compare_with_bed(bed_dataframe, other_file, minLen):
    """

    Compares the results of ploidy estimation with a bed file defined in other_file.

    :param bed_dataframe: a pandas.DataFrame of the bedfile of the sample (pandas.DataFrame)
    :param other_file: The path to the bedfile of the other sample. (str)
    :param minLen: The minimum length of a region to be considered different from the other_file. (int)

    :returns: df_joined: A pandas.DataFrame containing region information from both the PloidyEstimation object and the other_file.

    """
    cols = ['chrom', 'chromStart', 'chromEnd', 'ploidy', 'LOH']

    df1 = bed_dataframe

    if other_file.__class__ == str:
        if not __os.path.isfile(other_file):
            raise ValueError('Error: file ' + other_file + ' does not exist, bedfile comparison failed.')
        df2 = __pd.read_csv(other_file, header=0, names=cols)
    elif other_file.__class__ == __pd.core.frame.DataFrame:
        df2 = other_file
    else:
        raise TypeError('Error: Type of "other_file" must be either str or pandas DataFrame.')

    for c in cols:
        if c != 'chrom':
            df1[c] = __pd.to_numeric(df1[c])
            df2[c] = __pd.to_numeric(df2[c])
    conn = __sqlite3.connect(':memory:')
    df1.to_sql('df1', conn, index=False)
    df2.to_sql('df2', conn, index=False)
    query = '''SELECT df1.chrom as chrom,
            MAX(df1.chromStart, df2.chromStart) as intStart, MIN(df1.chromEnd, df2.chromEnd) as intEnd,
            df1.ploidy as ploidy1, df2.ploidy as ploidy2, df1.LOH as LOH1, df2.LOH as LOH2
            FROM df1,df2
            WHERE ((df1.chromStart between df2.chromStart and df2.chromEnd)
            OR (df2.chromStart between df1.chromStart and df1.chromEnd))
            AND (df1.chrom == df2.chrom)
            AND (df1.ploidy != df2.ploidy OR df1.LOH != df2.LOH)'''

    df_joined = __pd.read_sql_query(query, conn)
    df_joined['intLength'] = df_joined['intEnd'] - df_joined['intStart']
    df_joined = df_joined[df_joined['intLength'] >= minLen]
    df_joined.sort_values(by='intLength', ascending=False, inplace=True)
    return df_joined


def check_interval_for_difference(chrom, chromStart, chromEnd, ploidy1, ploidy2,
                                  original_bamfile1, original_bamfile2, refgenome,
                                  prior_dist_dict1, prior_dist_dict2):
    """

    Checks if a genomic interval with different estimated ploidies is really different in the two samples.

    :param chrom: The chromosome of the genomic interval. (str)
    :param chromStart: The starting position of the genomic interval. (int)
    :param chromEnd: The ending position of the genomic interval. (int)
    :param ploidy1: The ploidy estimated for the genomic interval for original_bamfile1. (int)
    :param ploidy2: The ploidy estimated for the genomic interval for original_bamfile2. (int)
    :param original_bamfile1: The path to the original bamfile containing alignment information for sample1. (str)
    :param original_bamfile2: The path to the original bamfile containing alignment information for sample2. (str)
    :param refgenome: The path to the reference genome fasta file. (str)
    :param prior_dist_dict1: The parameters of the seven Gaussians fitted to the coverage distribution of sample1. (dict)
    :param prior_dist_dict2: The parameters of the seven Gaussians fitted to the coverage distribution of sample2. (dict)

    :returns: the ratio of the likelihood of the two samples having different ploidies in the region and the likelihood of them having the same ploidies (float)
        - if the ratio of them having different ploidies is smaller than the ratio of them having the same one, the returned value is 0

    """

    # posteriors are calculated using the prior distribution and AVERAGE coverages in the given region
    # thus the bam files have to be created, but the average values can be calculated on the fly

    cmd = "samtools mpileup -f " + refgenome + " -r " + str(chrom) + ":" + str(chromStart) + "-" + str(
        chromEnd) + " "
    cmd += original_bamfile1 + " " + original_bamfile2
    cmd += " | head | awk 'BEGIN{s1=0; s2=0; n=0;} {s1=s1+$4; s2=s2+$7; n=n+1;} END{print s1, s2, n;}'"

    s = __subprocess.check_output(cmd, shell=True)
    l = [int(k) for k in s.decode("utf-8").strip().split(' ')]

    if l[2] == 0:
        return 0

    avg_cov = [0, 0]
    avg_cov[0] = l[0] / l[2]
    avg_cov[1] = l[1] / l[2]

    pl = [ploidy1, ploidy2]
    prior = [prior_dist_dict1, prior_dist_dict2]

    post = []

    for i in range(2):
        for j in range(2):
            likelihood = __stats.norm.pdf(avg_cov[i], loc=prior[i]['mu'][pl[j] - 1],
                                               scale=prior[i]['sigma'][pl[j] - 1])
            post.append(likelihood * prior[i]['p'][pl[j] - 1])

    prob_of_same_ploidy = __np.max([post[0] * post[2], post[1] * post[3]])
    prob_of_diff_ploidy = post[0] * post[3]

    if prob_of_same_ploidy > 0:
        return prob_of_diff_ploidy / prob_of_same_ploidy
    else:
        return 0

def compare_with_other_PloidyEstimator(ob1, ob2, minLen, minQual):
    """

    Compare the estimated ploidies of the PloidyEstimation object with the ploidies of another PloidyEstimator object.

    :param ob2: the other PloidyEstimator object (isomut2py.ploidyestimation.PloidyEstimator)
    :param ob1: the first PloidyEstimator object (isomut2py.ploidyestimation.PloidyEstimator)
    :param minLen: The minimum length of a region to be considered different from the other object. (int)
    :param minQual: The minimum quality of a region to be considered different from the other object. (float)

    :returns: df_intervals: The differing intervals meeting the above criteria. (pandas.DataFrame)

    """

    # check if they really are ploidyEstimator objects
    if ob1.__class__ != ploidyestimation.PloidyEstimator or ob2.__class__ != ploidyestimation.PloidyEstimator:
        raise ValueError('Error: both "ob1" and "ob2" must be PloidyEstimator objects.')

    # loading coverage distribution estimates
    if not hasattr(ob1, 'distribution_dict'):
        ob1.load_cov_distribution_parameters_from_file()
    if not hasattr(ob1, 'bed_dataframe'):
        ob1.load_bedfile_from_file()
    if not hasattr(ob2, 'distribution_dict'):
        ob2.load_cov_distribution_parameters_from_file()
    if not hasattr(ob2, 'bed_dataframe'):
        ob2.load_bedfile_from_file()

    # compare bed files
    df_intervals = compare_with_bed(bed_dataframe=ob1.bed_dataframe,
                                      other_file=ob2.bed_dataframe,
                                      minLen=minLen)

    level = 0
    qualities = []

    print('\t' * (level) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - Comparing regions of different ploidies in detail\n')
    print('\t' * (level + 1) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                             __time.localtime()) + ' - Number of intervals to check: ' + str(
        df_intervals.shape[0]) + '\n')
    print('\t' * (level + 2) + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Currently running:', end=' ')
    i = 0
    for rIDX, r in df_intervals.iterrows():
        quality = check_interval_for_difference(r['chrom'], r['intStart'], r['intEnd'], r['ploidy1'],
                                                     r['ploidy2'],
                                                     original_bamfile1=ob1.input_dir + ob1.bam_filename,
                                                     original_bamfile2=ob2.input_dir + ob2.bam_filename,
                                                     refgenome=ob1.ref_fasta,
                                                     prior_dist_dict1=ob1.distribution_dict,
                                                     prior_dist_dict2=ob2.distribution_dict)
        qualities.append(quality)
        i += 1
        print(str(i), end=" ")
        __sys.stdout.flush()

    df_intervals['quality'] = qualities
    df_intervals = df_intervals[df_intervals['quality'] >= minQual]
    df_intervals.sort_values(by=['chrom', 'intStart'], inplace=True)

    df_intervals.to_csv(
        ob1.output_dir + '/' +  ob1.bam_filename.split('.')[0] + '_VS_' + ob2.bam_filename.split('.')[0] + '_COMP.bed',
        index=False, sep='\t')

    print('\n')
    print('\t' * (level + 1) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                             __time.localtime()) + ' - Differing genomic intervals with quality scores saved to file ' + ob1.output_dir +
          ob1.bam_filename.split('.')[0] + '_VS_' + ob2.bam_filename.split('.')[0] + '_COMP.bed')
    print('\n')
    print('\t' * (level) + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Comparison finished.')

    return df_intervals