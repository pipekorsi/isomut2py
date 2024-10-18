try:
    import numpy as __np
    import os as __os
    import pandas as __pd
    import pickle as __pickle
    import subprocess as __subprocess
    from datetime import datetime as __datetime
except ImportError:
    print('ImportError in isomut2py.ploidyEstLoad, loading functions will not work.')


def get_coverage_distribution(chromosomes, output_dir, cov_max, cov_min):
    """

    Sets the coverage_sample attribute of the PloidyEstimation object to the coverage distribution obtained from the temporary files created by
    __PE_prepare_temp_files(). Positions are filtered according to the attributes of the PloidyEstimation object. The number of positions in the
    final sample is decreased to 2000 for faster inference.

    :param chromosomes: list of chromosomes in the samples (array-like)
    :param output_dir: path to the directory where temporary files (PEtmp_fullchrom*) are located (str)
    :param cov_max: the maximum value of the coverage for a position to be included in the analysis (int)
    :param cov_min: the minimum value of the coverage for a position to be included in the analysis (int)

    :returns: a 2000-element sample from the coverage distribution

    """

    covs = __np.array([])
    for c in chromosomes:
        if __os.path.isfile(output_dir + '/' + 'PEtmp_fullchrom_' + c + '.txt'):
            fname = output_dir + '/' + 'PEtmp_fullchrom_' + c + '.txt'
            df = __pd.read_csv(fname, sep='\t', names=['chrom', 'pos', 'cov', 'rnf', 'CG'])
        elif __os.path.isfile(output_dir + '/' + 'PE_fullchrom_' + c + '.txt'):
            fname = output_dir + '/' + 'PE_fullchrom_' + c + '.txt'
            df = __pd.read_csv(fname, sep='\t', header=0)
        else:
            error_msg = 'No files were available to calculate coverage distribution.\n'
            error_msg += 'Try running "run_ploidy_estimation()" first.'
            raise ValueError(error_msg)
        covs = __np.append(covs, __np.array(list(df['cov'])))
    covs = covs[(covs < cov_max) * (covs > cov_min)]
    covs_few = __np.random.choice(covs, 2000)
    return covs_few


def save_obj(obj, name):
    """

    Saves a python object (obj) with the specified name as (name).pkl.

    :param obj: object to save
    :param name: filename (without extension)

    """
    with open(name + '.pkl', 'wb') as f:
        __pickle.dump(obj, f, __pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads a python object from a file with the name (name).pkl.

    :param name: filename (without extension)
    :return: loaded object
    """
    with open(name + '.pkl', 'rb') as f:
        return __pickle.load(f)


def load_cov_distribution_parameters_from_file(filename=None, output_dir=None):
    """

    Loads the parameters of the seven fitted Gaussians to the coverage distribution of the sample from the specified filename (that was saved with
    pickle beforehand). If one such file is available, the computationally expensive ploidy estimation process can be skipped.

    :param filename: The path to the file with the coverage distribution parameters. (default: None) (str)
    :param output_dir: if filename is not supplied, distribution parameters are attempted to be loaded from [output_dir]/GaussDistParams.pkl (default: None) (str)

    :returns: dictionary containing the fitted coverage distribution parameters

    """

    if filename is None and output_dir is None:
        raise ValueError(
            'Error: either "filename" or "output_dir" arguments must be defined.')
    elif filename is None:
        filename = output_dir + '/GaussDistParams.pkl'
    if filename[-4:] != '.pkl':
        raise ValueError(
            'Error: file ' + filename + ' does not have ".pkl" file extension, distribution parameters could not be loaded.')
    if not __os.path.isfile(filename):
        raise ValueError(
            'Error: file ' + filename + ' does not exist, distribution parameters could not be loaded.')

    return load_obj(filename.split('.pkl')[0])


def load_bedfile_from_file(filename=None, output_dir=None, bam_filename=None):
    """

    Loads the bedfile containing previous ploidy estimated for the given sample from the path specified in filename.

    :param filename: The path to the bedfile. If not supplied, the bed file is attempted to be loaded from [output_dir]/[bam_filename]_ploidy.bed. (default: None) (str)
    :param output_dir: The path to the directory where the default bedfile is located. (default: None) (str)
    :param bam_filename: the filename of the BAM file of the sample (default: None) (str)

    :returns: the pandas DataFrame of the bedfile

    """

    if filename is None and (output_dir is None or bam_filename is None):
        raise ValueError(
            'Error: either "filename" or "output_dir" and "bam_filename" arguments must be defined.')
    elif filename is None:
        filename = output_dir + '/' + bam_filename.split('.bam')[0] + '_ploidy.bed'
    if not __os.path.isfile(filename):
        raise ValueError('Error: file ' + filename + ' does not exist, bedfile could not be loaded.')

    cols = ['chrom', 'chromStart', 'chromEnd', 'ploidy', 'LOH']
    return __pd.read_csv(filename, header=0, names=cols)


def __print_filtered_results(output_dir, filename, unique_ploidies, score_lim_dict, control_samples, FPs_per_genome):
    """

    Prints filtered results (using the optimized score values) to a file.

    :param filename: The desired path to the file. (str)
    :param unique_ploidies: The list of unique ploidies in the genome. (list)
    :param output_dir: the path to the directory where non-filtered results are located (str)
    :param score_lim_dict: a dictionary containing optimized score values for each ploidy (dict)
    :param control_samples: a list of control sample filenames (array-like)
    :param FPs_per_genome: the maximum number of false positive mutations allowed in a control sample (int)

    """

    # subprocess.check_call('rm ' + self.output_dir + '/filtered_results.csv', shell=True)
    if __os.path.isfile(filename):
        __subprocess.check_call('rm ' + filename, shell=True)

    with open(filename, 'a') as f:
        f.write('# IsoMut2py filtered results - ' + str(__datetime.now()).split('.')[0] + '\n')
        f.write('# Original results:\n')
        f.write('#\t' + output_dir + '/all_SNVs.isomut2\n')
        f.write('#\t' + output_dir + '/all_indels.isomut2\n')
        f.write('# Control samples:\n')
        for s in control_samples:
            f.write('#\t' + s + '\n')
        f.write('# Total allowed number of false positives per genome: ' + str(FPs_per_genome) + '\n')
        f.write('#\n')
        f.write('#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n')

    filter_cmd = 'cat ' + output_dir + '/all_SNVs.isomut2 | awk \'BEGIN{FS=\"\t\"; OFS=\"\t\";}{'
    for i in range(len(unique_ploidies)):
        filter_cmd += 'if ($11 == ' + str(unique_ploidies[i]) + ' && $5 > ' + str(
            score_lim_dict['SNV'][i]) + ') print $0; '
    filter_cmd += '}\' >> ' + filename

    __subprocess.check_call(filter_cmd, shell=True)

    filter_cmd = 'cat ' + output_dir + '/all_indels.isomut2 | awk \'BEGIN{FS=\"\t\"; OFS=\"\t\";}{'
    for i in range(len(unique_ploidies)):
        filter_cmd += 'if ($4 == "INS" && $11 == ' + str(unique_ploidies[i]) + ' && $5 > ' + str(
            score_lim_dict['INS'][i]) + ') print $0; '
        filter_cmd += 'if ($4 == "DEL" && $11 == ' + str(unique_ploidies[i]) + ' && $5 > ' + str(
            score_lim_dict['DEL'][i]) + ') print $0; '
    filter_cmd += '}\' >> ' + filename

    __subprocess.check_call(filter_cmd, shell=True)


def load_mutations(output_dir, filename=None):
    """

    Loads mutations from a file or a list of files.

    :param filename: The path to the file, where mutations are stored. A list of paths can be also supplied, in this case, all of them will be loaded to a single dataframe. If None, the file [output_dir]/filtered_mutations.csv will be loaded. (default: None) (str)
    :param output_dir: The path to the directory containing the file filtered_results.csv. (str)

    :returns: a pandas.DataFrame containing the mutations

    """
    if filename is not None:
        df_total = []
        if filename.__class__ == str:
            filename = [filename]
        for f in filename:
            if not __os.path.isfile(f):
                raise ValueError('File ' + f + ' does not exist, mutations cannot be loaded.')
            s = __subprocess.check_output('grep "^#" ' + f + ' | tail -1', shell=True)
            if (s.decode('utf-8').strip().split('\t') != ['#sample_name', 'chr', 'pos', 'type', 'score',
                                                          'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                          'ploidy']):
                raise ValueError('Incorrect header in file ' + f + ', mutations cannot be loaded.')
            else:
                df = __pd.read_csv(f, comment='#',
                                   names=['sample_name', 'chr', 'pos', 'type', 'score',
                                          'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                                   sep='\t',
                                   low_memory=False)
                df_total.append(df)
        df = __pd.concat(df_total)
    else:
        if not __os.path.isfile(output_dir + '/filtered_results.csv'):
            raise ValueError(
                'File ' + output_dir + '/filtered_results.csv does not exist, filtered results cannot be loaded.')
        else:
            df = __pd.read_csv(output_dir + '/filtered_results.csv', comment='#',
                               names=['sample_name', 'chr', 'pos', 'type', 'score',
                                      'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                               sep='\t',
                               low_memory=False)
    df.drop_duplicates(inplace=True)
    df['chr'] = df['chr'].apply(str)
    return df
