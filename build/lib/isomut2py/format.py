try:
    from isomut2py import ploidyestimation

    import pandas as __pd
    import numpy as __np
    import os as __os
    import subprocess as __subprocess
except ImportError:
    print('ImportError in isomut2py.format, generating bed files will not work.')


def get_bed_format_for_sample(chromosomes, chrom_length, output_dir, bam_filename=None, ownbed_filepath=None):
    """

    Creates bed file of constant ploidies for a given sample from a file of positional ploidy data.
    If the ownbed_filepath attribute of the PloidyEstimation object is set, saves the bedfile to the path specified there. Otherwise,
    saves it to the output_dir with the "_ploidy.bed" suffix. Also sets the bed_dataframe attribute to the pandas.Dataframe containing the bed file.

    :param chromosomes: list of chromosomes (array-like)
    :param chrom_length: list of chromosome lengths in bp (array-like)
    :param output_dir: path to the directory where the PE_fullchrom_* files are located. (str)
    :param bam_filename: filename of the BAM file of the sample (default: None) (str)
    :param ownbed_filepath: path to the bed file where results should be saved (default: None) (str)

    :returns: (ownbed_filepath, df)

        - ownbed_filepath: path the the bed file where results are saved
        - df: the bed file in a pandas.DataFrame

    """

    def get_ploidy_ranges_for_chrom(outputdir, chrom, chr_list, before_list, after_list, pl_list, loh_list,
                                    chrom_len_dict):
        """

        Gets ranges of constant ploidies from a file of positional ploidy data.
        At breakpoints, the rounded average of the two bordering positions are taken as the breakpoint position.

        :param outputdir: The path to the directory, where the PE_fullchrom_[chrom].txt file is located. (str)
        :param chrom: The name of the chromosome to be analysed. (str)
        :param chr_list: The list of chromosomes to append current information to.
        :param before_list: The list of starting positions to append current information to.
        :param after_list: The list of ending positions to append current information to.
        :param pl_list: The list of ploidies to append current information to.
        :param loh_list: The list of LOHs to append current information to.
        :param chrom_len_dict: A dictionary containing {chromosome: chromosome length} pairs.

        :return: (chr_list, before_list, after_list, pl_list, loh_list)

        """
        df = __pd.read_csv(outputdir + '/PE_fullchrom_' + chrom + '.txt', sep='\t').sort_values(by='pos')
        p = __np.array(list(df['pos']))
        pl = __np.array(list(df['ploidy']))
        loh = __np.array(list(df['LOH']))
        pl_loh = __np.array([str(pl_c) + ',' + str(loh_c) for pl_c, loh_c in zip(pl, loh)])
        pl_loh_change = __np.where(pl_loh[:-1] != pl_loh[1:])[0]

        before_pos = 0
        after_pos = 0
        for i in range(len(pl_loh_change)):
            after_idx = pl_loh_change[i]
            after_pos = int(round(__np.mean([p[after_idx], p[after_idx + 1]])))
            chr_list.append(chrom)
            before_list.append(before_pos)
            after_list.append(after_pos)
            pl_list.append(pl[after_idx])
            loh_list.append(loh[after_idx])
            before_pos = after_pos + 1
        chr_list.append(chrom)
        before_list.append(before_pos)
        after_list.append(chrom_len_dict[chrom])
        pl_list.append(pl[-1])
        loh_list.append(loh[-1])

        return chr_list, before_list, after_list, pl_list, loh_list

    chrom_len_dict = {c: l for c, l in zip(chromosomes, chrom_length)}
    chr_list = []
    before_list = []
    after_list = []
    pl_list = []
    loh_list = []
    for c in chromosomes:
        if not __os.path.isfile(output_dir + '/PE_fullchrom_' + c + '.txt'):
            raise ValueError(
                'File ' + output_dir + '/PE_fullchrom_' + c + '.txt is not yet created, call "run_ploidy_estimation" '
                                                              'first.')
        chr_list, before_list, after_list, pl_list, loh_list = get_ploidy_ranges_for_chrom(output_dir, c,
                                                                                           chr_list, before_list,
                                                                                           after_list, pl_list,
                                                                                           loh_list, chrom_len_dict)
    df = __pd.DataFrame()
    df['chrom'] = chr_list
    df['chromStart'] = before_list
    df['chromEnd'] = after_list
    df['ploidy'] = pl_list
    df['LOH'] = loh_list

    if ownbed_filepath != None:
        df.to_csv(ownbed_filepath, index=False)
    elif bam_filename != None:
        df.to_csv(output_dir + '/' + bam_filename.split('.bam')[0] + '_ploidy.bed', index=False)
        ownbed_filepath = output_dir + '/' + bam_filename.split('.bam')[0] + '_ploidy.bed'
    else:
        df.to_csv(output_dir + '/' + 'ploidy.bed', index=False)
        ownbed_filepath = output_dir + '/' + 'ploidy.bed'

    return ownbed_filepath, df


def generate_ploidy_info_file(filename=None, sample_names=None, bed_filepaths=None, ploidy_estimation_objects=None,
                              sample_groups=None, group_bed_filepaths=None):
    """

    Generate ploidy info file for mutation detection in samples with nondefault ploidies. Make sure to supply one of the following arguments:
        - ploidy_estimation_objects
        - sample_names AND bed_filepaths
        - sample_groups AND group_bed_filepaths

    :param filename: The desired path to the generated ploidy info file. If None, ploidy information is saved to ploidy_info_file.txt in the current directory. (default: None) (str)
    :param sample_names: List of bam filenames for the samples, must be supplied together with bed_filepaths. (default: None) (list of str)
    :param bed_filepaths: Must be supplied together with sample_names. The list of filepaths to each bed file describing the given sample in sample_names. (default: None) (list of str)
    :param ploidy_estimation_objects: List of PloidyEstimation objects for each sample. (default: None) (list of isomut2py.PloidyEstimation)
    :param sample_groups: List of lists of str. Each list in sample_groups must contain the name of bam files in that group. Must be supplied together with group_bed_filepaths. (default: None) (list of list of str, example: [['sample1.bam', 'sample2.bam', 'sample3.bam'], ['sample4.bam', 'sample5.bam'], ['sample6.bam']])
    :param group_bed_filepaths: List of filepaths to the bed files describing each sample group in sample_groups. Must be supplied together with sample_groups. (default: None) (list of str, example: ['bedfile_of_samples123.txt', 'bedfile_of_samples45.txt' 'bedfile_of_samples6.txt'])

    """
    if filename == None:
        filename = 'ploidy_info_file.txt'
        print('Argument "filename" not set, saving ploidy info file to ploidy_info_file.txt')

    if (ploidy_estimation_objects == None and
            not (sample_names != None and bed_filepaths != None)
            and not (sample_groups != None and group_bed_filepaths != None)):
        raise TypeError(
            'Either "ploidy_estimation_objects" OR "sample_names" together with "bed_filepaths" OR "sample_groups" '
            'together with "group_bed_filepaths" must be defined.')

    __subprocess.call('rm ' + filename, shell=True)

    with open(filename, "a") as f:
        f.write('#file_path\tsample_names_list\n')
        if ploidy_estimation_objects != None:
            if ploidy_estimation_objects.__class__ != list:
                raise TypeError(
                    'Argument "ploidy_estimation_objects" must be a list.')
            for PE in ploidy_estimation_objects:
                if not isinstance(PE, ploidyestimation.PloidyEstimator):
                    raise TypeError('All elements of ploidy_estimation_objects must be PloidyEstimator objects.')
                if not hasattr(PE, "ownbed_filepath"):
                    raise ValueError(
                        'All PloidyEstimator objects must have valid "ownbed_filepath" arguments. Try running '
                        '"run_ploidy_estimation()" or set "ownbed_filepath" manually.')
                if not __os.path.isfile(PE.ownbed_filepath):
                    raise ValueError(
                        'All PloidyEstimator objects must have valid "ownbed_filepath" arguments. Try running '
                        '"run_ploidy_estimation()" or set "ownbed_filepath" manually.')
                else:
                    s = __subprocess.check_output('head -1 ' + PE.ownbed_filepath, shell=True)
                    sep = ','
                    if (len(s.decode('utf-8').strip().split('\t')) > 1):
                        sep = '\t'
                    elif (len(s.decode('utf-8').strip().split(';')) > 1):
                        sep = ';'
                    df = __pd.read_csv(PE.ownbed_filepath, sep=sep)
                    if "chrom" not in list(df.columns) or "chromStart" not in list(
                            df.columns) or "chromEnd" not in list(df.columns) or "ploidy" not in list(df.columns):
                        raise ValueError(
                            "File " + PE.ownbed_filepath + "does not have one of the following columns: chrom, "
                                                           "chromStart, chromEnd, ploidy.")
                    else:
                        df = df[['chrom', 'chromStart', 'chromEnd', 'ploidy']]
                        df.to_csv(PE.ownbed_filepath + '_im2format', sep='\t', index=False)
                        f.write(PE.ownbed_filepath + '_im2format\t' + PE.bam_filename + '\n')
        elif sample_names != None and bed_filepaths != None:
            if sample_names.__class__ != list:
                raise TypeError(
                    'Argument "sample_names" must be a list.')
            elif bed_filepaths.__class__ != list:
                raise TypeError(
                    'Argument "bed_filepaths" must be a list.')
            elif len(sample_names) != len(bed_filepaths):
                raise ValueError('Arguments "sample_names" and "bed_filepaths" must have the same length.')
            else:
                for sn, bfp in zip(sample_names, bed_filepaths):
                    if (not __os.path.isfile(bfp)):
                        raise ValueError('File ' + bfp + ' does not exist.')
                    else:
                        s = __subprocess.check_output('head -1 ' + bfp, shell=True)
                        sep = ','
                        if len(s.decode('utf-8').strip().split('\t')) > 1:
                            sep = '\t'
                        elif len(s.decode('utf-8').strip().split(';')) > 1:
                            sep = ';'
                        df = __pd.read_csv(bfp, sep=sep)
                        if ("chrom" not in list(df.columns) or "chromStart" not in list(
                                df.columns) or "chromEnd" not in list(df.columns) or "ploidy" not in list(df.columns)):
                            raise ValueError(
                                "File " + bfp + "does not have one of the following columns: chrom, chromStart, "
                                                "chromEnd, ploidy.")
                        else:
                            df = df[['chrom', 'chromStart', 'chromEnd', 'ploidy']]
                            df.to_csv(bfp + '_im2format', sep='\t', index=False)
                            f.write(bfp + '_im2format\t' + sn + '\n')
        elif sample_groups != None and group_bed_filepaths != None:
            if sample_groups.__class__ != list:
                raise TypeError(
                    'Argument "sample_groups" must be a list of lists.')
            for sg in sample_groups:
                if sg.__class__ != list:
                    raise TypeError(
                        'Argument "sample_groups" must be a list of lists.')
            if group_bed_filepaths.__class__ != list:
                raise TypeError(
                    'Argument "group_bed_filepaths" must be a list.')
            elif len(sample_groups) != len(group_bed_filepaths):
                raise ValueError('Arguments "sample_groups" and "group_bed_filepaths" must have the same length.')
            else:
                for sg, gbfp in zip(sample_groups, group_bed_filepaths):
                    if not __os.path.isfile(gbfp):
                        raise ValueError('File ' + gbfp + ' does not exist.')
                    else:
                        s = __subprocess.check_output('head -1 ' + gbfp, shell=True)
                        sep = ','
                        if len(s.decode('utf-8').strip().split('\t')) > 1:
                            sep = '\t'
                        elif len(s.decode('utf-8').strip().split(';')) > 1:
                            sep = ';'
                        df = __pd.read_csv(gbfp, sep=sep)
                        if ("chrom" not in list(df.columns) or "chromStart" not in list(
                                df.columns) or "chromEnd" not in list(df.columns) or "ploidy" not in list(df.columns)):
                            raise ValueError(
                                "File " + gbfp + "does not have one of the following columns: chrom, chromStart, "
                                                 "chromEnd, ploidy.")
                        else:
                            df = df[['chrom', 'chromStart', 'chromEnd', 'ploidy']]
                            df.to_csv(gbfp + '_im2format', sep='\t', index=False)
                            f.write(gbfp + '_im2format\t' + ', '.join(sg) + '\n')
