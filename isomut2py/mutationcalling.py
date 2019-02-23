try:
    from isomut2py import process
    from isomut2py import postprocess
    from isomut2py import io
    from isomut2py import plot
    from isomut2py import SAMTOOLS_MAX_DEPTH

    from datetime import datetime as datetime
    import time as time
    import os as os
    import subprocess as subprocess
except ImportError:
    print('ImportError in isomut2py.mutationcalling, MutationCaller object cannot be defined.')


class MutationCaller:
    '''

    The MutationCaller class is designed to keep all parameter values, directories and filepaths in one place that are needed for the mutation
    detection and postprocessing of a single or multiple sample(s).

    - List of basic parameters:
        - ref_fasta: The path to the fasta file of the reference genome. (str)

        - output_dir: The path to a directory that can be used for temporary files and output files. The user must have permission to write the directory. (str)

        - input_dir: The path to the directory, where the bam file(s) of the sample(s) is/are located. (str)

        - bam_filename: A list of the name(s) of the bam file(s) of the sample(s). (Without path, eg. ["sample_1.bam", "sample_2.bam", "sample_3.bam", ...].) (list of str)

        - samtools_fullpath: The path to samtools on the computer. (default: "samtools") (str)

    - Other parameters with default values:
        - n_min_block: The approximate number of blocks to partition the analysed genome to for parallel computing. The actual number might be slightly larger that this. (default: 200) (int)

        - n_conc_blocks: The number of blocks to process at the same time. (default: 4) (int)

        - chromosomes: The list of chromosomes to analyse. (default: all chromosomes included included in the reference genome specified in ref_fasta) (list of str)

        - base_quality_limit: The base quality limit used by samtools in order to decide if a base should be included in the pileup file. (default: 30) (int)

        - samtools_flags: The samtools flags to be used for pileup file generation. (default: " -B -d 1000 ") (str)

        - unique_mutations_only: If True, only those mutations are sought that are unique to specific samples. Setting it to False greatly increases computation time, but allows for the detection of shared mutations and thus the analysis of phylogenetic connections between samples. If True, print_shared_by_all is ignored. (default: True) (boolean)

        - print_shared_by_all: If False, mutations that are present in all analysed samples are not printed to the output files. This decreases both memory usage and computation time. (default: False) (boolean)

        - min_sample_freq: The minimum frequency of the mutated base at a given position in the mutated sample(s). (default: 0.21) (float)

        - min_other_ref_freq: The minimum frequency of the reference base at a given position in the non-mutated sample(s). (default: 0.95) (float)

        - cov_limit: The minimum coverage at a given position in the mutated sample(s). (default: 5) (float)

        - min_gap_dist_snv: Minimum genomic distance from an identified SNV for a position to be considered as a potential mutation. (default: 0) (int)

        - min_gap_dist_indel: Minimum genomic distance from an identified indel for a position to be considered as a potential mutation. (default: 20) (int)

        - use_local_realignment: By default, mutation detection is run only once, with the samtools mpileup command with option -B. This turns off the probabilistic realignment of reads while creating the temporary pileup file, which might result in false positives due to alignment error. To filter out these mutations, setting the above parameter to True runs the whole mutation detection pipeline again for possibly mutated positions without the -B option as well, and only those mutations are kept that are still present with the probabilistic realignment turned on. Setting use_local_realignment = True increases runtime. (default: False) (boolean)

        - ploidy_info_filepath: Path to the file containing ploidy information (see below for details) of the samples. (default: None) (str)

        - constant_ploidy: The default ploidy to be used in regions not contained in ploidy_info_filepath. (default: 2) (int)

        - bedfile: Path to a bedfile containing a list of genomic regions, if mutations are only sought in a limited part of the genome. (default: None) (str)

        - control_samples: list of the bam filenames of the control samples. Control samples are not expected to have any unique mutations, thus they can be used to optimize the results. (default: None) (array-like)

        - FPs_per_genome: the maximum number of false positive mutations allowed in a control sample (default: None) (int)

        - mutations: a pandas DataFrame of mutations found in the sample set (default: None) (pandas.DataFrame)

        - chrom_length: the length of chromosomes (default: not set) (array-like)

        - genome_length: the total length of the genome (default: not set) (int)

        - IDspectra: a dictionary containing the indel spectra of all the samples in the dataset (default: not set) (dict with keys=bam_filename and values=array-like)

        - DNVspectra: a dictionary containing the dinucleotide variation spectra of all the samples in the dataset (default: not set) (dict with keys=bam_filename and values=array-like)

        - SNVspectra: a dictionary containing the single nucleotide variation spectra of all the samples in the dataset (default: not set) (dict with keys=bam_filename and values=array-like)

    '''

    def __init__(self, **kwargs):
        """

        Initializes the MutationCaller object by setting attributes to the ones defined in kwargs.

        :param kwargs: keyword arguments

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __check_params(self, list_of_params):
        """

        Checks and updates parameters values listed in list_of_param.

        :param list_of_params: parameter values to check (array-like)

        """
        def get_default_chroms_with_len(ref_genome, chrom_list):
            """

            Gets the length of chromosomes defined in chrom_list or if chrom_list is empty, gets the default list of chromosomes from a specified
            reference genome fasta file (set in ref_genome) with their length.

            :param ref_genome: The path to the reference genome fasta file.
            :param chrom_list: A predefined list of chromosomes to be included.

            :return: (chrom, lens):
                - chrom: a list of detected chromosomes (list of str)
                - len: a list of their lengths (list of int)

            """
            newchroms, newlens = [], []
            with open(ref_genome + '.fai') as f_h:
                for line in f_h:
                    chrom, leng = line.split('\t')[0], line.split('\t')[1]
                    if (chrom in chrom_list or len(chrom_list) == 0):
                        newchroms.append(chrom)
                        newlens.append(int(leng))
            if len(chrom_list) > 0:
                return chrom_list, [newlens[newchroms.index(c)] for c in chrom_list]
            else:
                return sorted(newchroms), [newlens[newchroms.index(c)] for c in sorted(newchroms)]

        defaults = dict()
        defaults['unique_mutations_only'] = True
        defaults['min_sample_freq'] = 0.21
        defaults['min_other_ref_freq'] = 0.95
        defaults['cov_limit'] = 5
        defaults['min_gap_dist_snv'] = 0
        defaults['min_gap_dist_indel'] = 20
        defaults['use_local_realignment'] = False
        defaults['ploidy_info_filepath'] = 'no_ploidy_info'
        defaults['constant_ploidy'] = 2
        defaults['print_shared_by_all'] = False
        defaults['n_min_block'] = 200
        defaults['n_conc_blocks'] = 4
        defaults['chromosomes'] = None
        defaults['base_quality_limit'] = 30
        defaults['bedfile'] = None
        defaults['samtools_flags'] = ' -B -d ' + str(SAMTOOLS_MAX_DEPTH) + ' '
        defaults['samtools_fullpath'] = 'samtools'
        defaults['control_samples'] = None
        defaults['FPs_per_genome'] = None
        defaults['mutations'] = None

        for p in list_of_params:
            if p in ['output_dir', 'input_dir', 'bam_filename']:
                if not hasattr(self, p):
                    raise ValueError('Error: Value of "' + p + '" is not set, cannot proceed.')
                elif p == 'input_dir' or p == 'bam_filename':
                    if not os.path.isdir(self.input_dir):
                        raise ValueError(
                            'Error: Input directory "' + self.input_dir + '" does not exist, cannot proceed.')
                    for bm in self.bam_filename:
                        if not os.path.isfile(self.input_dir + '/' + bm):
                            raise ValueError(
                                'Error: Bam file "' + self.input_dir + '/' + bm + '" does not exist, '
                                                                                  'cannot proceed.')
            elif p == 'samtools_fullpath':
                if not hasattr(self, p):
                    setattr(self, p, 'samtools')
                s = subprocess.getoutput(self.samtools_fullpath).strip().split(' ')[0] == 'Program:'
                if not s:
                    error_msg = 'Error: samtools not found, cannot proceed.'
                    error_msg += '\n'
                    error_msg += 'Specify path to samtools with the "samtools_fullpath" keyword argument.'
                    raise ValueError(error_msg)
            elif p == 'ref_fasta':
                if not hasattr(self, p):
                    raise ValueError('Error: Value of "' + p + '" is not set, cannot proceed.')
                if not os.path.isfile(self.ref_fasta):
                    raise ValueError(
                        'Error: Reference genome file "' + self.ref_fasta + '" does not exist, cannot proceed.')
                if not os.path.isfile(self.ref_fasta + '.fai'):
                    error_msg = 'Error: No faidx file found for reference genome file "' + self.ref_fasta + '", cannot proceed.'
                    error_msg += '\n'
                    error_msg += 'Use the samtools command: samtools faidx [ref.fasta]'
                    raise ValueError(error_msg)
            elif p == 'chromosomes':
                if not hasattr(self, 'chromosomes') or self.chromosomes is None or not hasattr(self, 'chrom_length'):
                    if not os.path.isfile(self.ref_fasta):
                        raise ValueError(
                            'Error: Reference genome file "' + self.ref_fasta + '" does not exist, cannot proceed.')
                    if not os.path.isfile(self.ref_fasta + '.fai'):
                        error_msg = 'Error: No faidx file found for reference genome file "' + self.ref_fasta + '", cannot proceed.'
                        error_msg += '\n'
                        error_msg += 'Use the samtools command: samtools faidx [ref.fasta]'
                        raise ValueError(error_msg)
                    if hasattr(self, 'chromosomes') and self.chromosomes is not None:
                        oldchrom = self.chromosomes
                    else:
                        oldchrom = []
                    c, l = get_default_chroms_with_len(self.ref_fasta, oldchrom)
                    setattr(self, 'chromosomes', c)
                    setattr(self, 'chrom_length', l)
                    setattr(self, 'genome_length', sum(l))
            elif p in ['IDspectra', 'DNVspectra', 'SNVspectra']:
                if not hasattr(self, p):
                    raise ValueError('Error: ' + p + ' dictionary not yet defined.')
            elif not hasattr(self, p):
                setattr(self, p, defaults[p])

            if p == 'min_sample_freq' and (self.min_sample_freq < 0 or self.min_sample_freq > 1):
                print('Value of "min_sample_freq" not in range [0,1], using default value of 0.21.')
                setattr(self, 'min_sample_freq', defaults['min_sample_freq'])
            if p == 'min_other_ref_freq' and (self.min_other_ref_freq < 0 or self.min_other_ref_freq > 1):
                print('Value of "min_other_ref_freq" not in range [0,1], using default value of 0.95.')
                setattr(self, 'min_other_ref_freq', defaults['min_other_ref_freq'])
            if p == 'ploidy_info_filepath' and not os.path.isfile(self.ploidy_info_filepath):
                print('File ' + self.ploidy_info_filepath + ' does not exist. Using constant default ploidy of 2.')
                setattr(self, 'ploidy_info_filepath', defaults['ploidy_info_filepath'])

    def run_isomut2_mutdet(self, **kwargs):
        """

        Runs IsoMut2 mutation detection pipeline on the MutationDetection object, using parameter values specified in the respective attributes of
        the object.

        :param kwargs: keyword arguments for MutationDetection object attributes

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['n_min_block',
                             'n_conc_blocks',
                             'chromosomes',
                             'ref_fasta',
                             'bam_filename',
                             'input_dir',
                             'output_dir',
                             'samtools_fullpath',
                             'samtools_flags',
                             'bedfile',
                             'min_sample_freq',
                             'min_other_ref_freq',
                             'cov_limit',
                             'base_quality_limit',
                             'min_gap_dist_snv',
                             'min_gap_dist_indel',
                             'constant_ploidy',
                             'ploidy_info_filepath',
                             'unique_mutations_only',
                             'print_shared_by_all',
                             'use_local_realignment'])

        if self.print_shared_by_all:
            msg = 'Warning: printing mutations shared by all samples might necessiate a large amount of disk space. ' \
                  'If your samples are largely different from the reference genome used and are generally similar to each other, ' \
                  'you should consider ' \
                  'skipping mutations shared by all of your samples. ' \
                  'To do this, set the attribute "print_shared_by_all" to False. ' \
                  '\n\n'
            print(msg)

        starting_time = datetime.now()
        level = 0

        print('\t' * level + time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime()) + ' - Mutation detection with IsoMut2\n')

        if (not self.use_local_realignment):  # single run only without local realignment
            process.single_run(mutcallObject=self, level=level)
        else:
            process.double_run(mutcallObject=self, level=level)

        finish_time = datetime.now()
        total_time = finish_time - starting_time
        total_time_h = int(total_time.seconds / 3600)
        total_time_m = int((total_time.seconds % 3600) / 60)
        total_time_s = (total_time.seconds % 3600) % 60
        print('\n' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' - IsoMut2 finished. (' + str(
            total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), ' + str(total_time_m) + ' min(s), ' + str(
            total_time_s) + ' sec(s).)')

    def optimize_results(self, control_samples, FPs_per_genome, **kwargs):
        """

        Optimizes the list of detected mutations according to the list of control samples and desired level of false positives set by the user.
        Filtered results will be loaded to the mutations attribute of the MutationDetection object as a pandas.DataFrame. Optimized values for the
        score are stored in the attribute optimized_score_values.

        :param control_samples: List of sample names that should be used as control samples in the sense, that no unique mutations are expected in them. (The sample names listed here must match a subset of the sample names listed in the attribute bam_filename.) (list of str)
        :param FPs_per_genome: The total number of false positives tolerated in a control sample. (int)
        :param kwargs: possible keyword arguments besides MutationCaller attributes:

            - plot_roc_curve: If True, ROC curves will be plotted as a visual representation of the optimization process. (default: False) (boolean)
            - plot_tuning_curve: If True, tuning curves displaying the number of mutations found in different samples with different score filters will be plotted as a visual representation of the optimization process. (default: False) (boolean)

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        setattr(self, 'control_samples', control_samples)
        setattr(self, 'FPs_per_genome', FPs_per_genome)

        self.__check_params(['bam_filename', 'output_dir'])

        score_lim_dict, filtered_muts = postprocess.optimize_results(sample_names=self.bam_filename,
                                                                     control_samples=self.control_samples,
                                                                     FPs_per_genome=self.FPs_per_genome,
                                                                     plot_roc=kwargs.get('plot_roc', False),
                                                                     plot_tuning_curve=kwargs.get('plot_tuning_curve',
                                                                                                  False),
                                                                     output_dir=self.output_dir)

        self.optimized_score_values = score_lim_dict
        self.mutations = filtered_muts

    def load_mutations(self, filename=None):
        """

        Loads mutations from a file or a list of files into the mutations attribute.

        :param filename: The path to the file, where mutations are stored. A list of paths can be also supplied, in this case, all of them will be loaded to a single dataframe. The mutations attribute of the MutationDetection object will be set to the loaded dataframe. If None, the file [output_dir]/filtered_mutations.csv will be loaded. (default: None) (str)

        """

        self.__check_params(['output_dir'])

        df = io.load_mutations(output_dir=self.output_dir,
                               filename=filename)

        self.mutations = df

    def plot_mutation_counts(self,
                             unique_only=False,
                             return_string=False,
                             mutations_filename=None,
                             mutations_dataframe=None,
                             **kwargs):
        """

        Plots the number of mutations found in all the samples in different ploidy regions.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: False) (boolean)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param mutations_filename: The path to the file, where mutations are stored, if the mutations attribute of the object does not exist, its value will be set to the file defined here. (default: None) (str)
        :param mutations_dataframe: If the mutations are not to be loaded from a file, but are contained in a pandas.DataFrame, this can be supplied with setting the mutations_dataframe parameter. (default: None) (pandas.DataFrame)
        :param kwargs: possible keyword arguments besides MutationCaller attributes

            - control_samples: List of sample names that should be used as control samples in the sense, that no unique mutations are expected in them. (The sample names listed here must match a subset of the sample names listed in bam_filename.) (list of str)

        :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a matplotlib figure.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'output_dir', 'control_samples'])

        return plot.plot_mutation_counts(sample_names=self.bam_filename,
                                         mutations_dataframe=mutations_dataframe,
                                         unique_only=unique_only,
                                         return_string=return_string,
                                         mutations_filename=mutations_filename,
                                         output_dir=self.output_dir,
                                         control_samples=self.control_samples)

    def plot_hierarchical_clustering(self,
                                     mutations_dataframe=None,
                                     return_string=False,
                                     mutations_filename=None,
                                     **kwargs):
        """

        Generates a heatmap based on the number of shared mutations found in all possible sample pairs.
        A dendrogram is also added that is the result of hierarchical clustering of the samples.

        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param mutations_filename: The path to the file, where mutations are stored, if the mutations attribute of the object does not exist, its value will be set to the file defined here. (default: None) (str)
        :param mutations_dataframe: If the mutations are not to be loaded from a file, but are contained in a pandas.DataFrame, this can be supplied with setting the mutations_dataframe parameter. (default: None) (pandas.DataFrame)
        :param kwargs: keyword arguments for MutationDetection object attributes

        :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a matplotlib figure.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'output_dir'])

        return plot.plot_hierarchical_clustering(sample_names=self.bam_filename,
                                                 mutations_dataframe=mutations_dataframe,
                                                 mutations_filename=mutations_filename,
                                                 output_dir=self.output_dir,
                                                 return_string=return_string)

    def calculate_SNV_spectrum(self, unique_only=True, **kwargs):
        """

        Calculates the triplet spectrum from the dataframe of relevant mutations, using the fasta file of the reference genome. Results are stored
        in the SNVspectra attribute of the object.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param kwargs: keyword arguments for MutationCaller attributes

            - mutations_dataframe: A pandas DataFrame where mutations are listed. (default: not set) (pandas.DataFrame)
            - sample_names: A list of sample names to plot results for, a subset of the list in "bam_filename" attribute. (default: not set) (list)
            - mutations_filaname: the path to the file where mutations are stored (default: not set) (str)

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'output_dir', 'chromosomes', 'ref_fasta', 'mutations'])

        SNVspectra = postprocess.calculate_SNV_spectrum(ref_fasta=self.ref_fasta,
                                                        mutations_dataframe=kwargs.get('mutations_dataframe',
                                                                                       self.mutations),
                                                        sample_names=kwargs.get('sample_names', self.bam_filename),
                                                        unique_only=unique_only,
                                                        chromosomes=self.chromosomes,
                                                        output_dir=self.output_dir,
                                                        mutations_filename=kwargs.get('mutations_filename', None))
        self.SNVspectra = SNVspectra

    def plot_SNV_spectrum(self, return_string=False, normalize_to_1=False, **kwargs):
        """

        Plots the triplet spectra for the samples in attribute bam_filename.

        :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['SNVspectra'])

        return plot.plot_SNV_spectrum(spectrumDict=self.SNVspectra,
                                      return_string=return_string,
                                      normalize_to_1=normalize_to_1)


    def calculate_and_plot_SNV_spectrum(self,
                                        unique_only=True,
                                        normalize_to_1=False,
                                        return_string=False,
                                        **kwargs):
        """

        Calculates and plots the SNV spectra of the samples listed in attribute bam_filename.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param kwargs: keyword arguments for MutationCaller attributes

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.calculate_SNV_spectrum(unique_only=unique_only, **kwargs)
        return self.plot_SNV_spectrum(return_string=return_string, normalize_to_1=normalize_to_1, **kwargs)

    def calculate_indel_spectrum(self, unique_only=True, **kwargs):
        """

        Calculates the indel spectrum from the dataframe of relevant mutations, using the fasta file of the reference genome.
        Results are stored in the IDspectra attribute of the object.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param kwargs: keyword arguments for MutationCaller attributes

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'output_dir', 'chromosomes', 'ref_fasta', 'mutations'])

        IDspectra = postprocess.calculate_indel_spectrum(ref_fasta=self.ref_fasta,
                                                         mutations_dataframe=kwargs.get('mutations_dataframe',
                                                                                        self.mutations),
                                                         sample_names=kwargs.get('sample_names', self.bam_filename),
                                                         unique_only=unique_only,
                                                         chromosomes=self.chromosomes,
                                                         output_dir=self.output_dir,
                                                         mutations_filename=kwargs.get('mutations_filename', None))
        self.IDspectra = IDspectra

    def plot_indel_spectrum(self, return_string=False, normalize_to_1=False, **kwargs):
        """

        Plots the indel spectra for the samples in attribute bam_filename.

        :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['IDspectra'])

        return plot.plot_indel_spectrum(spectrumDict=self.IDspectra,
                                        return_string=return_string,
                                        normalize_to_1=normalize_to_1)

    def calculate_and_plot_indel_spectrum(self,
                                          unique_only=True,
                                          normalize_to_1=False,
                                          return_string=False,
                                          **kwargs):
        """

        Calculates and plots the indel spectra of the samples listed in attribute bam_filename.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param kwargs: keyword arguments for MutationCaller attributes

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.calculate_indel_spectrum(unique_only=unique_only, **kwargs)
        return self.plot_indel_spectrum(return_string=return_string, normalize_to_1=normalize_to_1, **kwargs)

    def calculate_DNV_matrix(self, unique_only=True, **kwargs):
        """

        Calculates the dinucleotide variation spectrum in matrix format from the dataframe of relevant mutations, using the fasta file of the
        reference genome. Results are stored in the DNVmatrice attribute of the object.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param kwargs: keyword arguments for MutationCaller attributes

            - mutations_dataframe: A pandas DataFrame where mutations are listed. (default: not set) (pandas.DataFrame)
            - sample_names: A list of sample names to plot results for, a subset of the list in "bam_filename" attribute. (default: not set) (list)
            - mutations_filaname: the path to the file where mutations are stored (default: not set) (str)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'output_dir', 'chromosomes', 'ref_fasta', 'mutations'])

        DNVmatrice = postprocess.calculate_DNV_matrix(
            mutations_dataframe=kwargs.get('mutations_dataframe', self.mutations),
            sample_names=kwargs.get('sample_names', self.bam_filename),
            unique_only=unique_only,
            chromosomes=self.chromosomes,
            output_dir=self.output_dir,
            mutations_filename=kwargs.get('mutations_filename', None))

        self.DNVmatrice = DNVmatrice

    def plot_DNV_heatmap(self, return_string=False, **kwargs):
        """

        Plots the DNV spectra as a heatmap for the samples in attribute bam_filename.

        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['DNVmatrice'])

        return plot.plot_DNV_heatmap(matrixDict=self.DNVmatrice,
                                     return_string=return_string)

    def calculate_and_plot_DNV_heatmap(self,
                                       unique_only=True,
                                       return_string=False,
                                       **kwargs):
        """

        Calculates and plots the DNV spectra as a heatmap of the samples listed in attribute bam_filename.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param kwargs: keyword arguments for MutationCaller attributes

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.calculate_DNV_matrix(unique_only=unique_only, **kwargs)
        return self.plot_DNV_heatmap(return_string=return_string, **kwargs)

    def calculate_DNV_spectrum(self, unique_only=True, **kwargs):
        """

        Calculates the dinucleotide variation spectrum from the dataframe of relevant mutations, using the fasta file of the reference genome.
        Results are stored in the DNVmatrice attribute of the object.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param kwargs: keyword arguments for MutationCaller attributes

            - mutations_dataframe: A pandas DataFrame where mutations are listed. (default: not set) (pandas.DataFrame)
            - sample_names: A list of sample names to plot results for, a subset of the list in "bam_filename" attribute. (default: not set) (list)
            - mutations_filaname: the path to the file where mutations are stored (default: not set) (str)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'output_dir', 'chromosomes', 'ref_fasta', 'mutations'])

        DNVspectra = postprocess.calculate_DNV_spectrum(
            mutations_dataframe=kwargs.get('mutations_dataframe', self.mutations),
            sample_names=kwargs.get('sample_names', self.bam_filename),
            unique_only=unique_only,
            chromosomes=self.chromosomes,
            output_dir=self.output_dir,
            mutations_filename=kwargs.get('mutations_filename', None))

        self.DNVspectra = DNVspectra

    def plot_DNV_spectrum(self, return_string=False, normalize_to_1=False, **kwargs):
        """

        Plots the DNV spectra for the samples in attribute bam_filename.

        :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['DNVspectra'])

        return plot.plot_DNV_spectrum(spectrumDict=self.DNVspectra,
                                      return_string=return_string,
                                      normalize_to_1=normalize_to_1)

    def calculate_and_plot_DNV_spectrum(self,
                                        unique_only=True,
                                        return_string=False,
                                        normalize_to_1=False,
                                        **kwargs):
        """

        Calculates and plots the DNV spectra of the samples listed in attribute bam_filename.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param kwargs: keyword arguments for MutationCaller attributes

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.calculate_DNV_spectrum(unique_only=unique_only, **kwargs)
        return self.plot_DNV_spectrum(return_string=return_string, normalize_to_1=normalize_to_1, **kwargs)

    def plot_rainfall(self,
                      unique_only=True,
                      return_string=False,
                      **kwargs):
        """

        Plots the rainfall plot of mutations found in the samples listed in the attribute bam_filename. Displaying rainfall plots is a good
        practice to detect mutational clusters throughout the genome. The horizontal axis is the genomic position of each mutation,
        while the vertical shows the genomic distance between the mutation and the previous one. Thus mutations that are clustered together appear
        close to each other horizontally and on the lower part of the plot vertically.

        :param unique_only: If True, only unique mutations are plotted for each sample. (default: True) (boolean)
        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
        :param kwargs: keyword arguments for MutationCaller attributes

            - mutations_dataframe: A pandas DataFrame where mutations are listed. (default: not set) (pandas.DataFrame)
            - sample_names: A list of sample names to plot results for, a subset of the list in "bam_filename" attribute. (default: not set) (list)
            - mut_types: list of mutation types to display (default: not set, displaying SNVs, insertions and deletions) (a list containing any combination of these items: ['SNV', 'INS', 'DEL'])
            - plot_range: The genomic range to plot. (default: not set) (str, example: "chr9:2342-24124")

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'chromosomes', 'mutations'])

        return plot.plot_rainfall(mutations_dataframe=kwargs.get('mutations_dataframe', self.mutations),
                                  chromosomes=self.chromosomes,
                                  chrom_length=self.chrom_length,
                                  sample_names=kwargs.get('sample_names', self.bam_filename),
                                  return_string=return_string,
                                  muttypes=kwargs.get('muttypes', ['SNV', 'INS', 'DEL']),
                                  unique_only=unique_only,
                                  plot_range = kwargs.get('plot_range', None))

    def check_pileup(self, chrom_list, from_pos_list, to_pos_list, print_original=True, savetofile=None,
                     **kwargs):
        """

        Loads pileup information for a list of genomic regions.

        :param chrom_list: List of chromosomes for the regions. (list of str)
        :param from_pos_list: List of starting positions for the regions. (list of int)
        :param to_pos_list: List of ending positions for the regions. (list of int)
        :param print_original: If True, prints the original string generated with samtools mpileup as well (default: True) (bool)
        :param savetofile: If savetofile is specified, results will be writted to that path. Otherwise, results are written to [output_dir]/checkPileup_tmp.csv (default: None) (str)

        :returns: df: The processed results of the pileup, containing information on all samples in a pandas.DataFrame.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['ref_fasta', 'input_dir', 'bam_filename', 'output_dir', 'samtools_fullpath',
                             'base_quality_limit', 'samtools_flags'])

        return postprocess.check_pileup(chrom_list=chrom_list,
                                        from_pos_list=from_pos_list,
                                        to_pos_list=to_pos_list,
                                        ref_fasta=self.ref_fasta,
                                        input_dir=self.input_dir,
                                        bam_filename=self.bam_filename,
                                        output_dir=self.output_dir,
                                        samtools_fullpath=self.samtools_fullpath,
                                        base_quality_limit=self.base_quality_limit,
                                        samtools_flags=self.samtools_flags,
                                        print_original=print_original,
                                        filename=savetofile)

    def get_details_for_mutations(self,
                                  mutations_filename=None,
                                  **kwargs):
        """

        Get detailed results for the list of mutations contained in the mutations attribute of the object.

        :param mutations_filename: The path(s) to the file(s) where mutations are stored. (default: None) (list of str)
        :param kwargs: keyword arguments for MutationDetection object attributes

            - mutations_dataframe: A pandas DataFrame where mutations are listed. (default: not set) (pandas.DataFrame)

        :returns: df_joined: A dataframe containing the detailed results. (pandas.DataFrame)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['ref_fasta', 'input_dir', 'bam_filename', 'output_dir', 'samtools_fullpath',
                             'base_quality_limit', 'samtools_flags', 'mutations'])

        return postprocess.get_details_for_mutations(ref_fasta=self.ref_fasta,
                                                     input_dir=self.input_dir,
                                                     bam_filename=self.bam_filename,
                                                     mutations_dataframe=kwargs.get('mutations_dataframe',
                                                                                    self.mutations),
                                                     output_dir=self.output_dir,
                                                     mutations_filename=mutations_filename,
                                                     samtools_fullpath=self.samtools_fullpath,
                                                     base_quality_limit=self.base_quality_limit,
                                                     samtools_flags=self.samtools_flags)

    def decompose_indel_spectra(self,
                                sample_names=None,
                                unique_only=True,
                                signatures_file=None,
                                equal_initial_proportions=False,
                                tol=0.0001,
                                max_iter=1000,
                                filter_percent=0,
                                filter_count=0,
                                keep_top_n=None,
                                use_signatures=None,
                                ignore_signatures=None,
                                **kwargs):
        """

        Run the whole pipeline of decomposing indel spectra for the samples specified in sample_names.

        :param sample_names: The list of sample names analysed. (list of str)
        :param unique_only: If True, only unique mutations are used to construct the original spectrum.
        :param signatures_file: The path to the csv file containing the signature matrix. (str)
        :param equal_initial_proportions: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)
        :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)
        :param max_iter: The maximum number of iterations (default: 1000) (int)
        :param filter_percent: Filter signatures, that contribute less than filter_percent of the mutations in the sample. (default: 0) (float)
        :param filter_count: Filter signatures, that contribute less than filter_count number of mutations in the sample. (default: 0) (int)
        :param keep_top_n: Only keep those signatures that contribute to the mixture with the top keep_top_n number of mutation. (default: None) (int)
        :param use_signatures: Use a specific subset of all signatures. A list of signature names. (default: None) (list of str)
        :param ignore_signatures: Exclude a specific subset of all signatures from the analysis. A list of signature names. (default: None) (list of str)
        :param kwargs: keyword arguments for MutationDetection object attributes

        :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'IDspectra') or self.IDspectra.__class__ != dict:
            print('Indel spectra have not been calculated yet, starting with this step.')
            self.calculate_indel_spectrum(unique_only=unique_only, **kwargs)

        return postprocess.decompose_indel_spectra(IDspectrumDict=self.IDspectra,
                                                   sample_names=sample_names,
                                                   signatures_file=signatures_file,
                                                   equal_initial_proportions=equal_initial_proportions,
                                                   tol=tol,
                                                   max_iter=max_iter,
                                                   filter_percent=filter_percent,
                                                   filter_count=filter_count,
                                                   keep_top_n=keep_top_n,
                                                   use_signatures=use_signatures,
                                                   ignore_signatures=ignore_signatures)

    def decompose_DNV_spectra(self,
                              sample_names=None,
                              unique_only=True,
                              signatures_file=None,
                              equal_initial_proportions=False,
                              tol=0.0001,
                              max_iter=1000,
                              filter_percent=0,
                              filter_count=0,
                              keep_top_n=None,
                              use_signatures=None,
                              ignore_signatures=None,
                              **kwargs):
        """

        Run the whole pipeline of decomposing DNV spectra for the samples specified in sample_names.

        :param sample_names: The list of sample names analysed. (list of str)
        :param unique_only: If True, only unique mutations are used to construct the original spectrum.
        :param signatures_file: The path to the csv file containing the signature matrix. (str)
        :param equal_initial_proportions: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)
        :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)
        :param max_iter: The maximum number of iterations (default: 1000) (int)
        :param filter_percent: Filter signatures, that contribute less than filter_percent of the mutations in the sample. (default: 0) (float)
        :param filter_count: Filter signatures, that contribute less than filter_count number of mutations in the sample. (default: 0) (int)
        :param keep_top_n: Only keep those signatures that contribute to the mixture with the top keep_top_n number of mutation. (default: None) (int)
        :param use_signatures: Use a specific subset of all signatures. A list of signature names. (default: None) (list of str)
        :param ignore_signatures: Exclude a specific subset of all signatures from the analysis. A list of signature  names. (default: None) (list of str)
        :param kwargs: keyword arguments for MutationDetection object attributes

        :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'DNVspectra') or self.DNVspectra.__class__ != dict:
            print('DNV spectra have not been calculated yet, starting with this step.')
            self.calculate_DNV_spectrum(unique_only=unique_only, **kwargs)

        return postprocess.decompose_DNV_spectra(DNVspectrumDict=self.DNVspectra,
                                                 sample_names=sample_names,
                                                 signatures_file=signatures_file,
                                                 equal_initial_proportions=equal_initial_proportions,
                                                 tol=tol,
                                                 max_iter=max_iter,
                                                 filter_percent=filter_percent,
                                                 filter_count=filter_count,
                                                 keep_top_n=keep_top_n,
                                                 use_signatures=use_signatures,
                                                 ignore_signatures=ignore_signatures)

    def decompose_SNV_spectra(self,
                              sample_names=None,
                              unique_only=True,
                              signatures_file=None,
                              equal_initial_proportions=False,
                              tol=0.0001,
                              max_iter=1000,
                              filter_percent=0,
                              filter_count=0,
                              keep_top_n=None,
                              use_signatures=None,
                              ignore_signatures=None,
                              **kwargs):
        """

        Run the whole pipeline of decomposing SNV spectra for the samples specified in sample_names.

        :param sample_names: The list of sample names analysed. (list of str)
        :param unique_only: If True, only unique mutations are used to construct the original spectrum.
        :param signatures_file: The path to the csv file containing the signature matrix. (str)
        :param equal_initial_proportions: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)
        :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)
        :param max_iter: The maximum number of iterations (default: 1000) (int)
        :param filter_percent: Filter signatures, that contribute less than filter_percent of the mutations in the sample. (default: 0) (float)
        :param filter_count: Filter signatures, that contribute less than filter_count number of mutations in the sample. (default: 0) (int)
        :param keep_top_n: Only keep those signatures that contribute to the mixture with the top keep_top_n number of mutation. (default: None) (int)
        :param use_signatures: Use a specific subset of all signatures. A list of signature names. (default: None) (list of str)
        :param ignore_signatures: Exclude a specific subset of all signatures from the analysis. A list of signature names. (default: None) (list of str)
        :param kwargs: keyword arguments for MutationDetection object attributes

        :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'SNVspectra') or self.SNVspectra.__class__ != dict:
            print('SNV spectra have not been calculated yet, starting with this step.')
            self.calculate_SNV_spectrum(unique_only=unique_only, **kwargs)

        return postprocess.decompose_SNV_spectra(SNVspectrumDict=self.SNVspectra,
                                                 sample_names=sample_names,
                                                 signatures_file=signatures_file,
                                                 equal_initial_proportions=equal_initial_proportions,
                                                 tol=tol,
                                                 max_iter=max_iter,
                                                 filter_percent=filter_percent,
                                                 filter_count=filter_count,
                                                 keep_top_n=keep_top_n,
                                                 use_signatures=use_signatures,
                                                 ignore_signatures=ignore_signatures)
