try:
    from isomut2py import process
    from isomut2py import io
    from isomut2py import bayesian
    from isomut2py import format
    from isomut2py import plot
    from isomut2py import compare

    from datetime import datetime as datetime
    import time as time
    import pandas as pd
    import os as os
    import subprocess as subprocess
    import sys as sys
except ImportError:
    print('ImportError in isomut2py.ploidyestimation, PloidyEstimator object cannot be defined.')

SAMTOOLS_MAX_DEPTH=1000

class PloidyEstimator:
    '''

    The PloidyEstimator class is designed to keep all parameter values, directories and filepaths in one place that are needed for the ploidy
    analysis of a single sample.

    - List of basic parameters:
        - ref_fasta: The path to the fasta file of the reference genome. (str)

        - output_dir: The path to a directory that can be used for temporary files and output files. The user must have permission to write the directory. (str)

        - input_dir: The path to the directory, where the bam file(s) of the sample(s) is/are located. (str)

        - bam_filename: The name of the bam file of the sample. (Without path, eg. "sample_1.bam".) (str)

        - samtools_fullpath: The path to samtools on the computer. (default: "samtools") (str)

    - Other parameters with default values:
        - n_min_block: The approximate number of blocks to partition the analysed genome to for  parallel computing. The actual number might be slightly larger that this. (default: 200) (int)

        - n_conc_blocks: The number of blocks to process at the same time. (default: 4) (int)

        - chromosomes: The list of chromosomes to analyse. (default: all chromosomes included included in the reference genome specified in ref_fasta) (list of str)

        - windowsize: The windowsize used for initial coverage smoothing of the bam file with a moving average method. Setting it too large might disguise CNV effects. (default: 10000) (int)

        - shiftsize: The shiftsize used for the moving average method of the initial coverage smoothing procedure. MUST be smaller than windowsize. (default: 3000) (int)

        - min_noise: The minimum frequency of non-reference or reference bases detected for a position to be considered for LOH detection. Setting it too small will result in poor noise filtering, setting it too large will result in a decreased number of measurement points. (default: 0.1) (float in range(0,1))

        - base_quality_limit: The base quality limit used by samtools in order to decide if a base should be included in the pileup file. (default: 0) (int)

        - print_every_nth: Even though LOH detection is limited to the positions with a noise level larger that min_noise, ploidy estimation is based on all the genomic positions meeting the above set criteria. By setting the attribute print_every_nth, the number of positions used can be controlled. Setting it large will result in overlooking ploidy variations in shorter genomic ranges, while setting it too small can cause an increase in both memory usage and computation time. Decrease only if a relatively short genome is analysed. (default: 100) (int)

        - windowsize_PE: The windowsize used for actual ploidy estimation after the initial coverage smoothing. (default: 1000000) (int)

        - shiftsize_PE: The shiftsize used for actual ploidy estimation after the initial coverage smoothing. MUST be smaller than windowsize_PE. (default: 50000) (int)

        - cov_max: The maximum coverage in a genomic position that can be considered for ploidy estimation. Alignment errors might cause certain genomic positions to have an enormous coverage. These outliers are ignored, when cov_max is set. The value must be set in agreement with the average sequencing depth. Using a low value for a deeply sequenced sample can result in a decreased number of positions to be analysed. (default: 200) (int)

        - cov_min: The minimum coverage in a genomic position that can be considered for ploidy estimation. Sequencing noise might cause certain genomic positions to have very low coverage, frequently merely from misaligned reads. These outliers are ignored, when cov_min is set. The value must be set in agreement with the average sequencing depth. Using a high value for a shallowly sequenced sample can result in a decreased number of positions to be analysed. (default: 5) (int)

        - hc_percentile: The haploid coverage of the sequenced sample is estimated multiple times by fitting a mixture model to the coverage distribution of the sample. The actual value is chosen as a statistical measure of these multiple results, set by the value of hc_percentile. For example, setting hc_percentile to 50 results in using the median of the results. For more details on the suggested values, see Pipek et al. 2018. (default: 75) (int)

        - compare_to_bed: The path to a bed file to compare ploidy estimation results to. (default: None) (str)

        - samtools_flags: The samtools flags to be used for pileup file generation. (default: " -B -d 1000 ") (str)

        - user_defined_hapcov: During ploidy estimation, the haploid coverage is estimated from the coverage distribution of the sample. In some cases, the estimation might not find the real value of the haploid coverage. In these situations, supplying an estimate of the haploid coverage manually might improve the overall ploidy estimation results. If you have a generally diploid genome, using the half of the average coverage can be a good starting point. If user_defined_hapcov is set, hc_percentile is ignored. (default: None) (float)

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

            :returns: (chrom, lens):

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
            if len(chrom_list)>0:
                return chrom_list, [newlens[newchroms.index(c)] for c in chrom_list]
            else:
                return sorted(newchroms), [newlens[newchroms.index(c)] for c in sorted(newchroms)]

        defaults = dict()
        defaults['n_min_block'] = 200
        defaults['n_conc_blocks'] = 4
        defaults['chromosomes'] = None
        defaults['windowsize'] = 10000
        defaults['shiftsize'] = 3000
        defaults['min_noise'] = 0.1
        defaults['base_quality_limit'] = 0
        defaults['windowsize_PE'] = 1000000
        defaults['shiftsize_PE'] = 50000
        defaults['cov_max'] = 200
        defaults['cov_min'] = 5
        defaults['hc_percentile'] = 75
        defaults['print_every_nth'] = 100
        defaults['compare_to_bed'] = 'None'
        defaults['bedfile'] = None
        defaults['samtools_flags'] = ' -B -d ' + str(SAMTOOLS_MAX_DEPTH) + ' '
        defaults['samtools_fullpath'] = 'samtools'
        defaults['user_defined_hapcov'] = None
        defaults['ownbed_filepath'] = None
        defaults['bed_dataframe'] = None
        defaults['coverage_sample'] = None
        defaults['distribution_dict'] = None

        for p in list_of_params:
            if p in ['output_dir', 'input_dir', 'bam_filename']:
                if not hasattr(self, p):
                    raise ValueError('Error: Value of "' + p + '" is not set, cannot proceed.')
                elif (p == 'input_dir' or p == 'bam_filename'):
                    if not os.path.isdir(self.input_dir):
                        raise ValueError(
                            'Error: Input directory "' + self.input_dir + '" does not exist, cannot proceed.')
                    if not os.path.isfile(self.input_dir + '/' + self.bam_filename):
                        raise ValueError(
                            'Error: Bam file "' + self.input_dir + '/' + self.bam_filename + '" does not exist, '
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
            elif p == 'chromosomes' and (not hasattr(self, 'chromosomes') or not self.chromosomes):
                if not os.path.isfile(self.ref_fasta):
                    raise ValueError(
                        'Error: Reference genome file "' + self.ref_fasta + '" does not exist, cannot proceed.')
                if not os.path.isfile(self.ref_fasta + '.fai'):
                    error_msg = 'Error: No faidx file found for reference genome file "' + self.ref_fasta + '", cannot proceed.'
                    error_msg += '\n'
                    error_msg += 'Use the samtools command: samtools faidx [ref.fasta]'
                    raise ValueError(error_msg)
                c, l = get_default_chroms_with_len(self.ref_fasta, [])
                setattr(self, 'chromosomes', c)
                setattr(self, 'chrom_length', l)
                setattr(self, 'genome_length', sum(l))
            elif p == 'chrom_length' and not hasattr(self, 'chrom_length'):
                if not os.path.isfile(self.ref_fasta):
                    raise ValueError(
                        'Error: Reference genome file "' + self.ref_fasta + '" does not exist, cannot proceed.')
                if not os.path.isfile(self.ref_fasta + '.fai'):
                    error_msg = 'Error: No faidx file found for reference genome file "' + self.ref_fasta + '", cannot proceed.'
                    error_msg += '\n'
                    error_msg += 'Use the samtools command: samtools faidx [ref.fasta]'
                    raise ValueError(error_msg)
                c, l = get_default_chroms_with_len(self.ref_fasta, [])
                setattr(self, 'chromosomes', c)
                setattr(self, 'chrom_length', l)
                setattr(self, 'genome_length', sum(l))
            elif p == 'genome_length':
                setattr(self, p, sum(self.chrom_length))
            elif p == 'estimated_hapcov' and not hasattr(self, 'estimated_hapcov'):
                raise ValueError('Error: Value of "' + p + '" is not set, cannot proceed.')
            elif not hasattr(self, p):
                setattr(self, p, defaults[p])

        if hasattr(self, 'ownbed_filepath') and self.ownbed_filepath is None:
            self.ownbed_filepath = self.output_dir+'/'+self.bam_filename.split('.bam')[0] + '_ploidy.bed'


    def get_coverage_distribution(self, **kwargs):
        """

        Sets the coverage_sample attribute of the PloidyEstimation object to the coverage distribution obtained from the temporary files created by __PE_prepare_temp_files().
        Positions are filtered according to the attributes of the PloidyEstimation object. The number of positions in the final sample is decreased to 2000 for faster inference.

        :param kwargs: keyword arguments for PloidyEstimator object

        :returns: A 2000-element sample of the coverage distribution.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['chromosomes', 'output_dir', 'cov_max', 'cov_min'])

        covs_few = io.get_coverage_distribution(chromosomes=self.chromosomes,
                                                output_dir=self.output_dir,
                                                cov_max=self.cov_max,
                                                cov_min = self.cov_min)

        self.coverage_sample = covs_few
        return covs_few

    def estimate_hapcov_infmix(self, level=0, **kwargs):
        """

        Estimates the haploid coverage of the sample from the appropriate attributes of the PloidyEstimation object.
        If the user_defined_hapcov attribute is set manually, it sets the value of estimated_hapcov to that. Otherwise, a many-component (20)
        Gaussian mixture model is fitted to the coverage histogram of the sample 10 times. Each time, the haploid coverage is estimated from the
        center of the component with the maximal weight in the model. The final estimate of the haploid coverage is calculated as the qth
        percentile of the 10 measurements, with q = hc_percentile. Sets the "estimated_hapcov" attribute to the calculated haploid coverage and the "coverage_sample" attribute to a 2000-element sample of the coverage distribution.

        :param level: the level of indentation used in verbose output (default: 0) (int)
        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['user_defined_hapcov', 'cov_min', 'hc_percentile', 'chromosomes', 'output_dir', 'cov_max'])

        hc, cov_sample = bayesian.estimate_hapcov_infmix(level=level,
                                                         user_defined_hapcov=self.user_defined_hapcov,
                                                         cov_min = self.cov_min,
                                                         hc_percentile = self.hc_percentile,
                                                         chromosomes=self.chromosomes,
                                                         output_dir=self.output_dir,
                                                         cov_max=self.cov_max)

        self.estimated_hapcov = hc
        self.coverage_sample = cov_sample

    def fit_gaussians(self, level=0, **kwargs):
        """

        Fits a 7-component Gaussian mixture model to the coverage distribution of the sample, using the appropriate attributes of the
        PloidyEstimation object. The center of the first Gaussian is initialized from a narrow region around the value of the estimated_hapcov
        attribute. The centers of the other Gaussians are initialized in a region around the value of estimated_hapcov multiplied by consecutive
        whole numbers.

        The parameters of the fitted model (center, sigma and weight) for all seven Gaussians are both saved to the GaussDistParams.pkl file (in
        output_dir, for later reuse) and set as the value of the distribution_dict attribute.

        :param level: the level of indentation used in verbose output (default: 0) (int)
        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self, 'user_defined_hapcov') and self.user_defined_hapcov is not None:
            self.estimated_hapcov = self.user_defined_hapcov

        self.__check_params(['estimated_hapcov', 'output_dir', 'chromosomes', 'cov_max', 'cov_min'])


        prior_dict = bayesian.fit_gaussians(level=level,
                                            estimated_hapcov=self.estimated_hapcov,
                                            output_dir=self.output_dir,
                                            chromosomes=self.chromosomes,
                                            cov_max=self.cov_max,
                                            cov_min=self.cov_min,
                                            cov_sample=self.coverage_sample)
        self.distribution_dict = prior_dict

    def PE_on_chrom(self, chrom, **kwargs):
        """

        Runs the whole ploidy estimation pipeline on a given chromosome, using the appropriate attributes of the PloidyEstimation object by running PE_on_range() multiple times.
        Prints the results to the file: [self.output_dir]/PE_fullchrom_[chrom].txt.

        :param chrom: the name of the chromosome to analyse (str)
        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['output_dir', 'windowsize_PE', 'shiftsize_PE', 'distribution_dict'])

        bayesian.PE_on_chrom(chrom=chrom,
                             output_dir=self.output_dir,
                             windowsize_PE=self.windowsize_PE,
                             shiftsize_PE=self.shiftsize_PE,
                             distribution_dict=self.distribution_dict)

    def PE_on_whole_genome(self, **kwargs):
        """

        Runs the whole ploidy estimation pipeline on the whole genome by running PE_on_chrom() on all chromosomes.

        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['chromosomes'])

        for c in self.chromosomes:
            print(c, end=" ")
            sys.stdout.flush()
            self.PE_on_chrom(chrom=c)


    def plot_karyotype_summary(self, **kwargs):
        """

        Plots a simple karyotype summary for the whole genome. (Details coming soon.)

        :param kwargs: keyword arguments for PloidyEstimator object

        :returns: a matplotlib figure of the plot

        """

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.__check_params(['output_dir', 'chromosomes', 'ownbed_filepath', 'cov_min', 'cov_max', 'chrom_length'])

        if not hasattr(self, 'distribution_dict'):
            self.load_cov_distribution_parameters_from_file(filename=kwargs.get("distribution_file"))

        return plot.plot_karyotype_summary(haploid_coverage=self.distribution_dict['mu'][0],
                                           chromosomes=self.chromosomes,
                                           chrom_length=self.chrom_length,
                                           output_dir=self.output_dir,
                                           bed_filename=self.ownbed_filepath,
                                           bed_file_sep=kwargs.get('bed_file_sep', ','),
                                           binsize=kwargs.get('binsize', 1000000),
                                           overlap=kwargs.get('overlap', 50000),
                                           cov_min=self.cov_min,
                                           cov_max=self.cov_max,
                                           min_PL_length=kwargs.get('min_PL_length', 3000000),
                                           chroms_with_text=kwargs.get('chroms_with_text'))


    def get_bed_format_for_sample(self, **kwargs):
        """

        Creates bed file of constant ploidies for a given sample from a file of positional ploidy data. If the ownbed_filepath attribute of the
        PloidyEstimation object is set, saves the bedfile to the path specified there. Otherwise, saves it to the output_dir with the "_ploidy.bed" suffix.
        Also sets the bed_dataframe attribute to the pandas.Dataframe containing the bed file.

        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['chromosomes', 'chrom_length', 'output_dir', 'bam_filename', 'ownbed_filepath'])

        obf, df = format.get_bed_format_for_sample(chromosomes=self.chromosomes,
                                         chrom_length=self.chrom_length,
                                         output_dir=self.output_dir,
                                         bam_filename=self.bam_filename,
                                         ownbed_filepath=self.ownbed_filepath)

        self.ownbed_filepath = obf
        self.bed_dataframe = df

    def plot_karyotype_for_all_chroms(self, return_string=False, **kwargs):
        """

        Plots karyotype information (coverage, estimated ploidy, estimated LOH, reference base frequencies) about the sample for all analysed
        chromosomes.

        :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: True) (bool)
        :param kwargs: keyword arguments for PloidyEstimator object

        :returns: If the return_string value is True, a list of base64 encoded strings of the images. Otherwise, a list of matplotlib figures.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['chromosomes', 'output_dir'])

        return plot.plot_karyotype_for_all_chroms(chromosomes=self.chromosomes,
                                           output_dir=self.output_dir,
                                           return_string=return_string)

    def generate_HTML_report_for_ploidy_est(self, **kwargs):
        """

        Generates a HTML file with figures displaying the results of ploidy estimation and saves it to output_dir/PEreport.html.

        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['chromosomes', 'output_dir', 'min_noise'])

        plot.generate_HTML_report_for_ploidy_est(chromosomes=self.chromosomes,
                                                 output_dir=self.output_dir,
                                                 min_noise=self.min_noise)

    def load_cov_distribution_parameters_from_file(self, filename=None, **kwargs):
        """

        Loads the parameters of the seven fitted Gaussians to the coverage distribution of the sample from the specified filename (that was saved
        with pickle beforehand). If one such file is available, the computationally expensive ploidy estimation process can be skipped. The
        parameter values will be stored in the attribute "distribution_dict" as a dictionary.

        :param filename: The path to the file with the coverage distribution parameters. (default: [output_dir]/GaussDistParams.pkl) (str)
        :param kwargs: keyword arguments for PloidyEstimator object

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['output_dir'])

        d = io.load_cov_distribution_parameters_from_file(filename=filename, output_dir=self.output_dir)
        self.distribution_dict = d

    def load_bedfile_from_file(self, filename=None, **kwargs):
        """

        Loads the bedfile containing previous ploidy estimated for the given sample from the path specified in filename. The dataframe will be
        stored in the "bed_dataframe" attribute.

        :param filename: The path to the bedfile. (default: [output_dir]/[bam_filename]_ploidy.bed) (str)
        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['output_dir', 'bam_filename'])

        self.bed_dataframe = io.load_bedfile_from_file(filename=filename,
                                                       output_dir=self.output_dir,
                                                       bam_filename=self.bam_filename)

    def run_ploidy_estimation(self, **kwargs):
        """

        Runs the whole ploidy estimation pipeline on the PloidyEstimation object.

        :param kwargs: keyword arguments for PloidyEstimator object

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['bam_filename', 'ref_fasta', 'input_dir', 'output_dir', 'genome_length',
                             'n_min_block', 'n_conc_blocks', 'chromosomes', 'chrom_length', 'windowsize', 'shiftsize',
                             'min_noise', 'print_every_nth', 'base_quality_limit', 'samtools_fullpath', 'samtools_flags',
                             'bedfile', 'user_defined_hapcov'])

        # running ploidy estimation
        level = 0

        starting_time = datetime.now()
        print('\t' * level + time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime()) + ' - Ploidy estimation for file ' + self.bam_filename)
        print('\n')

        process.PE_prepare_temp_files(ref_fasta=self.ref_fasta,
                                        input_dir=self.input_dir,
                                        bam_filename=self.bam_filename,
                                        output_dir=self.output_dir,
                                        genome_length=self.genome_length,
                                        n_min_block=self.n_min_block,
                                        n_conc_blocks=self.n_conc_blocks,
                                        chromosomes=self.chromosomes,
                                        chrom_length=self.chrom_length,
                                        windowsize=self.windowsize,
                                        shiftsize=self.shiftsize,
                                        min_noise=self.min_noise,
                                        print_every_nth=self.print_every_nth,
                                        base_quality_limit=self.base_quality_limit,
                                        samtools_fullpath=self.samtools_fullpath,
                                        samtools_flags=self.samtools_flags,
                                        bedfile = self.bedfile,
                                        level=level+1)

        if (self.user_defined_hapcov):
            print('\t' * (level + 1) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime()) + ' - Collecting data for coverage distribution, using user-defined haploid coverage (' + str(
                self.user_defined_hapcov) + ')...\n')
            self.estimate_hapcov_infmix(level=level + 2)
        else:
            print('\t' * (level + 1) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime()) + ' - Estimating haploid coverage by fitting an infinite mixture model to the coverage distribution...\n')
            self.estimate_hapcov_infmix(level=level + 2)
            print('\t' * (level + 2) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime()) + ' - Raw estimate for the haploid coverage: ' + str(
                self.estimated_hapcov) + '\n')

        print('\t' * (level + 1) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime()) + ' - Fitting equidistant Gaussians to the coverage distribution using the raw haploid coverage as prior...\n')
        self.fit_gaussians(level=level + 2)
        print('\t' * (level + 2) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime()) + ' - Parameters of the distribution are saved to: ' + self.output_dir + '/GaussDistParams.pkl' + '\n')
        print('\t' * (level + 2) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime()) + ' - Final estimate for the haploid coverage: ' + str(
            self.distribution_dict['mu'][0]) + '\n')
        print('\t' * (level + 1) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime()) + ' - Estimating local ploidy using the previously determined Gaussians as priors on chromosomes: ', end=' ')
        for c in self.chromosomes:
            print(c, end=" ")
            sys.stdout.flush()
            self.PE_on_chrom(chrom=c)

        print('\n\n' + '\t' * (level + 1) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                          time.localtime()) + ' - Generating final bed file... \n')
        self.get_bed_format_for_sample()

        print('\n' + '\t' * (level + 1) + time.strftime("%Y-%m-%d %H:%M:%S",
                                                        time.localtime()) + ' - Generating HTML report... \n')
        self.generate_HTML_report_for_ploidy_est()

        finish_time = datetime.now()
        total_time = finish_time - starting_time
        total_time_h = int(total_time.seconds / 3600)
        total_time_m = int((total_time.seconds % 3600) / 60)
        total_time_s = (total_time.seconds % 3600) % 60
        print('\n' + '\t' * level + time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime()) + ' - Ploidy estimation finished. (' + str(
            total_time.days) + ' day(s), ' + str(total_time_h) + ' hour(s), ' + str(total_time_m) + ' min(s), ' + str(
            total_time_s) + ' sec(s).)')



    def compare_with_other(self, other, minLen=2000, minQual=0.1):
        """

        Compare ploidy estimation results with another PloidyEstimation object or a bed file.

        :param other: The other PloidyEstimation object or the path to the other bedfile. (isomut2py.PloidyEstimation or str)
        :param minLen: The minimum length of a region to be considered different from the other object or file. (int)
        :param minQual: The minimum quality of a region to be considered different from the other object or file. (float)

        """
        if other.__class__ == self.__class__:
            compare.compare_with_other_PloidyEstimator(self, other, minLen, minQual)
        elif other.__class__ == str or other.__class__ == pd.core.frame.DataFrame:
            compare.compare_with_bed(bed_dataframe=self.bed_dataframe, other=other, minLen=minLen)
        else:
            error_msg = 'Error: For ploidy estimation comparison, the type of the other object must be either str or PloidyEstimation.'
            raise ValueError(error_msg)

    def plot_coverage_distribution(self, **kwargs):
        """

        Plot the coverage distribution of the sample.

        :param kwargs: keyword arguments for PloidyEstimator object

        :returns: a matplotlib figure of the coverage distribution

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.__check_params(['coverage_sample', 'distribution_dict', 'chromosomes', 'output_dir',
                             'cov_max', 'cov_min'])

        return plot.plot_coverage_distribution(cov_sample = self.coverage_sample,
                                               chromosomes=self.chromosomes,
                                               output_dir = self.output_dir,
                                               cov_max = self.cov_max,
                                               cov_min = self.cov_min,
                                               distribution_dict=self.distribution_dict)