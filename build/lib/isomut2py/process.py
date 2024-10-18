try:
    from isomut2py import HOME
    from isomut2py import SAMTOOLS_MAX_DEPTH

    import sys as __sys
    import time as __time
    import numpy as __np
    import subprocess as __subprocess
    import os as __os
    import multiprocessing as __multiprocessing
    import glob as __glob
except ImportError:
    print('ImportError in isomut2py.processing, ploidy estimation and mutation calling functions might not work.')


def __define_parallel_blocks(genome_length, n_min_block, chromosomes, chrom_length, level=0):
    """

    Calculate blocks of parallelization on the reference genome.

    The number of returned blocks, are higher than min_block_no.
    No block overlap chromosomes, because samtools does not accept region calls overlapping chromosomes.

    :param genome_length: the total length of the genome in basepairs (int)
    :param n_min_block: The approximate number of blocks to partition the analysed genome to for parallel computing. The actual number might be slightly larger that this. (int)
    :param chromosomes: list of chromosomes to analyse (list of str)
    :param chrom_length: list of chromosome lengths (list of int)
    :param level: the level of indentation used in verbose output (default: 0) (int)

    :returns: list of tuples, each tuple is a block: [(chrom, posStart, posEnd), ...]

    """

    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Defining parallel blocks ...')
    __sys.stdout.flush()  # jupyter notebook needs this to have constant feedback

    # set maximum block size
    BLOCKSIZE = int(__np.floor(genome_length / n_min_block))

    # calculate blocks
    blocks = []
    # loop over chroms
    for chrom, leng in zip(chromosomes, chrom_length):
        pointer = 0
        # until chrom is chopped into pieces
        while (pointer < leng):
            block_size = min(BLOCKSIZE, leng - pointer)
            blocks.append([chrom, pointer, pointer + block_size])
            pointer += block_size

    # return chr, posmin, posmax of blocks
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Done\n')
    __sys.stdout.flush()
    return blocks

def __temp_file_from_block(chrom,
                           from_pos,
                           to_pos,
                           ref_fasta,
                           input_dir,
                           bam_filename,
                           output_dir,
                           windowsize=10000,
                           shiftsize=3000,
                           min_noise=0.1,
                           print_every_nth=1000,
                           base_quality_limit=0,
                           samtools_fullpath='samtools',
                           samtools_flags=None,
                           bedfile = None):
    """

    Run the samtools mpileup command and the PEprep C application in the system shell.

    Creates a temporary file for ploidy estimation by filtering positions based on their reference nucleotide frequency.

    One run is performed on a section of a chromosome.
    Specific parameters are set according to the PloidyEstimation object attributes.
    Automatically logs samtools stderr output to output_dir/samtools.log where output_dir is the output_dir attribute of the PloidyEstimation object.
    Results are saved to a file which name is created from the blocks parameters. (chrom, posStart, posEnd)

    :param chrom: block chromosome (str)
    :param from_pos: block posStart (int)
    :param to_pos:  block posEnd (int)
    :param ref_fasta: path to the reference genome fasta file (str)
    :param input_dir: path to the folder where bam files are located (str)
    :param bam_filename: list of bam filenames to analyse (list of str)
    :param output_dir: path to the directory where results should be saved (str)
    :param windowsize: windowsize for initial coverage smoothing with a moving average method (default: 10000) (int)
    :param shiftsize: shiftsize for initial coverage smoothing with a moving average method (default: 3000) (int)
    :param min_noise: The minimum frequency of non-reference or reference bases detected for a position to be considered for LOH detection. Setting it too small will result in poor noise filtering, setting it too large will result in a decreased number of measurement points. (default: 0.1) (float in range(0,1))
    :param print_every_nth: Even though LOH detection is limited to the positions with a noise level larger that min_noise, ploidy estimation is based on all the genomic positions meeting the above set criteria. By setting the attribute print_every_nth, the number of positions used can be controlled. Setting it large will result in overlooking ploidy variations in shorter genomic ranges, while setting it too small can cause an increase in both memory usage and computation time. Decrease only if a relatively short genome is analysed. (default: 100) (int)
    :param base_quality_limit: The base quality limit used by samtools in order to decide if a base should be included in the pileup file. (default: 0) (int)
    :param samtools_fullpath: The path to samtools on the computer. (default: "samtools") (str)
    :param samtools_flags: The samtools flags to be used for pileup file generation. (default: " -B -d 1000 ") (str)
    :param bedfile: path to the the bedfile to limit ploidy estimation to a specific region of the genome (default: None) (str)

    :returns: status of the run (0 if successful, otherwise 1)

    """

    # build the command
    cmd = ' ' + samtools_fullpath + '  mpileup'
    if samtools_flags != None:
        cmd += ' ' + samtools_flags
    cmd += ' -f ' + ref_fasta
    cmd += ' -r ' + chrom + ':' + str(from_pos) + '-' + str(to_pos) + ' '
    if (bedfile != None):
        cmd += ' -l ' + bedfile + ' '
    cmd += input_dir + '/' + bam_filename + ' '
    cmd += ' 2>> ' + output_dir + '/samtools.log | '+HOME+'/C/PEprep '
    cmd += ' '.join(map(str, [windowsize, shiftsize, min_noise, print_every_nth,
                              base_quality_limit, bam_filename])) + ' '
    cmd += ' > ' + output_dir + '/PEtmp_blockfile_' + chrom + '_' + str(from_pos) + '_' + str(to_pos) + '.csv'

    return __subprocess.check_call(cmd, shell=True)

def PE_prepare_temp_files(ref_fasta,
                            input_dir,
                            bam_filename,
                            output_dir,
                            genome_length,
                            n_min_block,
                            n_conc_blocks,
                            chromosomes,
                            chrom_length,
                            windowsize=10000,
                            shiftsize=3000,
                            min_noise=0.1,
                            print_every_nth=1000,
                            base_quality_limit=0,
                            samtools_fullpath='samtools',
                            samtools_flags=None,
                            bedfile = None,
                            level=0):
    """

    Prepares temporary files for ploidy estimation, by averaging coverage in moving windows for the whole genome and collecting positions with
    reference allele frequencies in the [min_noise, 1-min_noise] range.

    :param ref_fasta: path to the reference genome fasta file (str)
    :param input_dir: path to the folder where bam files are located (str)
    :param bam_filename: list of bam filenames to analyse (list of str)
    :param output_dir: path to the directory where results should be saved (str)
    :param genome_length: the total length of the genome in basepairs (int)
    :param n_min_block: The approximate number of blocks to partition the analysed genome to for parallel computing. The actual number might be slightly larger that this. (default: 200) (int)
    :param n_conc_blocks: The number of blocks to process at the same time. (default: 4) (int)
    :param chromosomes: list of chromosomes in the genome (list of str)
    :param chrom_length: list of chromosome lengths in the genome (list of int)
    :param windowsize: windowsize for initial coverage smoothing with a moving average method (default: 10000) (int)
    :param shiftsize: shiftsize for initial coverage smoothing with a moving average method (default: 3000) (int)
    :param min_noise: The minimum frequency of non-reference or reference bases detected for a position to be considered for LOH detection. Setting it too small will result in poor noise filtering, setting it too large will result in a decreased number of measurement points. (default: 0.1) (float in range(0,1))
    :param print_every_nth: Even though LOH detection is limited to the positions with a noise level larger that min_noise, ploidy estimation is based on all the genomic positions meeting the above set criteria. By setting the attribute print_every_nth, the number of positions used can be controlled. Setting it large will result in overlooking ploidy variations in shorter genomic ranges, while setting it too small can cause an increase in both memory usage and computation time. Decrease only if a relatively short genome is analysed. (default: 100) (int)
    :param base_quality_limit: The base quality limit used by samtools in order to decide if a base should be included in the pileup file. (default: 0) (int)
    :param samtools_fullpath: The path to samtools on the computer. (default: "samtools") (str)
    :param samtools_flags: The samtools flags to be used for pileup file generation. (default: " -B -d 1000 ") (str)
    :param bedfile: path to the the bedfile to limit ploidy estimation to a specific region of the genome (default: None) (str)
    :param level: the level of indentation used in verbose output (default: 0) (int)

    """
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - Preparing for parallelization...')

    # define blocks and create args
    blocks = __define_parallel_blocks(genome_length=genome_length,
                                      n_min_block=n_min_block,
                                      chromosomes=chromosomes,
                                      chrom_length=chrom_length,
                                      level=level+1)

    args = []
    for block in blocks:
        args.append([block[0], block[1], block[2], ref_fasta, input_dir, bam_filename, output_dir, windowsize,
                     shiftsize, min_noise, print_every_nth, base_quality_limit, samtools_fullpath, samtools_flags,
                     bedfile])

    # creating output directory (and cleaning if it already exists)
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - (All output files will be written to ' + output_dir + ')\n')
    if (__os.path.isdir(output_dir)):
        cmd = 'rm -r ' + output_dir
        __subprocess.call(cmd, shell=True)
    cmd = 'mkdir ' + output_dir
    __subprocess.call(cmd, shell=True)

    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - Generating temporary files, number of blocks to run: ' + str(
        len(args)))
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Currently running: ', end=' ')
    # start first n concurrent block
    procs = []
    for i in range(n_conc_blocks):
        procs.append(__multiprocessing.Process(target=__temp_file_from_block, args=args[len(procs)]))
        procs[-1].start()
        print(str(len(procs)), end=" ")
        __sys.stdout.flush()

    # when one finished restart the next
    while (len(procs) != len(args)):
        if sum([procs[i].is_alive() for i in range(len(procs))]) < n_conc_blocks and len(procs) != len(args):
            procs.append(__multiprocessing.Process(target=__temp_file_from_block, args=args[len(procs)]))
            procs[-1].start()
            print(str(len(procs)), end=" ")
            __sys.stdout.flush()

        # for i in range(1, n_conc_blocks + 1):
        #     # one finished start another one
        #     if (procs[-i].is_alive() == False and len(procs) != len(args)):
        #         procs.append(__multiprocessing.Process(target=__temp_file_from_block, args=args[len(procs)]))
        #         procs[-1].start()
        #         print(str(len(procs)), end=" ")
        #         __sys.stdout.flush()
        __time.sleep(0.1)

    # wait now only the last ones running
    finished = False
    while (not finished):
        finished = True
        for proc in procs:
            if (proc.is_alive() == True):
                finished = False
        __time.sleep(0.1)

    print('\n')
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - Temporary files created, merging, cleaning up...\n')

    # collecting temp position files to single files for each chromosome
    for c in chromosomes:
        cmd = 'cat ' + output_dir + '/PEtmp_blockfile_' + c + '_* > ' + output_dir + '/PEtmp_fullchrom_' + c + '.txt'
        __subprocess.check_call(cmd, shell=True)
        cmd = 'rm ' + output_dir + '/PEtmp_blockfile_' + c + '_*'
        __subprocess.check_call(cmd, shell=True)


def __run_isomut2_on_block(chrom,
                           from_pos,
                           to_pos,
                           ref_fasta,
                           bam_filename,
                           input_dir,
                           output_dir,
                           samtools_fullpath='samtools',
                           samtools_flags = None,
                           bedfile = None,
                           min_sample_freq = 0.21,
                           min_other_ref_freq = 0.95,
                           cov_limit = 5,
                           base_quality_limit = 30,
                           min_gap_dist_snv = 0,
                           min_gap_dist_indel = 20,
                           constant_ploidy = 2,
                           ploidy_info_filepath = 'no_ploidy_info',
                           unique_mutations_only = True,
                           print_shared_by_all = False):
    """

    Run the samtools mpileup command and the isomut2 C application in the system shell.

    Creates a temporary file of potential mutations by filtering positions based on the attributes of the MutationCaller object.

    One run is performed on a section of a chromosome.
    Automatically logs samtools stderr output to output_dir/samtools.log where output_dir is the output_dir attribute of the MutationCaller object.
    Results are saved to a file which name is created from the blocks parameters. (chrom, posStart, posEnd)

    :param chrom: block chromosome (str)
    :param from_pos: block posStart (int)
    :param to_pos:  block posEnd (int)
    :param ref_fasta: path to the reference genome fasta file (str)
    :param bam_filename: A list of the name(s) of the bam file(s) of the sample(s). (Without path, eg. ["sample_1.bam", "sample_2.bam", "sample_3.bam", ...].) (list of str)
    :param input_dir: The path to the directory, where the bam file(s) of the sample(s) is/are located. (str)
    :param output_dir: The path to a directory that can be used for temporary files and output files. The user must have permission to write the directory. (str)
    :param samtools_fullpath: The path to samtools on the computer. (default: "samtools") (str)
    :param samtools_flags: The samtools flags to be used for pileup file generation. (default: " -B -d 1000 ") (str)
    :param bedfile: Path to a bedfile containing a list of genomic regions, if mutations are only sought in a limited part of the genome. (default: None) (str)
    :param min_sample_freq: The minimum frequency of the mutated base at a given position in the mutated sample(s). (default: 0.21) (float)
    :param min_other_ref_freq: The minimum frequency of the reference base at a given position in the non-mutated sample(s). (default: 0.95) (float)
    :param cov_limit: The minimum coverage at a given position in the mutated sample(s). (default: 5) (float)
    :param base_quality_limit: The base quality limit used by samtools in order to decide if a base should be included in the pileup file. (default: 30) (int)
    :param min_gap_dist_snv: Minimum genomic distance from an identified SNV for a position to be considered as a potential mutation. (default: 0) (int)
    :param min_gap_dist_indel: Minimum genomic distance from an identified indel for a position to be considered as a potential mutation. (default: 20) (int)
    :param constant_ploidy: The default ploidy to be used in regions not contained in ploidy_info_filepath. (default: 2) (int)
    :param ploidy_info_filepath: Path to the file containing ploidy information (see below for details) of the samples. (default: None) (str)
    :param unique_mutations_only: If True, only those mutations are sought that are unique to specific samples. Setting it to False greatly increases computation time, but allows for the detection of shared mutations and thus the analysis of phylogenetic connections between samples. If True, print_shared_by_all is ignored. (default: True) (boolean)
    :param print_shared_by_all: If False, mutations that are present in all analysed samples are not printed to the output files. This decreases both memory usage and computation time. (default: False) (boolean)

    :returns: status of the run (0 if successful, otherwise 1)

    """
    # build the command
    cmd = ' ' + samtools_fullpath + '  mpileup ' + samtools_flags
    cmd += ' -f ' + ref_fasta
    cmd += ' -r ' + chrom + ':' + str(from_pos) + '-' + str(to_pos) + ' '
    if bedfile is not None:
        cmd += ' -l ' + bedfile + ' '
    for bam_file in bam_filename:
        cmd += input_dir + bam_file + ' '
    cmd += ' 2>> ' + output_dir + '/samtools.log | '+HOME+'/C/isomut2 '
    cmd += ' '.join(map(str, [min_sample_freq, min_other_ref_freq, cov_limit,
                              base_quality_limit, min_gap_dist_snv, min_gap_dist_indel,
                              constant_ploidy, ploidy_info_filepath, unique_mutations_only, int(print_shared_by_all)]))
    for bam_file in bam_filename:
        cmd += ' ' + __os.path.basename(bam_file)
    cmd += ' > ' + output_dir + '/tmp_isomut2_' + chrom + '_' + str(from_pos) + '_' + str(
        to_pos) + '_mut.csv  '

    return __subprocess.check_call(cmd, shell=True)


def __run_isomut2_in_parallel(genome_length,
                              n_min_block,
                              n_conc_blocks,
                              chromosomes,
                              chrom_length,
                              ref_fasta,
                              bam_filename,
                              input_dir,
                              output_dir,
                              samtools_fullpath='samtools',
                              samtools_flags = None,
                              bedfile = None,
                              min_sample_freq = 0.21,
                              min_other_ref_freq = 0.95,
                              cov_limit = 5,
                              base_quality_limit = 30,
                              min_gap_dist_snv = 0,
                              min_gap_dist_indel = 20,
                              constant_ploidy = 2,
                              ploidy_info_filepath = 'no_ploidy_info',
                              unique_mutations_only = True,
                              print_shared_by_all = False,
                              level=0):
    """

    Generates temporary files of potential mutations, by running the __run_isomut2_on_block() function multiple times in parallel for the whole
    analysed part of the genome.

    :param genome_length: the total length of the genome in basepairs (int)
    :param n_min_block: The approximate number of blocks to partition the analysed genome to for parallel computing. The actual number might be slightly larger that this. (default: 200) (int)
    :param n_conc_blocks: The number of blocks to process at the same time. (default: 4) (int)
    :param chromosomes: list of chromosomes in the genome (list of str)
    :param chrom_length: list of chromosome lengths in the genome (list of int)
    :param ref_fasta: path to the reference genome fasta file (str)
    :param bam_filename: A list of the name(s) of the bam file(s) of the sample(s). (Without path, eg. ["sample_1.bam", "sample_2.bam", "sample_3.bam", ...].) (list of str)
    :param input_dir: The path to the directory, where the bam file(s) of the sample(s) is/are located. (str)
    :param output_dir: The path to a directory that can be used for temporary files and output files. The user must have permission to write the directory. (str)
    :param samtools_fullpath: The path to samtools on the computer. (default: "samtools") (str)
    :param samtools_flags: The samtools flags to be used for pileup file generation. (default: " -B -d 1000 ") (str)
    :param bedfile: Path to a bedfile containing a list of genomic regions, if mutations are only sought in a limited part of the genome. (default: None) (str)
    :param min_sample_freq: The minimum frequency of the mutated base at a given position in the mutated sample(s). (default: 0.21) (float)
    :param min_other_ref_freq: The minimum frequency of the reference base at a given position in the non-mutated sample(s). (default: 0.95) (float)
    :param cov_limit: The minimum coverage at a given position in the mutated sample(s). (default: 5) (float)
    :param base_quality_limit: The base quality limit used by samtools in order to decide if a base should be included in the pileup file. (default: 30) (int)
    :param min_gap_dist_snv: Minimum genomic distance from an identified SNV for a position to be considered as a potential mutation. (default: 0) (int)
    :param min_gap_dist_indel: Minimum genomic distance from an identified indel for a position to be considered as a potential mutation. (default: 20) (int)
    :param constant_ploidy: The default ploidy to be used in regions not contained in ploidy_info_filepath. (default: 2) (int)
    :param ploidy_info_filepath: Path to the file containing ploidy information (see below for details) of the samples. (default: None) (str)
    :param unique_mutations_only: If True, only those mutations are sought that are unique to specific samples. Setting it to False greatly increases computation time, but allows for the detection of shared mutations and thus the analysis of phylogenetic connections between samples. If True, print_shared_by_all is ignored. (default: True) (boolean)
    :param print_shared_by_all: If False, mutations that are present in all analysed samples are not printed to the output files. This decreases both memory usage and computation time. (default: False) (boolean)
    :param level: the level of indentation used in verbose output (default: 0) (int)

    """


    # define blocks and create args
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - Preparations for parallelization:\n')

    blocks=__define_parallel_blocks(genome_length=genome_length, n_min_block=n_min_block, chromosomes=chromosomes,
                                    chrom_length=chrom_length, level=level+1)
    args = []
    for block in blocks:
        args.append([block[0], block[1], block[2], ref_fasta, bam_filename, input_dir, output_dir, samtools_fullpath,
                     samtools_flags, bedfile, min_sample_freq, min_other_ref_freq, cov_limit, base_quality_limit,
                     min_gap_dist_snv, min_gap_dist_indel, constant_ploidy, ploidy_info_filepath, unique_mutations_only,
                     print_shared_by_all])

    # creating output directory (and cleaning if it already exists)
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - (All output files will be written to ' + output_dir + ')\n')
    if (__os.path.isdir(output_dir)):
        cmd = 'rm -r ' + output_dir
        __subprocess.call(cmd, shell=True)
    cmd = 'mkdir ' + output_dir
    __subprocess.call(cmd, shell=True)

    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S",
                                         __time.localtime()) + ' - Generating temporary files, total number of blocks to run: ' + str(
        len(args)) + '\n')
    print('\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Currently running:', end = ' ')

    # start first n concurrent block
    procs = []
    for i in range(n_conc_blocks):
        procs.append(__multiprocessing.Process(target=__run_isomut2_on_block, args=args[len(procs)]))
        procs[-1].start()
        print(str(len(procs)), end=" ")
        __sys.stdout.flush()

    # when one finished restart teh next
    while (len(procs) != len(args)):
        if sum([procs[i].is_alive() for i in range(len(procs))]) < n_conc_blocks and len(procs) != len(args):
            procs.append(__multiprocessing.Process(target=__run_isomut2_on_block, args=args[len(procs)]))
            procs[-1].start()
            print(str(len(procs)), end=" ")
            __sys.stdout.flush()

        # for i in range(1, n_conc_blocks + 1):
        #     # one finished start another one
        #     if (procs[-i].is_alive() == False and len(procs) != len(args)):
        #         procs.append(__multiprocessing.Process(target=__run_isomut2_on_block, args=args[len(procs)]))
        #         procs[-1].start()
        #         print(str(len(procs)), end=" ")
        __time.sleep(0.1)

    # wait now only the last ones running
    finished = False
    while (not finished):
        finished = True
        for proc in procs:
            if (proc.is_alive() == True):
                finished = False
        __time.sleep(0.1)

    print('\n' + '\t' * level + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Finished this round.\n')

def __collect_mutations(output_dir, mut_type, tmp_filename='tmp_isomut2_*_mut.csv'):
    """

    Collects the mutations from multiple temporary files.

    :param output_dir: path to the directory where temporary files are located (str)
    :param mut_type: The type of mutations to collect ("SNV" or "indel"). (str)
    :param tmp_filename: The regular expression for the name of the temporary files to scan through. (default: "tmp_isomut2_*_mut.csv") (str)

    """
    # print header
    header = "#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n"
    with open(output_dir + '/all_' + mut_type + 's.isomut2', 'w') as f:
        f.write(header)

    # copy all lines from temporary files (except the header) and sort them by chr, pos
    cmd = 'tail -q -n+2 ' + output_dir + '/' + tmp_filename + ' | '
    if (mut_type == 'indel'):
        cmd += 'awk \'$4=="INS" || $4=="DEL" {print}\' | '
    elif (mut_type == 'SNV'):
        cmd += 'awk \'$4=="SNV" {print}\' | '
    cmd += 'sort -n -k2,2 -k3,3 >> ' + output_dir + '/all_' + mut_type + 's.isomut2'
    __subprocess.check_call(cmd, shell=True)

def __prepare_for_2nd_run(mutcallObject):
    """

    If use_local_realignment = True, isomut2 is run a second time with different samtools settings. This function generates a bedfile from the
    initially identified potential mutations from the first round, which is used to decrease computation time for the second round. Original
    temporary files are saved for later use under a different name.

    :param mutcallObject: an isomut2py.MutationCaller object

    """
    # create bedfile for SNVs for post processing
    __subprocess.check_call(
        'tail -q -n+2 ' + mutcallObject.output_dir + '/tmp_isomut2_*_mut.csv |\
        awk \'$4=="SNV" {print}\' | \
        sort -n -k2,2 -k3,3 | cut -f 2,3 > ' + mutcallObject.output_dir + '/tmp_isomut2.bed', shell=True)

    # renaming original files so they are not overwritten
    for init_res in __glob.glob(mutcallObject.output_dir + '/tmp_isomut2_*_mut.csv'):
        new_file_name = mutcallObject.output_dir + '/' + init_res.split('.')[-2].split('/')[-1] + '_orig.csv'
        __subprocess.check_call('mv ' + init_res + ' ' + new_file_name, shell=True)

    # change parameter values for postprocessing, save originals
    mutcallObject.__base_quality_limit_orig = mutcallObject.base_quality_limit
    mutcallObject.base_quality_limit = 13
    mutcallObject.__min_other_ref_freq_orig = mutcallObject.min_other_ref_freq
    mutcallObject.min_other_ref_freq = 0
    mutcallObject.__samtools_flags_orig = mutcallObject.samtools_flags
    mutcallObject.samtools_flags = ' -d ' + str(SAMTOOLS_MAX_DEPTH) + ' '
    mutcallObject.__bedfile_orig = mutcallObject.bedfile
    mutcallObject.bedfile = mutcallObject.output_dir + '/tmp_isomut2.bed'

def __pair_muts_with_orig_cleanliness(output_dir):
    """

    If use_local_realignment = True, isomut2 is run a second time with different samtools settings. This function pairs the mutations not filtered
    out in the second round with their original cleanliness values, calculated in the first round.

    :param output_dir: path to the directory where temporary files are located (str)

    """

    header = "#sample_name\tchr\tpos\ttype\tscore\tref\tmut\tcov\tmut_freq\tcleanliness\tploidy\n"

    for new_file_name in __glob.glob(output_dir + '/tmp_isomut2_*_mut.csv'):
        with open(new_file_name) as f_new:
            # let's collect original cleanliness into a dict first
            cleanliness_dict = dict()
            old_file_name = output_dir + '/' + new_file_name.split('.')[-2].split('/')[-1] + '_orig.csv'
            with open(old_file_name) as f_old:
                f_old.readline()  # skip header
                for line in f_old:
                    line_list = line.strip().split('\t')
                    if len(line_list) == 11:
                        key = '_'.join(line_list[1:4] + line_list[5:7])
                        cleanliness_dict[key] = [line_list[9], line_list[0]]
            # now it's okay to remove the old file
            __subprocess.check_call('rm ' + old_file_name, shell=True)
            # now we move on to the new files
            f_new.readline()  # skip header
            final_file_name = output_dir + '/' + new_file_name.split('.')[-2].split('/')[-1] + '_final.csv'
            with open(final_file_name, 'w') as f_final:
                f_final.write(header)
                for line in f_new:
                    line_list = line.strip().split('\t')
                    key = '_'.join(line_list[1:4] + line_list[5:7])
                    if key in cleanliness_dict:
                        f_final.write('\t'.join(
                            [cleanliness_dict[key][1]] + line_list[1:9] + [cleanliness_dict[key][0]] + [
                                line_list[10]]) + '\n')

def __del_tmp_files(output_dir):
    """

    Deletes all temporary files from the output_dir of the MutationCaller object. (All files matching the regular expression "tmp_*".)

    :param output_dir: path to the directory where temporary files are located (str)

    """

    __subprocess.check_call('rm ' + output_dir + '/tmp_*', shell=True)

def __cleanup_after_2nd_run(mutcallObject, level=0):
    """

    If use_local_realignment = True, isomut2 is run a second time with different samtools settings. This function resets parameters after the
    second round to their original values, checks the number of discarded mutations in the second round, pairs non-filtered mutations with their
    original cleanliness values, collects identified mutations into a single file and deletes temporary files.

    :param mutcallObject: an isomut2py.MutationCaller object
    :param level: the level of indentation used in verbose output (default: 0) (int)

    """
    # reset parameter values
    mutcallObject.base_quality_limit = mutcallObject.__base_quality_limit_orig
    del mutcallObject.__base_quality_limit_orig
    mutcallObject.min_other_ref_freq = mutcallObject.__min_other_ref_freq_orig
    del mutcallObject.__min_other_ref_freq_orig
    mutcallObject.samtools_flags = mutcallObject.__samtools_flags_orig
    del mutcallObject.__samtools_flags_orig
    mutcallObject.bedfile = mutcallObject.__bedfile_orig
    del mutcallObject.__bedfile_orig

    print('\t' * (level) + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Finalizing output...')

    # check number of discarded mutations
    number_of_lines_old = __subprocess.check_output(
        'cat ' + mutcallObject.output_dir + '/tmp_isomut2_*_mut_orig.csv | grep -v "^#" | wc -l', shell=True)
    number_of_lines_old = int(number_of_lines_old.strip())
    number_of_lines_new = __subprocess.check_output(
        'cat ' + mutcallObject.output_dir + '/tmp_isomut2_*_mut.csv | grep -v "^#" | wc -l', shell=True)
    number_of_lines_new = int(number_of_lines_new.strip())
    print('\t' * (level + 1) + ' - Discarded mutations due to BAQ realignment: ' + str(
        round(1 - (number_of_lines_new / number_of_lines_old), 2))) + '% (' + str(
        number_of_lines_old - number_of_lines_new) + ' total mutations).'

    # pair mutations with their old cleanliness values
    __pair_muts_with_orig_cleanliness(mutcallObject.output_dir)

    # now we collect SNVs
    __collect_mutations(output_dir=mutcallObject.output_dir, mut_type='SNV', tmp_filename='tmp_isomut2_*_mut_final.csv')

    # deleting temporary files
    print('\t' * (level) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                           __time.localtime()) + ' - Cleaning up temporary files...')

    __del_tmp_files(mutcallObject.output_dir)

def single_run(mutcallObject,level=0):
    """

    Runs the mutation detection pipeline on the MutationCaller object when no local realignment is used during the process.

    :param mutcallObject: an isomut2py.MutationCaller object
    :param level: the level of indentation used in verbose output (default: 0) (int)

    """
    print('\n' + '\t' * (level + 1) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                                      __time.localtime()) + ' - Running IsoMut2 without local realignment: \n\n')
    __run_isomut2_in_parallel(genome_length=mutcallObject.genome_length,
                              n_min_block=mutcallObject.n_min_block,
                              n_conc_blocks=mutcallObject.n_conc_blocks,
                              chromosomes=mutcallObject.chromosomes,
                              chrom_length=mutcallObject.chrom_length,
                              ref_fasta=mutcallObject.ref_fasta,
                              bam_filename=mutcallObject.bam_filename,
                              input_dir=mutcallObject.input_dir,
                              output_dir=mutcallObject.output_dir,
                              samtools_fullpath=mutcallObject.samtools_fullpath,
                              samtools_flags = mutcallObject.samtools_flags,
                              bedfile = mutcallObject.bedfile,
                              min_sample_freq = mutcallObject.min_sample_freq,
                              min_other_ref_freq = mutcallObject.min_other_ref_freq,
                              cov_limit = mutcallObject.cov_limit,
                              base_quality_limit = mutcallObject.base_quality_limit,
                              min_gap_dist_snv = mutcallObject.min_gap_dist_snv,
                              min_gap_dist_indel = mutcallObject.min_gap_dist_indel,
                              constant_ploidy = mutcallObject.constant_ploidy,
                              ploidy_info_filepath = mutcallObject.ploidy_info_filepath,
                              unique_mutations_only = mutcallObject.unique_mutations_only,
                              print_shared_by_all = mutcallObject.print_shared_by_all,
                              level=level+2)

    print('\t' * (level + 2) + __time.strftime("%Y-%m-%d %H:%M:%S", __time.localtime()) + ' - Finalizing output...')

    __collect_mutations(output_dir=mutcallObject.output_dir, mut_type='indel')
    __collect_mutations(output_dir=mutcallObject.output_dir, mut_type='SNV')

    print('\t' * (level + 2) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                               __time.localtime()) + ' - Cleaning up temporary files...')
    #         subprocess.check_call('rm '+self.output_dir+'/tmp_isomut2_*_mut.csv',shell=True)
    __del_tmp_files(output_dir=mutcallObject.output_dir)


def double_run(mutcallObject, level=0):
    """

    Runs the mutation detection pipeline on the MutationCaller object when with local realignment.

    :param mutcallObject: an isomut2py.MutationCaller object
    :param level: the level of indentation used in verbose output (default: 0) (int)

    """

    # run 1st
    print('\n' + '\t' * (level + 1) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                                      __time.localtime()) + ' - Running IsoMut2: round 1/2 \n\n')
    __run_isomut2_in_parallel(genome_length=mutcallObject.genome_length,
                              n_min_block=mutcallObject.n_min_block,
                              n_conc_blocks=mutcallObject.n_conc_blocks,
                              chromosomes=mutcallObject.chromosomes,
                              chrom_length=mutcallObject.chrom_length,
                              ref_fasta=mutcallObject.ref_fasta,
                              bam_filename=mutcallObject.bam_filename,
                              input_dir=mutcallObject.input_dir,
                              output_dir=mutcallObject.output_dir,
                              samtools_fullpath=mutcallObject.samtools_fullpath,
                              samtools_flags = mutcallObject.samtools_flags,
                              bedfile = mutcallObject.bedfile,
                              min_sample_freq = mutcallObject.min_sample_freq,
                              min_other_ref_freq = mutcallObject.min_other_ref_freq,
                              cov_limit = mutcallObject.cov_limit,
                              base_quality_limit = mutcallObject.base_quality_limit,
                              min_gap_dist_snv = mutcallObject.min_gap_dist_snv,
                              min_gap_dist_indel = mutcallObject.min_gap_dist_indel,
                              constant_ploidy = mutcallObject.constant_ploidy,
                              ploidy_info_filepath = mutcallObject.ploidy_info_filepath,
                              unique_mutations_only = mutcallObject.unique_mutations_only,
                              print_shared_by_all = mutcallObject.print_shared_by_all,
                              level=level+2)

    # prepare for 2nd round
    print('\t' * (level + 2) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                               __time.localtime()) + ' - Preparing files for post-processing...')
    __collect_mutations(output_dir=mutcallObject.output_dir, mut_type='indel')
    __prepare_for_2nd_run(mutcallObject)

    # run 2nd
    print('\t' * (level + 1) + __time.strftime("%Y-%m-%d %H:%M:%S",
                                               __time.localtime()) + ' - Running IsoMut2: round 2/2\n\n')
    __run_isomut2_in_parallel(genome_length=mutcallObject.genome_length,
                              n_min_block=mutcallObject.n_min_block,
                              n_conc_blocks=mutcallObject.n_conc_blocks,
                              chromosomes=mutcallObject.chromosomes,
                              chrom_length=mutcallObject.chrom_length,
                              ref_fasta=mutcallObject.ref_fasta,
                              bam_filename=mutcallObject.bam_filename,
                              input_dir=mutcallObject.input_dir,
                              output_dir=mutcallObject.output_dir,
                              samtools_fullpath=mutcallObject.samtools_fullpath,
                              samtools_flags=mutcallObject.samtools_flags,
                              bedfile=mutcallObject.bedfile,
                              min_sample_freq=mutcallObject.min_sample_freq,
                              min_other_ref_freq=mutcallObject.min_other_ref_freq,
                              cov_limit=mutcallObject.cov_limit,
                              base_quality_limit=mutcallObject.base_quality_limit,
                              min_gap_dist_snv=mutcallObject.min_gap_dist_snv,
                              min_gap_dist_indel=mutcallObject.min_gap_dist_indel,
                              constant_ploidy=mutcallObject.constant_ploidy,
                              ploidy_info_filepath=mutcallObject.ploidy_info_filepath,
                              unique_mutations_only=mutcallObject.unique_mutations_only,
                              print_shared_by_all=mutcallObject.print_shared_by_all,
                              level=level + 2)

    # cleanup
    __cleanup_after_2nd_run(mutcallObject, level=level + 2)