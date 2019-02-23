try:
    from isomut2py import plot
    from isomut2py import io
    from isomut2py import HOME
    from isomut2py import SAMTOOLS_MAX_DEPTH

    import copy as __copy
    import numpy as __np
    import pandas as __pd
    import os as __os
    import subprocess as __subprocess
    import scipy as __scipy
    from Bio import SeqIO as __SeqIO
    from Bio.Seq import Seq as __Seq
    from Bio.SeqRecord import SeqRecord as __SeqRecord

except ImportError:
    print('ImportError in isomut2py.postprocess, some postprocessing functions might not work.')


def optimize_score(mutation_dataframe, control_samples,
                   FPs_per_genome, score0=0, unique_samples=None):
    """

    Optimizes score values for different mutation types (SNV, INS, DEL) and ploidies according to the list of control samples and the desired
    level of false positives in the genome. The results are stored in the score_lim_dict attribute of the MutationDetection object. If plot = True,
    plots ROC curves for all mutations types (SNV, INS, DEL) and all ploidies.

    :param mutation_dataframe: The dataframe containing the mutations. (pandas.DataFrame)
    :param control_samples: a subset of bam_filename (list of sample names) that should be considered as control samples. Control samples are defined as samples where no unique mutations are expected to be found. (list of str)
    :param FPs_per_genome: the largest number of false positives tolerated in a control sample (int)
    :param score0: Score optimization starts with score0. If a larger score value is likely to be optimal, setting score0 to a number larger than 0 can decrease computation time. (default: 0) (float)
    :param unique_samples: list of unique samples where at least one mutation is detected (default: None) (list of str)

    :returns: a dictionary containing the optimized score values for each ploidy

    """

    steps = 50

    if unique_samples is None:
        unique_samples = sorted(
            list(set([item for s in list(mutation_dataframe['sample_name'].unique()) for item in s.split(',')])))

    if sum([1 for s in control_samples if s not in unique_samples]) > 0:
        raise ValueError('List of "control_samples" is not a subset of "unique_samples".')

    total_num_of_FPs_per_genome = FPs_per_genome

    unique_ploidies = sorted(list(mutation_dataframe['ploidy'].unique()))

    mut_types_all = ['SNV', 'INS', 'DEL']

    score_lim_dict = {'SNV': [], 'INS': [], 'DEL': []}

    control_idx = []
    treated_idx = []
    for i in range(len(unique_samples)):
        if (unique_samples[i] in control_samples):
            control_idx.append(i)
        else:
            treated_idx.append(i)

    control_idx = __np.array(control_idx)
    treated_idx = __np.array(treated_idx)

    if total_num_of_FPs_per_genome is not None:
        FPs_per_ploidy = dict()
        for m in mut_types_all:
            FPs_per_ploidy[m] = dict()
            for pl in unique_ploidies:
                totalmuts_per_ploidy = \
                    mutation_dataframe[(mutation_dataframe['ploidy'] == pl) & (mutation_dataframe['type'] == m)].shape[
                        0]
                totalmuts = mutation_dataframe[(mutation_dataframe['type'] == m)].shape[0]
                if (totalmuts == 0):
                    FPs_per_ploidy[m][pl] = total_num_of_FPs_per_genome
                else:
                    FPs_per_ploidy[m][pl] = int(
                        round((float(totalmuts_per_ploidy) / totalmuts) * total_num_of_FPs_per_genome))

    for m in range(len(mut_types_all)):
        if (len(unique_ploidies) == 1):
            fp, tp = [0 for k in range(steps)], [0 for k in range(steps)]
            fp_real, tp_real = [0 for k in range(steps)], [0 for k in range(steps)]
            for score_lim, j in zip(
                    __np.linspace(score0,
                                  mutation_dataframe[mutation_dataframe['type'] == mut_types_all[m]]['score'].max(),
                                  steps),
                    range(steps)):
                muts = []
                for s in unique_samples:
                    muts.append(mutation_dataframe[(mutation_dataframe['ploidy'] == unique_ploidies[0]) &
                                                   (mutation_dataframe['sample_name'] == s) &
                                                   (mutation_dataframe['score'] > score_lim) &
                                                   (mutation_dataframe['type'] == mut_types_all[m])].shape[0])
                muts = __np.array(muts)
                fp[j], tp[j] = 1e-6 * __np.max(muts[control_idx]), 1e-6 * __np.mean(muts[treated_idx])
                fp_real[j], tp_real[j] = __np.max(muts[control_idx]), __np.mean(muts[treated_idx])

            if (total_num_of_FPs_per_genome is not None):
                fp_real = __np.array(fp_real)
                tp_real = __np.array(tp_real)
                if (len(tp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]]) > 0):
                    tps = tp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]][0]
                    fps = fp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]][0]
                    score_lim = \
                        __np.linspace(score0,
                                      mutation_dataframe[mutation_dataframe['type'] == mut_types_all[m]]['score'].max(),
                                      steps)[
                            fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[0]]][0]
                else:
                    score_lim = 10000
                score_lim_dict[mut_types_all[m]].append(score_lim)

        else:
            for i in range(len(unique_ploidies)):
                fp, tp = [0 for k in range(steps)], [0 for k in range(steps)]
                fp_real, tp_real = [0 for k in range(steps)], [0 for k in range(steps)]
                for score_lim, j in zip(
                        __np.linspace(score0,
                                      mutation_dataframe[mutation_dataframe['type'] == mut_types_all[m]]['score'].max(),
                                      steps),
                        range(steps)):
                    muts = []
                    for s in unique_samples:
                        muts.append(mutation_dataframe[(mutation_dataframe['ploidy'] == unique_ploidies[i]) &
                                                       (mutation_dataframe['sample_name'] == s) &
                                                       (mutation_dataframe['score'] > score_lim) &
                                                       (mutation_dataframe['type'] == mut_types_all[m])].shape[0])
                    muts = __np.array(muts)
                    fp[j], tp[j] = 1e-6 * __np.max(muts[control_idx]), 1e-6 * __np.mean(muts[treated_idx])
                    fp_real[j], tp_real[j] = __np.max(muts[control_idx]), __np.mean(muts[treated_idx])

                if (total_num_of_FPs_per_genome is not None):
                    fp_real = __np.array(fp_real)
                    tp_real = __np.array(tp_real)
                    if (len(tp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]]) > 0):
                        tps = tp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        fps = fp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        score_lim = \
                            __np.linspace(score0, mutation_dataframe[mutation_dataframe['type'] == mut_types_all[m]][
                                'score'].max(), steps)[
                                fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                    else:
                        score_lim = 10000
                    score_lim_dict[mut_types_all[m]].append(score_lim)

    return score_lim_dict


def optimize_results(sample_names,
                     control_samples,
                     FPs_per_genome,
                     plot_roc=False,
                     plot_tuning_curve=False,
                     filtered_results_file=None,
                     output_dir=None,
                     mutations_dataframe=None):
    """

    Optimizes the list of detected mutations according to the list of control samples and desired level of false positives set by the user.
    Filtered results will be loaded to the mutations attribute of the MutationDetection object.

    :param sample_names: The list of sample names included in the analysis. (list of str)
    :param control_samples: List of sample names that should be used as control samples in the sense, that no unique mutations are expected in them. (The sample names listed here must match a subset of the sample names listed in bam_filename.) (list of str)
    :param FPs_per_genome: The total number of false positives tolerated in a control sample. (int)
    :param plot_roc: If True, ROC curves will be plotted as a visual representation of the optimization process. (default: False) (boolean)
    :param plot_tuning_curve: If True, tuning curves displaying the number of mutations found in different samples with different score filters will be plotted as a visual representation of the optimization process. (default: False) (boolean)
    :param filtered_results_file: The path to the file where filtered results should be saved. (default: [output_dir]/filtered_results.csv) (str)
    :param output_dir: the path to the directory where raw mutation tables are located (default: None) (str)
    :param mutations_dataframe: the pandas.DataFrame where mutations are located (default: None) (pandas.DataFrame)

    :returns: (score_lim_dict, filtered_results)

        - score_lim_dict: a dictionary containing the optimized score values for each ploidy separately
        - filtered_results: a pandas.DataFrame containing the filtered mutations

    """

    if (len(sample_names) < 2):
        raise ValueError('Result optimization cannot be performed on less than 2 samples.')

    if sum([1 for s in control_samples if s not in sample_names]) > 0:
        raise ValueError('List of "control_samples" is not a subset of "sample_names".')

    if filtered_results_file is None:
        if output_dir is not None:
            filtered_results_file = output_dir + '/filtered_results.csv'
        else:
            filtered_results_file = 'filtered_results.csv'

    if not isinstance(mutations_dataframe, type(None)):
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            raise ValueError('Error: "mutations_dataframe" must be a pandas DataFrame.')
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations_dataframe" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
        df_somatic = mutations_dataframe[~mutations_dataframe['sample_name'].str.contains(',')]

    else:
        if (not __os.path.isfile(output_dir + '/all_SNVs.isomut2')):
            raise ValueError(
                'SNV results (' + output_dir + '/all_SNVs.isomut2) could not be found, results cannot be optimized.')
        if (not __os.path.isfile(output_dir + '/all_indels.isomut2')):
            raise ValueError(
                'Indel results (' + output_dir + '/all_indels.isomut2) could not be found, results cannot be optimized.')

        __subprocess.check_call(
            'cat ' + output_dir + '/all_SNVs.isomut2 | awk \'BEGIN{FS="\t"; OFS="\t";}{if($1 !~ /,/) print $0;}\' > ' + output_dir + '/unique_SNVs.isomut2',
            shell=True)
        __subprocess.check_call(
            'cat ' + output_dir + '/all_indels.isomut2 | awk \'BEGIN{FS="\t"; OFS="\t";}{if($1 !~ /,/) print $0;}\' > ' + output_dir + '/unique_indels.isomut2',
            shell=True)

        df_SNV = __pd.read_csv(output_dir + '/unique_SNVs.isomut2', header=0,
                               names=['sample_name', 'chr', 'pos', 'type', 'score',
                                      'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                               sep='\t',
                               low_memory=False)
        df_indel = __pd.read_csv(output_dir + '/unique_indels.isomut2', header=0,
                                 names=['sample_name', 'chr', 'pos', 'type', 'score',
                                        'ref', 'mut', 'cov', 'mut_freq', 'cleanliness', 'ploidy'],
                                 sep='\t',
                                 low_memory=False)

        __subprocess.check_call('rm ' + output_dir + '/unique_SNVs.isomut2', shell=True)
        __subprocess.check_call('rm ' + output_dir + '/unique_indels.isomut2', shell=True)

        df_somatic = __pd.concat([df_SNV, df_indel])

    df_somatic['ploidy'] = __pd.to_numeric(df_somatic['ploidy'], errors='ignore')

    unique_ploidies = sorted(list(df_somatic['ploidy'].unique()))

    if plot_tuning_curve:
        plot.__plot_tuning_curve(control_samples=control_samples,
                                 mutation_dataframe=df_somatic,
                                 return_string=False,
                                 unique_samples=sample_names)

    score_lim_dict, f = plot.__plot_roc(mutation_dataframe=df_somatic,
                                        control_samples=control_samples,
                                        FPs_per_genome=FPs_per_genome,
                                        plot_roc=plot_roc,
                                        unique_samples=sample_names)

    if mutations_dataframe is not None:
        def filter(row):
            if row['score'] < score_lim_dict[row['type']][unique_ploidies.index(row['ploidy'])]:
                return 'FILTER'
            else:
                return 'PASS'

        mutations_dataframe_filtered = __copy.deepcopy(mutations_dataframe)
        mutations_dataframe_filtered['FILTER'] = mutations_dataframe_filtered.apply(filter, axis=1)
        return score_lim_dict, mutations_dataframe_filtered[mutations_dataframe_filtered['FILTER'] != 'FILTER'].drop(
            'FILTER', axis=1)

    else:
        io.__print_filtered_results(output_dir=output_dir,
                                    filename=filtered_results_file,
                                    unique_ploidies=unique_ploidies,
                                    score_lim_dict=score_lim_dict,
                                    control_samples=control_samples,
                                    FPs_per_genome=FPs_per_genome)

        return score_lim_dict, io.load_mutations(output_dir=output_dir, filename=filtered_results_file)


def calculate_SNV_spectrum(ref_fasta, mutations_dataframe=None,
                           sample_names=None, unique_only=True, chromosomes=None,
                           output_dir=None, mutations_filename=None):
    """

    Calculates the triplet spectrum from a dataframe of relevant mutations, using the fasta file of the reference genome.

    :param ref_fasta: the path to the reference genome fasta file (str)
    :param mutations_dataframe: Dataframe containing the list of mutations to be considered. (default: None) (pandas.DataFrame)
    :param sample_names: the list of sample names included in the analysis (default: None) (list of str)
    :param unique_only: if True, only unique mutations are considered for the spectrum (default: True) (bool)
    :param chromosomes: list of chromosomes in the analysis (default: None) (list of str)
    :param output_dir: path to the directory where mutation tables are located (default: None) (str)
    :param mutations_filename: path to the mutation table(s) (default: None) (list of str)

    :returns: A dictionary containing 96-element vectors (the counts for each mutation type) as values and sample names as keys. (str: numpy.array)

    """

    if not isinstance(mutations_dataframe, type(None)):
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations_dataframe" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations_dataframe" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    mutations_dataframe = mutations_dataframe[mutations_dataframe['type'] == 'SNV']

    all_muttypes = ["ACAA", "TGTT", "ACCA", "GGTT", "ACGA", "CGTT", "ACTA", "AGTT", "CCAA", "TGGT",
                    "CCCA", "GGGT", "CCGA", "CGGT", "CCTA", "AGGT", "GCAA", "TGCT", "GCCA", "GGCT",
                    "GCGA", "CGCT", "GCTA", "AGCT", "TCAA", "TGAT", "TCCA", "GGAT", "TCGA", "CGAT",
                    "TCTA", "AGAT", "ACAG", "TGTC", "ACCG", "GGTC", "ACGG", "CGTC", "ACTG", "AGTC",
                    "CCAG", "TGGC", "CCCG", "GGGC", "CCGG", "CGGC", "CCTG", "AGGC", "GCAG", "TGCC",
                    "GCCG", "GGCC", "GCGG", "CGCC", "GCTG", "AGCC", "TCAG", "TGAC", "TCCG", "GGAC",
                    "TCGG", "CGAC", "TCTG", "AGAC", "ACAT", "TGTA", "ACCT", "GGTA", "ACGT", "CGTA",
                    "ACTT", "AGTA", "CCAT", "TGGA", "CCCT", "GGGA", "CCGT", "CGGA", "CCTT", "AGGA",
                    "GCAT", "TGCA", "GCCT", "GGCA", "GCGT", "CGCA", "GCTT", "AGCA", "TCAT", "TGAA",
                    "TCCT", "GGAA", "TCGT", "CGAA", "TCTT", "AGAA", "ATAA", "TATT", "ATCA", "GATT",
                    "ATGA", "CATT", "ATTA", "AATT", "CTAA", "TAGT", "CTCA", "GAGT", "CTGA", "CAGT",
                    "CTTA", "AAGT", "GTAA", "TACT", "GTCA", "GACT", "GTGA", "CACT", "GTTA", "AACT",
                    "TTAA", "TAAT", "TTCA", "GAAT", "TTGA", "CAAT", "TTTA", "AAAT", "ATAC", "TATG",
                    "ATCC", "GATG", "ATGC", "CATG", "ATTC", "AATG", "CTAC", "TAGG", "CTCC", "GAGG",
                    "CTGC", "CAGG", "CTTC", "AAGG", "GTAC", "TACG", "GTCC", "GACG", "GTGC", "CACG",
                    "GTTC", "AACG", "TTAC", "TAAG", "TTCC", "GAAG", "TTGC", "CAAG", "TTTC", "AAAG",
                    "ATAG", "TATC", "ATCG", "GATC", "ATGG", "CATC", "ATTG", "AATC", "CTAG", "TAGC",
                    "CTCG", "GAGC", "CTGG", "CAGC", "CTTG", "AAGC", "GTAG", "TACC", "GTCG", "GACC",
                    "GTGG", "CACC", "GTTG", "AACC", "TTAG", "TAAC", "TTCG", "GAAC", "TTGG", "CAAC",
                    "TTTG", "AAAC"]

    fasta_records = dict()
    for record in __SeqIO.parse(ref_fasta, "fasta"):
        if not chromosomes or record.id in chromosomes:
            fasta_records[record.id] = record.seq

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    SNVspectra = dict()

    for sample in sample_names:
        df = mutations_dataframe[mutations_dataframe['sample_name'].str.contains(sample)]
        if unique_only:
            df = df[~df['sample_name'].str.contains(',')]

        chrom_list = list(df['chr'].unique())
        mutationtype_array = []

        for chrom in chrom_list:
            search_pos_array = list(df[df['chr'] == chrom]['pos'])
            ref_array = list(df[df['chr'] == chrom]['ref'])
            alt_array = list(df[df['chr'] == chrom]['mut'])

            for i in range(len(search_pos_array)):
                search_pos = search_pos_array[i]
                ref = ref_array[i]

                neighborhood_size = 1
                start = search_pos - neighborhood_size - len(ref)
                end = search_pos + neighborhood_size
                search_seq = str(fasta_records[chrom][start:end])

                mutationtype_array.append(search_seq + alt_array[i])

        spectrum = __np.zeros(96)
        for mutation_type in mutationtype_array:
            if mutation_type in all_muttypes:
                spectrum[int(__np.floor(all_muttypes.index(mutation_type) / 2))] += 1
        SNVspectra[sample] = spectrum

    return SNVspectra


def __get_neighborhood(fasta_records, chrom, pos, mut, ref, muttype):
    """

    Gets the reference bases in a neighborhood of a given mutation.

    :param fasta_records: List of fasta records of the reference genome, loaded with SeqIO.parse. (Bio.SeqIO.SeqRecord)
    :param chrom: Chromosome of the genomic position of the mutation. (str)
    :param pos: Position of the genomic position of the mutation. (int)
    :param mut: The mutated base(s) at the mutation. (str)
    :param ref: The reference base(s) at the mutation. (str)
    :param muttype: The mutation type ("INS" or "DEL"). (str)

    :returns: tuple (search_seq_before, mut, search_seq_after)

        - search_seq_before: the sequence before the mutation (str)
        - mut: the mutated base(s) (str)
        - search_seq_after: the sequence after the mutation (str)

    """

    if muttype == 'DEL':
        neighborhood_size = len(ref) * 10
        start = pos - neighborhood_size
        end = pos + neighborhood_size + len(ref)
        search_seq_before = str(fasta_records[chrom][start:pos])
        search_seq_after = str(fasta_records[chrom][pos + len(ref):end])
        search_seq_ref = str(fasta_records[chrom][pos:pos + len(ref)])

        return search_seq_before, search_seq_ref, search_seq_after

    elif muttype == 'INS':
        neighborhood_size = len(mut) * 10
        start = pos - neighborhood_size
        end = pos + neighborhood_size
        search_seq_before = str(fasta_records[chrom][start:pos])
        search_seq_after = str(fasta_records[chrom][pos:end])

        return search_seq_before, mut, search_seq_after


def __find_number_of_repeats(seq_before, seq, seq_after):
    """

    Finds the number of repeats before or after a mutation.

    :param seq_before: the genomic sequence before the mutation (str)
    :param seq: the genomic sequence of the mutation (str)
    :param seq_after: the sequence after the mutation (str)

    :returns: number of repeats (int)

    """
    tmp_seq = seq_after
    k = 0
    while tmp_seq[:len(seq)] == seq:
        k += 1
        tmp_seq = tmp_seq[len(seq):]
    tmp_seq = seq_before[::-1]
    seq = seq[::-1]
    while tmp_seq[:len(seq)] == seq:
        k += 1
        tmp_seq = tmp_seq[len(seq):]
    return k + 1


def __find_microhomology(seq_before, seq, seq_after):
    """

    Finds the length of microhomology before or after a mutation.

    :param seq_before: the genomic sequence before the mutation (str)
    :param seq: the genomic sequence of the mutation (str)
    :param seq_after: the sequence after the mutation (str)

    :returns: number of repeats (int)

    """
    i = 1
    number_of_bases_found = 0
    while seq[:i] == seq_after[:i]:
        number_of_bases_found += 1
        i += 1
    if number_of_bases_found == 0:
        i = 1
        while seq[::-1][:i] == seq_before[::-1][:i]:
            number_of_bases_found += 1
            i += 1
    return number_of_bases_found


def __classify_indel(seq_before, seq, seq_after, muttype):
    """

    Classify an indel into the categories described here: https://www.biorxiv.org/content/early/2018/05/15/322859

    :param seq_before: the genomic sequence before the mutation (str)
    :param seq: the genomic sequence of the mutation (str)
    :param seq_after: the sequence after the mutation (str)
    :param muttype: the type of the mutation ("INS" or "DEL") (str)

    :returns: the ID of the mutation class the indel has been classified as (int)

    """
    if len(seq) == 1:
        if muttype == 'DEL':
            base_type = 0
        elif muttype == 'INS':
            base_type = 12
        k = __find_number_of_repeats(seq_before, seq, seq_after)
        if (k >= 6):
            k = 6
        t = -42
        if seq == 'T' or seq == 'A':
            t = -1
        elif seq == 'C' or seq == 'G':
            t = 0
        if t == -42:
            return -1
        mut_class = base_type + k * 2 + t
        return mut_class
    if len(seq) > 1:
        k = __find_number_of_repeats(seq_before, seq, seq_after)
        if muttype == 'DEL':
            if k == 1:
                m = __find_microhomology(seq_before, seq, seq_after)
            if k > 1 or m == 0:
                if k >= 6:
                    k = 6
                base_type = 24
                del_size = len(seq)
                if del_size >= 5:
                    del_size = 5
                mut_class = base_type + (del_size - 2) * 6 + k
                return mut_class

            elif k == 1 and m > 0:  # no repeat, but found microhomology
                if len(seq) == 2 and m == 1: mut_class = 73
                if len(seq) == 3 and m == 1: mut_class = 74
                if len(seq) == 3 and m == 2: mut_class = 75
                if len(seq) == 4 and m == 1: mut_class = 76
                if len(seq) == 4 and m == 2: mut_class = 77
                if len(seq) == 4 and m == 3: mut_class = 78
                if len(seq) >= 5 and m == 1: mut_class = 79
                if len(seq) >= 5 and m == 2: mut_class = 80
                if len(seq) >= 5 and m == 3: mut_class = 81
                if len(seq) >= 5 and m == 4: mut_class = 82
                if len(seq) >= 5 and m >= 5: mut_class = 83
                return mut_class
        elif muttype == 'INS':
            base_type = 48
            if k >= 6:
                k = 6
            ins_size = len(seq)
            if ins_size >= 5:
                ins_size = 5
            mut_class = base_type + (ins_size - 2) * 6 + k
            return mut_class


def calculate_indel_spectrum(ref_fasta, mutations_dataframe=None,
                             sample_names=None, unique_only=True, chromosomes=None,
                             output_dir=None, mutations_filename=None):

    """

    Calculates the indel spectrum from a dataframe of relevant mutations, using the fasta file of the reference genome.

    :param ref_fasta: the path to the reference genome fasta file (str)
    :param mutations_dataframe: Dataframe containing the list of mutations to be considered. (default: None) (pandas.DataFrame)
    :param sample_names: the list of sample names included in the analysis (default: None) (list of str)
    :param unique_only: if True, only unique mutations are considered for the spectrum (default: True) (bool)
    :param chromosomes: list of chromosomes in the analysis (default: None) (list of str)
    :param output_dir: path to the directory where mutation tables are located (default: None) (str)
    :param mutations_filename: path to the mutation table(s) (default: None) (list of str)

    :returns: A dictionary containing 83-element vectors (the counts for each mutation type) as values and sample names as keys. (str: numpy.array)

    """

    if not isinstance(mutations_dataframe, type(None)):
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations_dataframe" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations_dataframe" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    mutations_dataframe = mutations_dataframe[mutations_dataframe['type'].isin(['INS', 'DEL'])]

    fasta_records = dict()
    for record in __SeqIO.parse(ref_fasta, "fasta"):
        if not chromosomes or record.id in chromosomes:
            fasta_records[record.id] = record.seq

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    IDspectra = dict()

    for sample in sample_names:
        df = mutations_dataframe[mutations_dataframe['sample_name'].str.contains(sample)]
        if unique_only:
            df = df[~df['sample_name'].str.contains(',')]

        if df.shape[0] == 0:
            print('No indels were found for sample ' + sample)
            IDspectra[sample] = [0] * 83
        else:
            indel_spectrum = __np.zeros(83)
            for rID, r in df.iterrows():
                seq_before, seq, seq_after = __get_neighborhood(fasta_records, r['chr'], r['pos'], r['mut'],
                                                                r['ref'],
                                                                muttype=r['type'])
                idx = __classify_indel(seq_before, seq, seq_after, muttype=r['type'])
                if idx > 0:
                    indel_spectrum[idx - 1] += 1
            IDspectra[sample] = indel_spectrum

    return IDspectra


def calculate_DNV_matrix(mutations_dataframe=None,
                         sample_names=None, unique_only=True,
                         output_dir=None, mutations_filename=None):

    """

    Calculates the DNV matrix from a dataframe of relevant mutations.

    :param mutations_dataframe: Dataframe containing the list of mutations to be considered. (default: None) (pandas.DataFrame)
    :param sample_names: the list of sample names included in the analysis (default: None) (list of str)
    :param unique_only: if True, only unique mutations are considered for the spectrum (default: True) (bool)
    :param output_dir: path to the directory where mutation tables are located (default: None) (str)
    :param mutations_filename: path to the mutation table(s) (default: None) (list of str)

    :returns: A dictionary containing DNV matrices as values and sample names as keys. (str: numpy.array)

    """

    if not isinstance(mutations_dataframe, type(None)):
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations_dataframe" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations_dataframe" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    mutations_dataframe = mutations_dataframe[mutations_dataframe['type'] == 'SNV']

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    DNVmatrice = dict()

    base_changes = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G', 'A>C', 'A>G', 'A>T', 'G>A', 'G>C', 'G>T']
    base_pairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    for sample in sample_names:
        df = mutations_dataframe[mutations_dataframe['sample_name'].str.contains(sample)]
        if unique_only:
            df = df[~df['sample_name'].str.contains(',')]

        if df.shape[0] == 0:
            print('No DNVs were found for sample ' + sample)
            DNVmatrice[sample] = __np.zeros((len(base_changes), len(base_changes)))
        else:
            df = df.sort_values(by=['chr', 'pos'])
            df['next pos'] = df['pos'].shift(-1)
            df['next ref'] = df['ref'].shift(-1)
            df['next mut'] = df['mut'].shift(-1)
            DNVs = df[df['pos'] + 1 == df['next pos']]

            matrix = __np.zeros((len(base_changes), len(base_changes)))
            ref1 = list(DNVs['ref'])
            alt1 = list(DNVs['mut'])
            ref2 = list(DNVs['next ref'])
            alt2 = list(DNVs['next mut'])
            for i in range(len(ref1)):
                firstone = ref1[i] + '>' + alt1[i]
                nextone = ref2[i] + '>' + alt2[i]
                if len(base_changes) - base_changes.index(firstone) <= base_changes.index(nextone):
                    firstone = base_pairs[ref2[i]] + '>' + base_pairs[alt2[i]]
                    nextone = base_pairs[ref1[i]] + '>' + base_pairs[alt1[i]]
                matrix[base_changes.index(nextone), base_changes.index(firstone)] += 1
            DNVmatrice[sample] = matrix

    return DNVmatrice


def calculate_DNV_spectrum(mutations_dataframe=None,
                           sample_names=None, unique_only=True,
                           output_dir=None, mutations_filename=None):
    """

    Calculates the indel spectrum from a dataframe of relevant mutations.

    :param mutations_dataframe: Dataframe containing the list of mutations to be considered. (default: None) (pandas.DataFrame)
    :param sample_names: the list of sample names included in the analysis (default: None) (list of str)
    :param unique_only: if True, only unique mutations are considered for the spectrum (default: True) (bool)
    :param output_dir: path to the directory where mutation tables are located (default: None) (str)
    :param mutations_filename: path to the mutation table(s) (default: None) (list of str)

    :returns: A dictionary containing DNV spectra arrays (the counts for each mutation type) as values and sample names as keys. (str: numpy.array)

    """

    if not isinstance(mutations_dataframe, type(None)):
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations_dataframe" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations_dataframe" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    mutations_dataframe = mutations_dataframe[mutations_dataframe['type'] == 'SNV']

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    DNVspectra = dict()

    base_pairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    DBS_types = ["AC>CA", "AC>CG", "AC>CT", "AC>GA", "AC>GG", "AC>GT", "AC>TA", "AC>TG", "AC>TT", "AT>CA",
                 "AT>CC",
                 "AT>CG", "AT>GA", "AT>GC", "AT>TA", "CC>AA", "CC>AG", "CC>AT", "CC>GA", "CC>GG", "CC>GT",
                 "CC>TA",
                 "CC>TG", "CC>TT", "CG>AT", "CG>GC", "CG>GT", "CG>TA", "CG>TC", "CG>TT", "CT>AA", "CT>AC",
                 "CT>AG",
                 "CT>GA", "CT>GC", "CT>GG", "CT>TA", "CT>TC", "CT>TG", "GC>AA", "GC>AG", "GC>AT", "GC>CA",
                 "GC>CG",
                 "GC>TA", "TA>AT", "TA>CG", "TA>CT", "TA>GC", "TA>GG", "TA>GT", "TC>AA", "TC>AG", "TC>AT",
                 "TC>CA",
                 "TC>CG", "TC>CT", "TC>GA", "TC>GG", "TC>GT", "TG>AA", "TG>AC", "TG>AT", "TG>CA", "TG>CC",
                 "TG>CT",
                 "TG>GA", "TG>GC", "TG>GT", "TT>AA", "TT>AC", "TT>AG", "TT>CA", "TT>CC", "TT>CG", "TT>GA",
                 "TT>GC",
                 "TT>GG"]

    for sample in sample_names:
        df = mutations_dataframe[mutations_dataframe['sample_name'].str.contains(sample)]
        if unique_only:
            df = df[~df['sample_name'].str.contains(',')]

        if df.shape[0] == 0:
            print('No DNVs were found for sample ' + sample)
            DNVspectra[sample] = __np.zeros(len(DBS_types))
        else:
            df = df.sort_values(by=['chr', 'pos'])
            df['next pos'] = df['pos'].shift(-1)
            df['next ref'] = df['ref'].shift(-1)
            df['next mut'] = df['mut'].shift(-1)
            DNVs = df[df['pos'] + 1 == df['next pos']]

            spectrum = __np.zeros(len(DBS_types))
            ref1 = list(DNVs['ref'])
            alt1 = list(DNVs['mut'])
            ref2 = list(DNVs['next ref'])
            alt2 = list(DNVs['next mut'])
            for i in range(len(ref1)):
                firstone = ref1[i] + '>' + alt1[i]
                nextone = ref2[i] + '>' + alt2[i]
                if (firstone.split('>')[0] + nextone.split('>')[0] + '>' + firstone.split('>')[1] + nextone.split('>')[
                    1]
                        not in DBS_types):
                    firstone = base_pairs[ref2[i]] + '>' + base_pairs[alt2[i]]
                    nextone = base_pairs[ref1[i]] + '>' + base_pairs[alt1[i]]
                spectrum[DBS_types.index(
                    firstone.split('>')[0] + nextone.split('>')[0] + '>' + firstone.split('>')[1] + nextone.split('>')[
                        1])] += 1
            DNVspectra[sample] = spectrum

    return DNVspectra


def check_pileup(chrom_list, from_pos_list, to_pos_list, ref_fasta, input_dir, bam_filename,
                 output_dir='',
                 samtools_fullpath='samtools',
                 base_quality_limit=30,
                 samtools_flags=' -B -d ' + str(SAMTOOLS_MAX_DEPTH) + ' ',
                 print_original=True, filename=None):
    """

    Loads pileup information for a list of genomic regions.

    :param chrom_list: List of chromosomes for the regions. (list of str)
    :param from_pos_list: List of starting positions for the regions. (list of int)
    :param to_pos_list: List of ending positions for the regions. (list of int)
    :param ref_fasta: the path to the reference genome fasta file (str)
    :param bam_filename: the filenames for the bam files of the investigated samples (list of str)
    :param output_dir: path to the directory where temporary files should be saved (default: '') (str)
    :param samtools_fullpath: path to samtools on the computer (default: "samtools') (str)
    :param base_quality_limit: base quality limit for the pileup generation (default: 30) (int)
    :param samtools_flags: additional flags for samtools (default: ' -B -d ' + str(SAMTOOLS_MAX_DEPTH) + ' ') (str)
    :param print_original: If True, prints the original string generated with samtools mpileup as well (default: True) (bool)
    :param filename: If filename is specified, results will be writted to that path. Otherwise, results are written to [output_dir]/checkPileup_tmp.csv (default: None) (str)

    :returns: df: The processed results of the pileup, containing information on all samples in a pandas.DataFrame.

    """

    rmfile = False
    if filename is None:
        filename = output_dir + '/checkPileup_tmp.csv'
        rmfile = True

    __subprocess.call('rm ' + filename, shell=True)

    cmd = ' ' + samtools_fullpath + '  mpileup -Q ' + str(base_quality_limit) + ' ' + samtools_flags
    cmd += ' -f ' + ref_fasta

    if chrom_list.__class__ != list and from_pos_list.__class__ != list and to_pos_list.__class__ != list:
        cmd += ' -r ' + chrom_list + ':' + str(int(from_pos_list) + 1) + '-' + str(to_pos_list) + ' '
    else:
        if __os.path.isfile(output_dir + '/tmp_bedfile.bed'):
            __subprocess.call('rm ' + output_dir + '/tmp_bedfile.bed', shell=True)
        for c, pf, pt in zip(chrom_list, from_pos_list, to_pos_list):
            with open(output_dir + '/tmp_bedfile.bed', 'a') as f:
                f.write(c + '\t' + str(pf) + '\t' + str(pt) + '\n')
        cmd += ' -l ' + output_dir + '/tmp_bedfile.bed '
    for bam_file in bam_filename:
        cmd += input_dir + bam_file + ' '
    cmd += ' 2>> ' + output_dir + '/samtools.log '

    if print_original:
        pileup_string = __subprocess.check_output(cmd, shell=True).decode('utf-8')
        print(pileup_string)

    cmd += '| ' + HOME + '/C/checkPileup ' + str(base_quality_limit) + ' '
    for bam_file in bam_filename:
        cmd += bam_file + ' '
    cmd += '>> ' + filename

    __subprocess.call(cmd, shell=True)

    if not (chrom_list.__class__ != list and from_pos_list.__class__ != list and to_pos_list.__class__ != list):
        __subprocess.call('rm ' + output_dir + '/tmp_bedfile.bed', shell=True)

    df = __pd.read_csv(filename, sep='\t')
    if rmfile:
        __subprocess.call('rm ' + filename, shell=True)
    df['#chr'] = df['#chr'].apply(str)
    return df


def get_details_for_mutations(ref_fasta, input_dir, bam_filename,
                              mutations_dataframe=None,
                              output_dir=None,
                              mutations_filename=None,
                              samtools_fullpath='samtools',
                              base_quality_limit=30,
                              samtools_flags=' -B -d ' + str(SAMTOOLS_MAX_DEPTH) + ' '):
    """

    Get detailed results for the list of mutations contained in the mutations attribute of the object.

    :param ref_fasta: the path to the reference genome fasta file (str)
    :param input_dir: the path to the directory where bam files are located (str)
    :param bam_filename: the filenames for the bam files of the investigated samples (list of str)
    :param mutations_dataframe: a pandas.DataFrame where mutations are stored (default: None) (pandas.DataFrame)
    :param output_dir: path to the directory where temporary files should be saved (default: '') (str)
    :param mutations_filename: path to the file(s) where mutations are stored (default: None) (list of str)
    :param samtools_fullpath: path to samtools on the computer (default: "samtools') (str)
    :param base_quality_limit: base quality limit for the pileup generation (default: 30) (int)
    :param samtools_flags: additional flags for samtools (default: ' -B -d ' + str(SAMTOOLS_MAX_DEPTH) + ' ') (str)

    :returns: df_joined: A dataframe containing the detailed results. (pandas.DataFrame)

    """

    if not isinstance(mutations_dataframe, type(None)):
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations_dataframe" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations_dataframe" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    chrom_list = list(mutations_dataframe['chr'])
    from_pos_list = [int(k) - 1 for k in list(mutations_dataframe['pos'])]
    to_pos_list = list(mutations_dataframe['pos'])
    df = check_pileup(chrom_list=chrom_list,
                      from_pos_list=from_pos_list,
                      to_pos_list=to_pos_list,
                      ref_fasta=ref_fasta,
                      input_dir=input_dir,
                      bam_filename=bam_filename,
                      output_dir=output_dir,
                      samtools_fullpath=samtools_fullpath,
                      base_quality_limit=base_quality_limit,
                      samtools_flags=samtools_flags,
                      print_original=False)

    df.rename(columns={"#chr": "chr_x", "pos": "pos_x", "ref": "ref_x"}, inplace=True)

    df_joined = __pd.merge(mutations_dataframe, df, how='inner', left_on=['chr', 'pos', 'ref'],
                           right_on=['chr_x', 'pos_x', 'ref_x'])

    cols_to_keep = list(mutations_dataframe.columns)

    for bam_file in bam_filename:
        cols_to_keep.append('cov_' + bam_file)
        cols_to_keep.append('refFreq_' + bam_file)
        cols_to_keep.append('mutFreq_' + bam_file)
        df_joined['refFreq_' + bam_file] = df_joined.apply(__get_ref_freq, args=(bam_file,), axis=1)
        df_joined['mutFreq_' + bam_file] = df_joined.apply(__get_mut_freq, args=(bam_file,), axis=1)

    df_joined = df_joined[cols_to_keep]

    df_joined.sort_values(by=['chr', 'pos'], inplace=True)

    return df_joined


def __get_ref_freq(row, sample_name):
    """

    Get the reference base frequency in a row of the database containing detailed mpileup information.

    :param row: the row of the database
    :param sample_name: the name of the sample (str)

    :returns: the value of the reference base frequency (float)

    """
    col_name = row['ref'].upper() + 'freq_' + sample_name
    if row[col_name] >= 0:
        return row[col_name]
    else:
        return 0.0


def __get_mut_freq(row, sample_name):
    """

    Get the mutated base frequency in a row of the database containing detailed mpileup information.

    :param row: the row of the database
    :param sample_name: the name of the sample (str)

    :returns: the value of the mutated base frequency (float)

    """
    if row['type'] == 'SNV':
        col_name = row['mut'].upper() + 'freq_' + sample_name
    elif row['type'] == 'INS':
        col_name = 'insfreq_' + sample_name
    elif row['type'] == 'DEL':
        col_name = 'delfreq_' + sample_name
    if row[col_name] >= 0:
        return row[col_name]
    else:
        return 0.0


def __get_cosine_similarity(spectrum1, spectrum2):
    """

    Get cosine similarity between two vectors.

    :param spectrum1: vector1 (list, numpy.array)
    :param spectrum2: vector2 (list, numpy.array)

    :returns: cosine similarity (float)

    """
    return 1 - __scipy.spatial.distance.cosine(spectrum1, spectrum2)


def __initialize_proportions(spectrum, sig_matrix, equal=False):
    """

    Initialize the initial proportions of the signature mixture.

    :param spectrum: The spectrum to be decomposed. (list, numpy.array)
    :param sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types, columns: reference signatures)
    :param equal: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)

    :returns: The initial proportions of the signatures. (numpy.array)

    """
    cos_similarities = []
    for sig in sig_matrix.T:
        cos_similarities.append(__get_cosine_similarity(spectrum, sig))
    init_prop = __np.array(cos_similarities) / __np.sum(cos_similarities)

    if equal:
        return __np.ones(len(init_prop)) / len(init_prop)

    return init_prop


def __get_delta_jg(spectrum, sig_matrix):
    """

    Initialize the probability matrix of each mutation in the spectrum originating from the given signature.

    :param spectrum: Original spectrum to decompose. (numpy.array)
    :param sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types, columns: reference signatures)

    :returns: delta_matrix: A matrix containing each mutation as a row, with the probability of it originating from a specific reference signature as a column. (numpy.array)

    """
    delta_matrix = __np.zeros((int(__np.sum(spectrum)), sig_matrix.shape[1]))

    i = 0
    for sID, s in enumerate(spectrum):
        for k in range(int(s)):
            delta_matrix[i, :] = sig_matrix[sID, :]
            i += 1
    return delta_matrix


def __calculate_z_jg(theta, delta_matrix):
    """

    Calculate the conditional probability of a mutation originating from a specific signature, given the proportion of the reference signatures in
    the mixture.

    :param theta: The current proportions of the reference signatures in the mixture. (numpy.array)
    :param delta_matrix: The original probability matrix of a given mutation originating from a specific reference signature. (numpy.array)

    :returns: z_matrix: A matrix containing the conditional probabilities of a mutation (row) arising from a specific reference signature (column), given the current mixture proportions of the signatures. (numpy.array)

    """
    z_matrix = __np.divide((theta * delta_matrix).T, (theta * delta_matrix).sum(axis=1)).T
    return z_matrix


def __calculate_theta(z_matrix):
    """

    Calculate the new mixture proportions from the conditional probability matrix.

    :param z_matrix: Conditional probability matrix. (numpy.array)

    :returns: theta: Updated mixture proportions. (numpy.array)

    """
    theta = z_matrix.sum(axis=0) / z_matrix.sum()
    return theta


def __check_if_converged(theta, prev_theta, tol=0.0001):
    """

    Check if the iterations have converged, ie. the old and the updated mixture proportions do not differ much.

    :param theta: Current mixture proportions. (numpy.array)
    :param prev_theta: Previous mixture proportions. (numpy.array)
    :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)

    :returns: True if converged, False otherwise (boolean)

    """
    if (__np.sum(__np.abs(__np.exp(prev_theta) - __np.exp(theta))) < tol):
        return True
    else:
        return False


def __reconstruct_spectrum(theta, sig_matrix):
    """

    Reconstruct the spectrum using the estimated mixture proportions of the reference signatures and the signature matrix.

    :param theta: estimated mixture proportions (numpy.array)
    :param sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types, columns: reference signatures) (numpy.array)

    :returns: rec_spectrum: Reconstructed spectrum. (numpy.array)

    """
    rec_spectrum = (theta * sig_matrix).sum(axis=1) / (theta * sig_matrix).sum()
    return rec_spectrum


def __EM(spectrum, sig_matrix,
         max_iter=1000,
         equal_initial_proportions=False,
         tol=0.0001):
    """

    Perform expectation maximization to infer the proportions of reference signatures in a spectrum.

    :param spectrum: The original spectrum (counts). (numpy.array)
    :param sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types, columns: reference signatures) (numpy.array)
    :param max_iter: The maximum number of iterations (default: 1000) (int)
    :param equal_initial_proportions: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)
    :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)

    :returns: (theta, z_matrix)

        - theta: The final mixture proportions. (numpy.array)
        - z_matrix: The final set of conditional probabilities, that a mutation originated from a specific reference signature, given the final mixture proportions.

    """
    # initialize proportions
    theta_prev = __initialize_proportions(spectrum, sig_matrix, equal=equal_initial_proportions)

    # calculate d_matrix
    delta_matrix = __get_delta_jg(spectrum, sig_matrix)

    for i in range(max_iter):
        z_matrix = __calculate_z_jg(theta_prev, delta_matrix)
        theta = __calculate_theta(z_matrix)
        if (__check_if_converged(theta, theta_prev, tol=tol)):
            break
        else:
            theta_prev = theta

    return theta, z_matrix


def __filter_sigs(sig_matrix, z_matrix, filter_percent=0, filter_count=0, keep_top_n=None):
    """

    Filter signatures with negligible contribution to the mixture.

    :param sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types, columns: reference signatures) (numpy.array)
    :param z_matrix: Conditional probability matrix. (numpy.array)
    :param filter_percent: Filter signatures, that contribute less than filter_percent of the mutations in the sample. (default: 0) (float)
    :param filter_count: Filter signatures, that contribute less than filter_count number of mutations in the sample. (default: 0) (int)
    :param keep_top_n: Only keep those signatures that contribute to the mixture with the top keep_top_n number of mutation. (default: None) (int)

    :returns: The filtered signature matrix without the filtered signatures. (numpy.array)

    """
    sig_IDs = z_matrix.argmax(axis=1)
    unique_sigs, sig_counts = __np.unique(sig_IDs, return_counts=True)
    sig_percents = sig_counts * 100 / sig_counts.sum()

    realSig_IDs = unique_sigs[(sig_counts >= filter_count) * (sig_percents >= filter_percent)]

    if keep_top_n:
        realSig_IDs = unique_sigs[sig_counts.argsort()[-keep_top_n:][::-1]]

    return sig_matrix[:, realSig_IDs], realSig_IDs


def __read_signatures_files(fn, own_file=False, sig_type='indel'):
    """

    Read signature matrix from file.

    :param fn: The path to the csv file containing the signature matrix. (str)
    :param own_file: If using an own file, set to True. (default: False) (bool)
    :param sig_type: Signature type ('indel', 'SNV' or 'DNV') (default: "indel") (str)

    :returns: (sig_matrix, sig_names)

        - sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types,
        columns: reference signatures) (numpy.array)
        - sig_names: Names of the signatures (numpy.array)

    """
    df_sigs = __pd.read_csv(fn)

    sig_type_string = 'ID'
    if sig_type == 'SNV':
        sig_type_string = 'SBS'
    elif sig_type == 'DNV':
        sig_type_string = 'DBS'

    if own_file:
        sig_names = [k for k in list(df_sigs.columns)]
    else:
        sig_names = [k for k in list(df_sigs.columns) if sig_type_string in k]

    sig_matrix = __np.array(df_sigs[sig_names])

    return sig_matrix, sig_names


def __decompose_spectrum(spectrum, sig_matrix, sig_names,
                         equal_initial_proportions=False,
                         tol=0.0001,
                         max_iter=1000,
                         filter_percent=0,
                         filter_count=0,
                         keep_top_n=None,
                         use_signatures=None):
    """

    Decompose a specific spectrum to a mixture of reference signatures.

    :param spectrum: The original spectrum (counts). (numpy.array)
    :param sig_matrix: The matrix containing the weights of each mutational type in each reference signature. (rows: mutational types, columns: reference signatures) (numpy.array)
    :param sig_names: Names of the signatures (numpy.array)
    :param equal_initial_proportions: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)
    :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)
    :param max_iter: The maximum number of iterations (default: 1000) (int)
    :param filter_percent: Filter signatures, that contribute less than filter_percent of the mutations in the sample. (default: 0) (float)
    :param filter_count: Filter signatures, that contribute less than filter_count number of mutations in the sample. (default: 0) (int)
    :param keep_top_n: Only keep those signatures that contribute to the mixture with the top keep_top_n number of mutation. (default: None) (int)
    :param use_signatures: Use a specific subset of all signatures. A list of signature names. (default: None) (list of str)

    :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

    """
    if use_signatures != None:
        use_IDs = __np.array([sig_names.index(us) for us in use_signatures])
        sig_matrix = sig_matrix[:, use_IDs]
        sig_names = __np.array(sig_names)[use_IDs]

    # do one round of EM
    final_theta, final_z_matrix = __EM(spectrum, sig_matrix,
                                       max_iter=max_iter,
                                       tol=tol,
                                       equal_initial_proportions=equal_initial_proportions)

    # filter potential contributors
    new_sig_matrix, realSig_IDs = __filter_sigs(sig_matrix, final_z_matrix,
                                                filter_percent=filter_percent,
                                                filter_count=filter_count,
                                                keep_top_n=keep_top_n)

    # do next round of EM
    final_theta, final_z_matrix = __EM(spectrum, new_sig_matrix,
                                       max_iter=max_iter,
                                       tol=tol,
                                       equal_initial_proportions=equal_initial_proportions)

    all_theta = []
    k = 0
    for i in range(len(sig_names)):
        if i in realSig_IDs:
            all_theta.append(final_theta[k])
            k += 1
        else:
            all_theta.append(0)

    return all_theta


def decompose_indel_spectra(IDspectrumDict,
                            sample_names=None,
                            signatures_file=None,
                            equal_initial_proportions=False,
                            tol=0.0001,
                            max_iter=1000,
                            filter_percent=0,
                            filter_count=0,
                            keep_top_n=None,
                            use_signatures=None,
                            ignore_signatures=None):
    """

    Run the whole pipeline of decomposing indel spectra for the samples specified in sample_names.

    :param IDspectrumDict: dictionary containing indel spectra as values and sample names as keys (dictionary)
    :param sample_names: The list of sample names analysed. (list of str)
    :param signatures_file: The path to the csv file containing the signature matrix. (str)
    :param equal_initial_proportions: If True, initial weights are initialized to be equal for all reference signature. Otherwise, they are initialized so the signatures with large cosine similarity with the original spectrum get larger weights. (default: False) (bool)
    :param tol: The maximal difference between the proportions for convergence to be True. (default: 0.0001) (float)
    :param max_iter: The maximum number of iterations (default: 1000) (int)
    :param filter_percent: Filter signatures, that contribute less than filter_percent of the mutations in the sample. (default: 0) (float)
    :param filter_count: Filter signatures, that contribute less than filter_count number of mutations in the sample. (default: 0) (int)
    :param keep_top_n: Only keep those signatures that contribute to the mixture with the top keep_top_n number of mutation. (default: None) (int)
    :param use_signatures: Use a specific subset of all signatures. A list of signature names. (default: None) (list of str)
    :param ignore_signatures: Exclude a specific subset of all signatures from the analysis. A list of signature names. (default: None) (list of str)

    :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

    """

    if sample_names == None:
        sample_names = sorted(list(IDspectrumDict.keys()))

    if signatures_file == None:
        signatures_file = HOME + '/alexandrovSignatures/sigProfiler_ID_signatures.csv'

    sig_matrix, sig_names = __read_signatures_files(signatures_file, sig_type='indel')

    sig_names_filtered = sig_names
    if use_signatures is not None:
        sig_names_filtered = sorted(use_signatures)
    if ignore_signatures is not None:
        sig_names_filtered = sorted(list(set(sig_names_filtered) - set(ignore_signatures)))

    theta_matrix = []
    for s in sample_names:
        theta = __decompose_spectrum(IDspectrumDict[s], sig_matrix, sig_names,
                                     equal_initial_proportions=equal_initial_proportions,
                                     tol=tol,
                                     max_iter=max_iter,
                                     filter_percent=filter_percent,
                                     filter_count=filter_count,
                                     keep_top_n=keep_top_n,
                                     use_signatures=sig_names_filtered)
        theta_matrix.append(theta)

    return plot.__plot_spectrum_decomposition(sample_names, theta_matrix, sig_names_filtered, spectrum_type='indel')


def decompose_DNV_spectra(DNVspectrumDict,
                          sample_names=None,
                          signatures_file=None,
                          equal_initial_proportions=False,
                          tol=0.0001,
                          max_iter=1000,
                          filter_percent=0,
                          filter_count=0,
                          keep_top_n=None,
                          use_signatures=None,
                          ignore_signatures=None):
    """

    Run the whole pipeline of decomposing DNV spectra for the samples specified in sample_names.

    :param DNVspectrumDict: dictionary containing DNV spectra as values and sample names as keys (dictionary)
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

    :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

    """

    if sample_names == None:
        sample_names = sorted(list(DNVspectrumDict.keys()))

    if signatures_file == None:
        signatures_file = HOME + '/alexandrovSignatures/sigProfiler_DBS_signatures.csv'

    sig_matrix, sig_names = __read_signatures_files(signatures_file, sig_type='DNV')

    sig_names_filtered = sig_names
    if use_signatures != None:
        sig_names_filtered = sorted(use_signatures)
    if ignore_signatures != None:
        sig_names_filtered = sorted(list(set(sig_names_filtered) - set(ignore_signatures)))

    theta_matrix = []
    for s in sample_names:
        theta = __decompose_spectrum(DNVspectrumDict[s], sig_matrix, sig_names,
                                     equal_initial_proportions=equal_initial_proportions,
                                     tol=tol,
                                     max_iter=max_iter,
                                     filter_percent=filter_percent,
                                     filter_count=filter_count,
                                     keep_top_n=keep_top_n,
                                     use_signatures=sig_names_filtered)
        theta_matrix.append(theta)

    return plot.__plot_spectrum_decomposition(sample_names, theta_matrix, sig_names_filtered, spectrum_type='DNV')


def decompose_SNV_spectra(SNVspectrumDict,
                          sample_names=None,
                          signatures_file=None,
                          equal_initial_proportions=False,
                          tol=0.0001,
                          max_iter=1000,
                          filter_percent=0,
                          filter_count=0,
                          keep_top_n=None,
                          use_signatures=None,
                          ignore_signatures=None):
    """

    Run the whole pipeline of decomposing SNV spectra for the samples specified in sample_names.

    :param SNVspectrumDict: dictionary containing SNV spectra as values and sample names as keys (dictionary)
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

    :returns: The final proportions of all signatures in the mixture. (Filtered out or not used signatures appear with a proportion of zero.) (numpy.array)

    """

    if sample_names == None:
        sample_names = sorted(list(SNVspectrumDict.keys()))

    if signatures_file == None:
        signatures_file = HOME + '/alexandrovSignatures/sigProfiler_SBS_signatures_2018_03_28.csv'

    sig_matrix, sig_names = __read_signatures_files(signatures_file, sig_type='SNV')

    sig_names_filtered = sig_names
    if use_signatures != None:
        sig_names_filtered = sorted(use_signatures)
    if ignore_signatures != None:
        sig_names_filtered = sorted(list(set(sig_names_filtered) - set(ignore_signatures)))

    theta_matrix = []
    for s in sample_names:
        theta = __decompose_spectrum(SNVspectrumDict[s], sig_matrix, sig_names,
                                     equal_initial_proportions=equal_initial_proportions,
                                     tol=tol,
                                     max_iter=max_iter,
                                     filter_percent=filter_percent,
                                     filter_count=filter_count,
                                     keep_top_n=keep_top_n,
                                     use_signatures=sig_names_filtered)
        theta_matrix.append(theta)

    return plot.__plot_spectrum_decomposition(sample_names, theta_matrix, sig_names_filtered, spectrum_type='SNV')
