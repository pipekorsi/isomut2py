try:
    from isomut2py import io

    import matplotlib as __mpl

    __mpl.use('Agg')
    import matplotlib.pyplot as __plt
    from matplotlib.patches import Rectangle as __Rectangle
    import base64 as __base64
    import numpy as __np
    from io import BytesIO as __BytesIO
    import os as __os
    import pandas as __pd
    from datetime import datetime as __datetime
    from scipy import stats as __stats
    import seaborn as __sns
except ImportError:
    print('ImportError in isomut2py.plot, some plotting functions might not work.')


def plot_karyotype_for_chrom(chrom, df, return_string=True):
    """

    Plots karyotype information (coverage, estimated ploidy, estimated LOH, reference base frequencies) about the sample for a given chromosome.

    :param chrom: The chromosome to plot. (str)
    :param df: The dataframe containing ploidy and LOH information. (pandas.DataFrame)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: True) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a matplotlib figure.

    """
    try:
        __sns.set(style="whitegrid", font="DejaVu Sans")
    except:
        pass

    covcolor = '#FFA500'  # '#FFD700'
    rnfcolor = '#ADD8E6' # '#483D8B'  # (209 / 255., 10 / 255., 124 / 255.)
    rnfalpha = 0.4
    guidelinecolor = '#E6E6FA'

    p = list(df.sort_values(by='pos')['pos'])
    cov = list(df.sort_values(by='pos')['cov'])
    rf = list(df.sort_values(by='pos')['mut_freq'])
    rf = list(1 - __np.array(rf))
    pl = list(df.sort_values(by='pos')['ploidy'])
    pl_i = list(float(1) / __np.array(pl))
    loh = __np.array(list(df.sort_values(by='pos')['LOH']))
    loh_change = __np.where(loh[:-1] != loh[1:])[0]

    f, ax1 = __plt.subplots()
    f.set_size_inches(20, 10)
    ax2 = ax1.twinx()
    for i in range(len(loh_change)):
        if (i == 0 and loh[loh_change[i]] == 1):
            w = p[loh_change[i]] - p[0]
            k = __Rectangle((p[0], 0), w, 1, alpha=0.1, facecolor='black', edgecolor='none')
            ax2.add_patch(k)
        if (loh[loh_change[i]] == 0):
            if (i == len(loh_change) - 1):
                w = max(p) - p[loh_change[i]]
            else:
                w = p[loh_change[i + 1]] - p[loh_change[i]]
            k = __Rectangle((p[loh_change[i]], 0), w, 1, alpha=0.1, facecolor='black', edgecolor='none')
            ax2.add_patch(k)

    ax1.plot(p, cov, c=covcolor)

    for i in range(2, 10):
        ax2.plot(p, [1 - float(1) / i] * len(p), c=guidelinecolor)
        ax2.plot(p, [float(1) / i] * len(p), c=guidelinecolor)
    ax2.scatter(p, rf, c='none', edgecolor=rnfcolor, alpha=rnfalpha)
    ax2.scatter(p, pl_i, c='none', edgecolor='black', alpha=1)

    ax2.set_ylabel('reference base frequency\n', size=15, color=rnfcolor)
    ax1.set_xlabel('\n\ngenomic position', size=15)
    ax2.yaxis.set_tick_params(labelsize=15, colors=rnfcolor)
    ax2.xaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.set_ylabel('coverage\n', size=15, color=covcolor)
    ax1.yaxis.set_tick_params(labelsize=15, colors=covcolor)
    ax1.set_ylim([0, 1000])
    ax2.set_ylim([0, 1])
    ax2.set_xlim([min(p), max(p)])
    ax1.spines['bottom'].set_color('lightgrey')
    ax1.spines['top'].set_color('lightgrey')
    ax1.spines['left'].set_color('lightgrey')
    ax1.spines['right'].set_color('lightgrey')
    ax2.spines['bottom'].set_color('lightgrey')
    ax2.spines['top'].set_color('lightgrey')
    ax2.spines['left'].set_color('lightgrey')
    ax2.spines['right'].set_color('lightgrey')
    __plt.title('Chromosome: ' + chrom + '\n\n', size=20)

    if (return_string):
        figfile = __BytesIO()
        __plt.savefig(figfile, bbox_inches='tight', format='png')
        __plt.close()
        figfile.seek(0)
        figdata_png = __base64.b64encode(figfile.getvalue())
        return figdata_png
    else:
        __plt.show()
        __plt.close()
        return f


def plot_karyotype_for_all_chroms(chromosomes, output_dir, return_string=False):
    """

    Plots karyotype information (coverage, estimated ploidy, estimated LOH, reference base frequencies) about the sample for all analysed chromosomes.

    :param chromosomes: the list of chromosomes to plot (list of str)
    :param output_dir: the path to the directory where PE_fullchrom_[chrom].txt files are stored. (str)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: True) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, nothing.

    """

    all_chrom_figfiles = []
    for c in chromosomes:
        if not __os.path.isfile(output_dir + '/PE_fullchrom_' + c + '.txt'):
            raise ValueError(
                'File ' + output_dir + '/PE_fullchrom_' + c + '.txt is not yet created, call "run_ploidy_estimation" first.')
        df = __pd.read_csv(output_dir + '/PE_fullchrom_' + c + '.txt', sep='\t')
        df = df[['chrom', 'pos', 'cov', 'mut_freq', 'ploidy', 'LOH']]
        if (return_string):
            all_chrom_figfiles.append(plot_karyotype_for_chrom(chrom=c, df=df, return_string=True))
        else:
            all_chrom_figfiles.append(plot_karyotype_for_chrom(chrom=c, df=df, return_string=False))
    return all_chrom_figfiles


def __get_BAF_and_DR(avg_dip_cov, chroms, chrom_length_list, datadir,
                     binsize=1000000,
                     overlap=50000,
                     cov_min=5,
                     cov_max=200):
    """

    Calculates depth ratio means, 25th and 75th percentiles in windows of genomic positions with a moving average method.
    Calculates B-allele frequency means, 25th and 75th percentiles in windows of genomic positions with a moving average method.

    :param avg_dip_cov: average coverage of diploid regions (float)
    :param chroms: list of chromosomes of the genome (list of str)
    :param chrom_length_list: list of chromosome lengths in basepairs (list of int)
    :param datadir: the path to the directory where PE_fullchrom_[chrom].txt files are stored
    :param binsize: the binsize used for moving average (default: 1000000) (int)
    :param overlap: the overlap used for moving average (default: 50000) (int, smaller than binsize)
    :param cov_min: the minimum coverage for a position to be included (default: 5) (int)
    :param cov_max: the maximum coverage for a position to be included (default: 2000) (int)

    :returns: (real_pos, dr, dr_25, dr_75, baf, baf_25, baf_75)

        - real_pos: list of real genomic positions (int in range (1-genome_length))
        - dr: list of average depth ratios in the neighborhood of the above positions
        - dr_25: list of 25th percentiles of depth ratios in the neighborhood of the above positions
        - dr_75: list of 75th percentiles of depth ratios in the neighborhood of the above positions
        - baf: list of average B-allele frequencies in the neighborhood of the above positions
        - baf_25: list of 25th percentiles of B-allele frequencies in the neighborhood of the above positions
        - baf_75: list of 75th percentiles of B-allele frequencies in the neighborhood of the above positions

    """

    real_pos, dr, dr_25, dr_75, baf, baf_25, baf_75 = [[] for i in range(7)]

    for c in chroms:
        posbase = sum(chrom_length_list[:chroms.index(c)])
        tmp = __pd.read_csv(datadir + '/PE_fullchrom_' + c + '.txt', sep='\t')
        tmp = tmp[(tmp['cov'] >= cov_min) & (tmp['cov'] <= cov_max)]

        posstart = tmp['pos'].min()
        posmax = tmp['pos'].max()
        while posstart < posmax:
            if tmp[(tmp['pos'] >= posstart) & (tmp['pos'] < posstart + binsize)].shape[0] > 0:
                bafs = __np.array(list(tmp[(tmp['pos'] >= posstart) & (tmp['pos'] < posstart + binsize)]['mut_freq']))
                if len(bafs[bafs >= 0.5]) > 0:
                    real_pos.append(int(posstart + binsize / 2) + posbase)
                    bafs = 1 - bafs[bafs >= 0.5]
                    baf.append(__np.mean(bafs))
                    baf_25.append(__np.percentile(bafs, 25))
                    baf_75.append(__np.percentile(bafs, 75))
                    covs = __np.array(list(tmp[(tmp['pos'] >= posstart) & (tmp['pos'] < posstart + binsize)]['cov']))
                    dr.append(__np.mean(covs / avg_dip_cov))
                    dr_25.append(__np.percentile(covs / avg_dip_cov, 25))
                    dr_75.append(__np.percentile(covs / avg_dip_cov, 75))
            if posstart + binsize > posmax:
                break
            posstart += (binsize - overlap)

    return real_pos, dr, dr_25, dr_75, baf, baf_25, baf_75


def __get_PL_and_LOH(bed_filename,
                     chroms,
                     chrom_lenght_list,
                     bed_file_sep=',',
                     numtoplot=10000,
                     minlength=3000000):
    """

    From a bed file of ploidies and LOH regions, generates a list of numtoplot positions with respective ploidies and LOHs.

    :param bed_filename: path to the bed file of the sample containing ploidy and LOH information (str)
    :param chroms: list of chromosomes in the genome (list of str)
    :param chrom_lenght_list: list of chromosome lengths in basepairs (list of int)
    :param bed_file_sep: bed file separator (default: ",") (str)
    :param numtoplot: the number measurement points (default: 10000) (int)
    :param minlength: the minimal length of a region to be plotted (default: 3000000) (int)

    :returns: (toplot_pos, toplot_pl, toplot_pos_loh, toplot_loh)

        - toplot_pos: genomic positions in which ploidy should be plotted
        - toplot_pl: the ploidy in these positions
        - toplot_pos_loh: genomic positions in which LOHs should be plotted
        - toplot_loh: the LOH in these positions

    """

    toplot_pos = __np.arange(numtoplot) * int(sum(chrom_lenght_list) / numtoplot)
    toplot_pl = __np.zeros(numtoplot)
    toplot_loh = __np.zeros(numtoplot)

    pl_res = __pd.read_csv(bed_filename, sep=bed_file_sep)

    cs_list, ce_list, p_list, l_list = [[] for i in range(4)]

    for c, cs, ce, p, l in zip(list(pl_res['chrom']), list(pl_res['chromStart']), list(pl_res['chromEnd']),
                               list(pl_res['ploidy']), list(pl_res['LOH'])):
        if ce - cs > minlength:
            cs_list.append(cs + sum(chrom_lenght_list[:chroms.index(c)]))
            ce_list.append(ce + sum(chrom_lenght_list[:chroms.index(c)]))
            p_list.append(p)
            l_list.append(l)
    for i in range(len(cs_list) - 1):
        if ce_list[i] + 1 < cs_list[i + 1]:
            bp = int(__np.mean([ce_list[i], cs_list[i + 1]]))
            ce_list[i] = bp
            cs_list[i + 1] = bp + 1
        for j in range(numtoplot):
            if toplot_pos[j] >= cs_list[i] and toplot_pos[j] < ce_list[i]:
                toplot_pl[j] = p_list[i]
                toplot_loh[j] = l_list[i]
    toplot_pos_loh = toplot_pos[toplot_loh == 1]
    toplot_loh = __np.ones(len(toplot_pos_loh))

    return toplot_pos, toplot_pl, toplot_pos_loh, toplot_loh


def __plot_karyotype(real_pos, dr, dr_25, dr_75, baf, baf_25, baf_75, s0, s1, loh_pos, loh,
                     all_chroms,
                     chrom_length_list,
                     chroms_with_text=None):
    """

    Plots a karyotype summary for the whole genome.

    :param real_pos: list of real genomic positions (int in range (1-genome_length))
    :param dr: list of average depth ratios in the neighborhood of the above positions
    :param dr_25: list of 25th percentiles of depth ratios in the neighborhood of the above positions
    :param dr_75: list of 75th percentiles of depth ratios in the neighborhood of the above positions
    :param baf: list of average B-allele frequencies in the neighborhood of the above positions
    :param baf_25: list of 25th percentiles of B-allele frequencies in the neighborhood of the above positions
    :param baf_75: list of 75th percentiles of B-allele frequencies in the neighborhood of the above positions
    :param s0: list of genomic positions in which to plot ploidy information
    :param s1: ploidies in these positions
    :param loh_pos: list of genomic positions in which to plot LOH information
    :param loh: list of LOH values in these positions
    :param all_chroms: list of all chromosomes in the genome (list of str)
    :param chrom_length_list: list of chromosome lengths (list of int)
    :param chroms_with_text: the list of chromosomes to be indicated with text on the plot (list of str) (If there are many short chromosomes or they have long names, it is useful to only indicate a few with text on the plot.)

    :returns: a matplotlib figure

    """

    if chroms_with_text is None:
        chroms_with_text = all_chroms

    try:
        __sns.set(style="white", font="DejaVu Sans")
    except:
        pass

    fig, axes = __plt.subplots(nrows=2, sharex=True)
    fig.set_size_inches(18 / 2.5, 9 / 2.5)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97, hspace=0.2)

    ax_dr, ax_baf = axes
    ax_pl1 = ax_dr.twinx()
    ax_pl2 = ax_baf.twinx()

    idx = __np.random.choice(__np.arange(len(real_pos)), size=int(__np.min([5000, len(real_pos)])), replace=False)

    # plotting depth ratios

    error = [__np.array(dr_25)[idx], __np.array(dr_75)[idx]]
    ax_dr.errorbar(__np.array(real_pos)[idx], __np.array(dr)[idx], yerr=error, fmt='o', ecolor='#E6E6FA',
                   markeredgecolor='black', markerfacecolor='black', ms=2, capsize=0)
    ax_pl1.scatter(s0, s1, edgecolor='red', facecolor='red', s=3)
    ax_dr.set_ylabel('Depth ratio\n', size=10)
    ax_pl1.set_ylabel('\n\nEstimated ploidy', size=10)
    ax_dr.set_ylim([-0.25, 2.75])
    ax_pl1.set_ylim([-0.5, 5.5])
    ax_dr.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax_pl1.set_yticks([0, 1, 2, 3, 4, 5])
    ax_dr.tick_params(axis='x', size=0)
    ax_pl1.tick_params(axis='x', size=0)

    # plotting chrom borders

    for c in all_chroms:
        ax_dr.plot([sum(chrom_length_list[:all_chroms.index(c)]), sum(chrom_length_list[:all_chroms.index(c)])],
                   [-0.25, 2.75], lw=1.5, c='#9C9C9C')
        ax_baf.plot([sum(chrom_length_list[:all_chroms.index(c)]), sum(chrom_length_list[:all_chroms.index(c)])],
                    [-0.05, 0.55], lw=1.5, c='#9C9C9C')
    for c in chroms_with_text:
        ax_dr.text(sum(chrom_length_list[:all_chroms.index(c)]) + 0.5 * chrom_length_list[all_chroms.index(c)], -0.65,
                   c, fontsize=6, color='#8F8F8F', horizontalalignment='center')

    # plotting bafs

    error = [__np.array(baf_25)[idx], __np.array(baf_75)[idx]]
    ax_baf.errorbar(__np.array(real_pos)[idx], __np.array(baf)[idx], yerr=error, fmt='o', ecolor='#E6E6FA',
                    markeredgecolor='black', markerfacecolor='black', ms=2, capsize=0)
    ax_pl2.scatter(s0, 1 / __np.array(s1) * (__np.array(s1) != 1), edgecolor='red', facecolor='red', s=3)
    if len(loh_pos) > 0:
        ax_pl2.scatter(loh_pos, loh * (-0.15), edgecolor='orange', facecolor='orange', s=1.5, clip_on=False)
    ax_pl2.text(-0.085, -0.14, 'LOH',
                verticalalignment='center', horizontalalignment='center',
                transform=ax_pl2.transAxes, fontsize=10, rotation=90)

    ax_baf.set_ylabel('BAF\n', size=10)
    ax_pl2.set_ylabel('\n1/Estimated ploidy', size=10)
    ax_baf.set_ylim([-0.05, 0.55])
    ax_baf.set_xlim([0, max(real_pos)])
    ax_pl2.set_ylim([-0.05, 0.55])
    ax_baf.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_pl2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_pl2.tick_params(axis='x', size=0)
    ax_baf.tick_params(axis='x', size=0)
    ax_baf.set_xticklabels(size=0, labels=[1, 2, 3])
    ax_pl2.set_xticklabels(size=0, labels=[1, 2, 3])

    # plotting fig
    __plt.show()
    return fig


def plot_karyotype_summary(haploid_coverage,
                           chromosomes,
                           chrom_length,
                           output_dir,
                           bed_filename,
                           bed_file_sep=',',
                           binsize=1000000,
                           overlap=50000,
                           cov_min=5,
                           cov_max=200,
                           min_PL_length=3000000,
                           chroms_with_text=None):
    """

    Plots karyotype summary for the whole genome with data preparation.

    :param haploid_coverage: the average coverage of haploid regions (or the half of that of diploid regions)
    :param chromosomes: list of chromosomes in the genome (list of str)
    :param chrom_length: list of chromosome lengths (list of int)
    :param output_dir: the path to the directory where PE_fullchrom_[chrom].txt files are located (str)
    :param bed_filename: the path to the bed file of the sample with ploidy and LOH information (str)
    :param bed_file_sep: bed file separator (default: ',') (str)
    :param binsize: the binsize used for moving average (default: 1000000) (int)
    :param overlap: the overlap used for moving average (default: 50000) (int, smaller than binsize)
    :param cov_min: the minimum coverage for a position to be included (default: 5) (int)
    :param cov_max: the maximum coverage for a position to be included (default: 2000) (int)
    :param min_PL_length: the minimal length of a region to be plotted (default: 3000000) (int)
    :param chroms_with_text: the list of chromosomes to be indicated with text on the plot (list of str) (If there are many short chromosomes or they have long names, it is useful to only indicate a few with text on the plot.)

    :returns: a matplotlib figure

    """

    real_pos, dr, dr_25, dr_75, baf, baf_25, baf_75 = __get_BAF_and_DR(avg_dip_cov=haploid_coverage * 2,
                                                                       chroms=chromosomes,
                                                                       chrom_length_list=chrom_length,
                                                                       datadir=output_dir,
                                                                       binsize=binsize,
                                                                       overlap=overlap,
                                                                       cov_min=cov_min,
                                                                       cov_max=cov_max)
    s0, s1, loh_pos, loh = __get_PL_and_LOH(bed_filename=bed_filename,
                                            chroms=chromosomes,
                                            chrom_lenght_list=chrom_length,
                                            bed_file_sep=bed_file_sep,
                                            numtoplot=5000,
                                            minlength=min_PL_length)

    f = __plot_karyotype(real_pos=real_pos,
                         dr=dr,
                         dr_25=dr_25,
                         dr_75=dr_75,
                         baf=baf,
                         baf_25=baf_25,
                         baf_75=baf_75,
                         s0=s0,
                         s1=s1,
                         loh_pos=loh_pos,
                         loh=loh,
                         all_chroms=chromosomes,
                         chrom_length_list=chrom_length,
                         chroms_with_text=chroms_with_text)

    return f


def generate_HTML_report_for_ploidy_est(chromosomes, output_dir, min_noise=__np.nan):
    """

    Generates a HTML file with figures displaying the results of ploidy estimation and saves it to output_dir/PEreport.html.

    :param chromosomes: list of chromosomes in the genome (list of str)
    :param output_dir: the path to the directory where PE_fullchrom_[chrom].txt files are located (str)
    :param min_noise: the minimal B-allele frequency for a position to be included in the analyses (default: numpy.nan) (float)

    """

    FIG_all_chroms = plot_karyotype_for_all_chroms(chromosomes, output_dir, return_string=True)

    string_for_all_chroms = ''
    for ch_figfile in FIG_all_chroms:
        string_for_all_chroms += '''<img src="data:image/png;base64,''' + ch_figfile.decode(
            'utf-8') + '''" alt="detailed_PEs.png"><br>'''

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
            <h2>IsoMut2 results - ploidy estimation</h2>
            <br>
            Date and time of analysis: ''' + str(__datetime.now()).split('.')[0] + ''' <br><br>
            Data stored at: <br>
            ''' + output_dir + ''' <br><br>
            <h3>Local ploidy estimates throughout the whole genome:</h3>
            Only those positions are included on the plots below, where the reference nucleotide frequency is in
            the range [''' + str(min_noise) + ''', ''' + str(1 - min_noise) + '''].
            <br><br>
            On the figures below yellow lines represent <i>coverage</i>, and purple dots the <i>reference nucleotide frequency</i> values for the above defined
            genomic positions. Black dots show the <i>inverse of the estimated copy number</i> at the given position. Grey rectangles indicate LOH (loss of
            heterozygosity) regions.
            <br><br>''' + string_for_all_chroms

    html_string += '''
            </body>
        </html>'''

    with open(output_dir + '/PEreport.html', 'w') as f:
        f.write(html_string)


def plot_coverage_distribution(cov_sample=None,
                               chromosomes=None,
                               output_dir=None,
                               cov_max=None,
                               cov_min=None,
                               distribution_dict=None):
    """

    Plot the coverage distribution of the sample.

    :param cov_sample: a sample of the coverage distribution (default: None) (array-like)
    :param chromosomes: the list of chromosomes in the genome (default: None) (list of str)
    :param output_dir: the path to the directory where PE_fullchrom_[chrom].txt files are located (default: None) (str)
    :param cov_max: the maximum value for the coverage for a position to be included on the plot (default: None) (int)
    :param cov_min: the minimum value for the coverage for a position to be included on the plot (default: None) (int)
    :param distribution_dict: a dictionary containing the fitted parameters of the coverage distribution (default: None) (dictionary with keys: 'mu', 'sigma', 'p')

    """

    try:
        __sns.set(style="whitegrid", font="DejaVu Sans")
    except:
        pass

    if cov_sample is None:
        cov_sample = io.get_coverage_distribution(chromosomes=chromosomes,
                                                  output_dir=output_dir,
                                                  cov_max=cov_max,
                                                  cov_min=cov_min)

    if distribution_dict is not None or (__os.path.isfile(output_dir + '/GaussDistParams.pkl')):
        if distribution_dict == None:
            distribution_dict = io.load_cov_distribution_parameters_from_file()

        range_max = distribution_dict['mu'][-1] + 3 * distribution_dict['sigma'][-1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
        x = __np.linspace(0, range_max, 500)

        for i in range(len(distribution_dict['mu'])):
            m = distribution_dict['p'][i] * __stats.norm.pdf(x, loc=distribution_dict['mu'][i],
                                                             scale=distribution_dict['sigma'][i])

            plot_temp = __plt.plot(x, m, label="Ploidy " + str(i + 1), lw=3, color=colors[i])
            __plt.fill_between(x, m, color=plot_temp[0].get_color(), alpha=0.3)
            __plt.xlim(0, range_max)

    __plt.hist(cov_sample, bins=100, histtype="step", density=True, color="k",
               lw=2, label="coverage histogram")
    __plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=0.0)

    __plt.title("Coverage distribution\n")
    __plt.xlabel('coverage')
    __plt.ylabel('relative occurence')
    f = __plt.gcf()
    __plt.show()
    __plt.close()

    return f


def __plot_tuning_curve(control_samples, mutation_dataframe, return_string=False, unique_samples=None):
    """

    Plots tuning curves for all mutations types (SNV, INS, DEL) and all ploidies for each available sample in the MutationDetection object.
    Samples listed as control_samples are highlighted with a different color.

    :param control_samples: a subset of bam_filename (list of sample names) that should be considered as control samples. Control samples are defined as samples where no unique mutations are expected to be found. (list of str)
    :param mutation_dataframe: The dataframe containing the mutations. (pandas.DataFrame)
    :param unique_samples: list of unique samples where at least one mutation is detected (default: None) (list of str)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

    """

    try:
        __sns.set(style="whitegrid", font="DejaVu Sans")
    except:
        pass

    if unique_samples == None:
        unique_samples = sorted(
            list(set([item for s in list(mutation_dataframe['sample_name'].unique()) for item in s.split(', ')])))

    ymax_SNV = mutation_dataframe[mutation_dataframe['type'] == 'SNV'].groupby(['sample_name']).count().max()['chr']
    ymax_INS = mutation_dataframe[mutation_dataframe['type'] == 'INS'].groupby(['sample_name']).count().max()['chr']
    ymax_DEL = mutation_dataframe[mutation_dataframe['type'] == 'DEL'].groupby(['sample_name']).count().max()['chr']
    ymax = [ymax_SNV, ymax_INS, ymax_DEL]
    ymax_all = [10 ** (len(str(ym))) for ym in ymax]

    mut_types_all = ['SNV', 'INS', 'DEL']

    unique_ploidies = sorted([int(i) for i in list(mutation_dataframe['ploidy'].unique())])

    color_dict_base = {'control': '#008B8B',
                       'treated': '#8B008B'}

    color_list = [color_dict_base['control'] if s in control_samples
                  else color_dict_base['treated'] for s in unique_samples]

    fig, axes = __plt.subplots(len(unique_ploidies), 3)
    fig.set_size_inches(21, 5 * len(unique_ploidies))
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                        hspace=0.4, wspace=0.2)
    for m in range(len(mut_types_all)):
        if (len(unique_ploidies) == 1):
            ymax = 10 ** len(str(mutation_dataframe[(mutation_dataframe['type'] == mut_types_all[m]) & (
                    mutation_dataframe['ploidy'] == unique_ploidies[0])].groupby(['sample_name']).count().max()['chr']))
            for s, c in zip(unique_samples, color_list):
                l = 'control samples' if s in control_samples else 'treated samples'
                score = mutation_dataframe[
                    (mutation_dataframe['type'] == mut_types_all[m]) & (mutation_dataframe['sample_name'] == s) & (
                            mutation_dataframe['ploidy'] == unique_ploidies[0])].sort_values(by='score')['score']
                axes[m].plot(score, len(score) - __np.arange(len(score)), c=c, label=l)
            axes[m].set_xlabel(r'score threshold', fontsize=12)
            axes[m].set_title(mut_types_all[m] + ' (ploidy: ' + str(unique_ploidies[0]) + ')\n', fontsize=14)
            axes[m].set_ylabel(r'Mutations found', fontsize=12)
            axes[m].set_ylim(1, ymax)
            axes[m].set_yscale('log')
            axes[m].set_xlim(0, mutation_dataframe['score'].max())
            # axes[i][m].grid()
            handles, labels = axes[m].get_legend_handles_labels()
            labels, ids = __np.unique(labels, return_index=True)
            handles = [handles[k] for k in ids]
            axes[m].legend(handles, labels, loc='upper right', fancybox=True)
        else:
            for i in range(len(unique_ploidies)):
                ymax = 10 ** len(str(mutation_dataframe[(mutation_dataframe['type'] == mut_types_all[m]) & (
                        mutation_dataframe['ploidy'] == unique_ploidies[i])].groupby(['sample_name']).count().max()[
                                         'chr']))
                for s, c in zip(unique_samples, color_list):
                    l = 'control samples' if s in control_samples else 'treated samples'
                    score = mutation_dataframe[
                        (mutation_dataframe['type'] == mut_types_all[m]) & (mutation_dataframe['sample_name'] == s) & (
                                mutation_dataframe['ploidy'] == unique_ploidies[i])].sort_values(by='score')['score']
                    axes[i][m].plot(score, len(score) - __np.arange(len(score)), c=c, label=l)
                axes[i][m].set_xlabel(r'score threshold', fontsize=12)
                axes[i][m].set_title(mut_types_all[m] + ' (ploidy: ' + str(unique_ploidies[i]) + ')\n', fontsize=14)
                axes[i][m].set_ylabel(r'Mutations found', fontsize=12)
                axes[i][m].set_ylim(1, ymax)
                axes[i][m].set_yscale('log')
                axes[i][m].set_xlim(0, mutation_dataframe['score'].max())
                # axes[i][m].grid()
                handles, labels = axes[i][m].get_legend_handles_labels()
                labels, ids = __np.unique(labels, return_index=True)
                handles = [handles[k] for k in ids]
                axes[i][m].legend(handles, labels, loc='upper right', fancybox=True)

    if (not return_string):
        __plt.show()
        __plt.close()
        return fig
    else:
        figfile = __BytesIO()
        __plt.savefig(figfile, bbox_inches='tight', format='png')
        __plt.close()
        figfile.seek(0)
        figdata_png = __base64.b64encode(figfile.getvalue())
        return figdata_png


def __plot_roc(mutation_dataframe, control_samples,
               FPs_per_genome, score0=0, plot_roc=True, return_string=False, unique_samples=None):
    """

    Optimizes score values for different mutation types (SNV, INS, DEL) and ploidies according to the list of control samples and the desired
    level of false positives in the genome.
    The results are stored in the score_lim_dict attribute of the MutationDetection object. If plot = True, plots ROC curves for all mutations
    types (SNV, INS, DEL) and all ploidies.

    :param mutation_dataframe: The dataframe containing the mutations. (pandas.DataFrame)
    :param control_samples: a subset of bam_filename (list of sample names) that should be considered as control samples. Control samples are defined as samples where no unique mutations are expected to be found. (list of str)
    :param FPs_per_genome: the largest number of false positives tolerated in a control sample (int)
    :param score0: Score optimization starts with score0. If a larger score value is likely to be optimal, setting score0 to a number larger than 0 can decrease computation time. (default: 0) (float)
    :param plot_roc: If True, ROC curves are plotted, otherwise score optimization is performed without generating any figures. (default: True) (bool)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
    :param unique_samples: list of unique samples where at least one mutation is detected (default: None) (list of str)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a matplotlib figure.

    """
    try:
        __sns.set(style="whitegrid", font="DejaVu Sans")
    except:
        pass

    steps = 50

    if unique_samples == None:
        unique_samples = sorted(
            list(set([item for s in list(mutation_dataframe['sample_name'].unique()) for item in s.split(', ')])))

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

    if plot_roc:
        fig, axes = __plt.subplots(len(unique_ploidies), 3)
        fig.set_size_inches(21, 5 * len(unique_ploidies))
        fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                            hspace=0.4, wspace=0.2)

    for m in range(len(mut_types_all)):
        if len(unique_ploidies) == 1:
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
            if (plot_roc):
                axes[m].step(fp, tp, c='#DB7093', lw=3)
                axes[m].set_title(mut_types_all[m] + ' (ploidy: ' + str(int(unique_ploidies[0])) + ')\n',
                                  fontsize=14)

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
                    if (plot_roc):
                        axes[m].plot(fps * 1e-6, tps * 1e-6, 'o', mec='#C71585', mfc='#C71585', ms=15, mew=3,
                                     label='score limit: ' + str(score_lim))
                        axes[m].text(0.95, 0.06, 'score limit: ' + str(score_lim),
                                     bbox={'facecolor': 'white', 'pad': 10}, verticalalignment='bottom',
                                     horizontalalignment='right', transform=axes[m].transAxes)
                else:
                    score_lim = 10000
                    if (plot_roc):
                        axes[m].text(0.95, 0.06, 'score limit: inf',
                                     bbox={'facecolor': 'white', 'pad': 10}, verticalalignment='bottom',
                                     horizontalalignment='right', transform=axes[m].transAxes)
                score_lim_dict[mut_types_all[m]].append(score_lim)
            if (plot_roc):
                axes[m].set_ylim(bottom=0)
                axes[m].set_xlim(left=0)
                axes[m].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                axes[m].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
                axes[m].set_xlabel('false positive rate 1/Mbp ', fontsize=12)
                axes[m].set_ylabel('mutation rate  1/Mbp ', fontsize=12)

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
                if plot_roc:
                    axes[i][m].step(fp, tp, c='#DB7093', lw=3)
                    axes[i][m].set_title(mut_types_all[m] + ' (ploidy: ' + str(int(unique_ploidies[i])) + ')\n',
                                         fontsize=14)

                if total_num_of_FPs_per_genome is not None:
                    fp_real = __np.array(fp_real)
                    tp_real = __np.array(tp_real)
                    if (len(tp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]]) > 0):
                        tps = tp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        fps = fp_real[fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        score_lim = \
                            __np.linspace(score0, mutation_dataframe[mutation_dataframe['type'] == mut_types_all[m]][
                                'score'].max(), steps)[
                                fp_real <= FPs_per_ploidy[mut_types_all[m]][unique_ploidies[i]]][0]
                        if (plot_roc):
                            axes[i][m].plot(fps * 1e-6, tps * 1e-6, 'o', mec='#C71585', mfc='#C71585', ms=15, mew=3,
                                            label='score limit: ' + str(score_lim))
                            axes[i][m].text(0.95, 0.06, 'score limit: ' + str(score_lim),
                                            bbox={'facecolor': 'white', 'pad': 10}, verticalalignment='bottom',
                                            horizontalalignment='right', transform=axes[i][m].transAxes)
                    else:
                        score_lim = 10000
                        if (plot_roc):
                            axes[i][m].text(0.95, 0.06, 'score limit: inf',
                                            bbox={'facecolor': 'white', 'pad': 10}, verticalalignment='bottom',
                                            horizontalalignment='right', transform=axes[i][m].transAxes)
                    score_lim_dict[mut_types_all[m]].append(score_lim)
                if plot_roc:
                    axes[i][m].set_ylim(bottom=0)
                    axes[i][m].set_xlim(left=0)
                    axes[i][m].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                    axes[i][m].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
                    axes[i][m].set_xlabel('false positive rate 1/Mbp ', fontsize=12)
                    axes[i][m].set_ylabel('mutation rate  1/Mbp ', fontsize=12)
            # axes[i][m].grid()

    if plot_roc:
        if not return_string:
            __plt.show()
            __plt.close()
            return score_lim_dict, fig
        else:
            figfile = __BytesIO()
            __plt.savefig(figfile, bbox_inches='tight', format='png')
            __plt.close()
            figfile.seek(0)
            figdata_png = __base64.b64encode(figfile.getvalue())
            return score_lim_dict, figdata_png
    else:
        return score_lim_dict, None


def plot_mutation_counts(sample_names=None,
                         mutations_dataframe=None,
                         unique_only=False,
                         return_string=False,
                         mutations_filename=None,
                         output_dir=None,
                         control_samples=None):
    """

    Plots the number of mutations found in all the samples in different ploidy regions.

    :param mutations_dataframe: The dataframe containing the mutations. (default: None) (pandas.DataFrame)
    :param sample_names: list of samples names to plot mutation counts for (default: None) (list of str)
    :param unique_only: If True, only unique mutations are plotted for each sample. (default: False) (boolean)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
    :param mutations_filename: The path to the file, where mutations are stored, if the mutations attribute of the object does not exist, its value will be set to the file defined here. (default: None) (str)
    :param output_dir: the path to the directory where mutation tables are located (default: None) (str)
    :param control_samples: List of sample names that should be used as control samples in the sense, that no unique mutations are expected in them. (The sample names listed here must match a subset of the sample names listed in bam_filename.) (list of str)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

    """

    try:
        __sns.set(style="whitegrid", font="DejaVu Sans")
    except:
        pass

    if mutations_dataframe is not None:
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    if control_samples is None:
        print('Warning: list of control samples not defined.')
        control_samples = []
    elif sum([1 for s in control_samples if s not in sample_names]) > 0:
        raise ValueError('List of "control_samples" is not a subset of "sample_names".')

    if unique_only:
        mutations_df = mutations_dataframe[~mutations_dataframe['sample_name'].str.contains(',')]
    else:
        mutations_df = mutations_dataframe

    unique_samples = sorted(sample_names)

    pos = list(range(len(unique_samples)))
    unique_ploidies = sorted(list(mutations_df['ploidy'].unique()))
    width = 1. / (len(unique_ploidies) + 1)

    color_dict_base = {'control': '#008B8B',
                       'treated': '#8B008B'}

    color_list = [color_dict_base['control'] if s in control_samples
                  else color_dict_base['treated'] for s in unique_samples]

    mut_types_all = ['SNV', 'INS', 'DEL']

    # Plotting the bars
    fig, axes = __plt.subplots(len(mut_types_all), 1)
    fig.set_size_inches(14, 6 * len(mut_types_all))
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                        hspace=0.4, wspace=0.2)

    for m in range(len(mut_types_all)):
        for i in range(len(unique_ploidies)):
            filtered_table = mutations_df[
                (mutations_df['type'] == mut_types_all[m]) & (mutations_df['ploidy'] == unique_ploidies[i])]
            count_list = []
            for s in unique_samples:
                count_list.append(filtered_table[filtered_table['sample_name'].str.contains(s)].shape[0])
            sample_counts = __pd.DataFrame()
            sample_counts['sample'] = unique_samples
            sample_counts['count'] = count_list

            sample_counts.sort_values(by='sample')

            if len(unique_ploidies) == 1:
                a = 1
            else:
                a = 1 - (i + 1) * (1. / len(unique_ploidies))
            barlist = axes[m].bar([p + i * width for p in pos], sample_counts['count'], width, alpha=a,
                                  color="violet", label=str(int(unique_ploidies[i])))
            for j in range(len(barlist)):
                barlist[j].set_color(color_list[j])

        # Set the y axis label
        axes[m].set_ylabel('Mutation count', fontsize=12)

        # Set the chart's title
        if unique_only and len(unique_ploidies) == 1:
            title = '\nUnique ' + mut_types_all[m] + ' counts\n'
        elif not unique_only and len(unique_ploidies) == 1:
            title = '\nAll ' + mut_types_all[m] + ' counts\n'
        if unique_only and len(unique_ploidies) > 1:
            title = '\nUnique ' + mut_types_all[m] + ' counts grouped by ploidy\n'
        elif not unique_only and len(unique_ploidies) > 1:
            title = '\nAll ' + mut_types_all[m] + ' counts grouped by ploidy\n'
        axes[m].set_title(title, fontsize=14)

        # Set the position of the x ticks
        axes[m].set_xticks([p - 0.5 + len(unique_ploidies) * 1 * width for p in pos])
        # Set the labels for the x ticks
        sample_labels = ['\n'.join([s[i:i + 10] for i in range(0, len(s), 10)]) for s in unique_samples]
        fs = 8
        axes[m].set_xticklabels(sample_labels, rotation=90, fontsize=fs)

        # Setting the x-axis and y-axis limits
        axes[m].set_xlim(min([p - 1 + len(unique_ploidies) * 0.5 * width for p in pos]),
                         max(pos) + width * (len(unique_ploidies) + 1))
        axes[m].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        axes[m].set_yscale('log')
        # axes[m].grid()

    if return_string:
        figfile = __BytesIO()
        __plt.savefig(figfile, bbox_inches='tight', format='png')
        __plt.close()
        figfile.seek(0)
        figdata_png = __base64.b64encode(figfile.getvalue())
        return figdata_png

    else:
        __plt.show()
        __plt.close()
        return fig


def plot_hierarchical_clustering(sample_names=None,
                                 mutations_dataframe=None,
                                 mutations_filename=None,
                                 output_dir=None,
                                 return_string=False,
                                 method='average'):
    """

    Generates a heatmap based on the number of shared mutations found in all possible sample pairs.
    A dendrogram is also added that is the result of hierarchical clustering of the samples.

    :param mutations_dataframe: The dataframe containing the mutations. (default: None) (pandas.DataFrame)
    :param sample_names: list of samples names to plot mutation counts for (default: None) (list of str)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
    :param mutations_filename: The path to the file, where mutations are stored, if the mutations attribute of the object does not exist, its value will be set to the file defined here. (default: None) (str)
    :param output_dir: the path to the directory where mutation tables are located (default: None) (str)
    :param method: method used for seaborn hierarchical clustering (default: 'average') ("single", "complete", "average", "weighted", "median", "ward")

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

    """

    if mutations_dataframe is not None:
        if not mutations_dataframe.__class__ == __pd.core.frame.DataFrame:
            msg = 'Error: Argument "mutations" is not a pandas DataFrame.'
            raise ValueError(msg)
        elif sorted(list(mutations_dataframe.columns)) != sorted(['sample_name', 'chr', 'pos', 'type', 'score',
                                                                  'ref', 'mut', 'cov', 'mut_freq', 'cleanliness',
                                                                  'ploidy']):
            msg = 'Error: The DataFrame supplied in argument "mutations" does not have the required columns.'
            msg += '\n'
            msg += 'Make sure to have the following columns: sample_name, chr, pos, type, score, ' \
                   'ref, mut, cov, mut_freq, cleanliness, ploidy'
            raise ValueError(msg)
    else:
        mutations_dataframe = io.load_mutations(output_dir=output_dir,
                                                filename=mutations_filename)

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    if len(sample_names) < 3:
        raise ValueError('Hierarchical clustering cannot be performed on less than 3 samples.')

    c = __np.zeros((len(sample_names), len(sample_names)))
    for i in range(len(sample_names)):
        for j in range(i + 1):
            c[i][j] = mutations_dataframe[(mutations_dataframe["sample_name"].str.contains(sample_names[i])) & (
                mutations_dataframe["sample_name"].str.contains(sample_names[j]))].shape[0]
            c[j][i] = c[i][j]

    d = __pd.DataFrame(c)
    d.columns = sample_names
    d.index = sample_names

    g = __sns.clustermap(d, method=method, cmap="viridis", robust=True);

    if return_string:
        figfile = __BytesIO()
        g.savefig(figfile, bbox_inches='tight', format='png')
        __plt.close()
        figfile.seek(0)
        figdata_png = __base64.b64encode(figfile.getvalue())
        return figdata_png
    else:
        __plt.show()
        __plt.close()
        return g


def plot_SNV_spectrum(spectrumDict,
                      return_string=False, normalize_to_1=False):
    """

    Plots the triplet spectrum for a list of 96-element vectors defined in spectrumDict.

    :param spectrumDict: a dictionary containing spectra as values and sample names as keys (dictionary)
    :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

    """

    try:
        __sns.set(style="white", font="DejaVu Sans")
    except:
        pass

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

    mut_xlabel = [i[0] + i[1] + i[2] for i in all_muttypes[::2]]
    mut_title = ['C > A', 'C > G', 'C > T', 'T > A', 'T > C', 'T > G']

    ind = __np.arange(16)
    width = 0.4

    spectrum_colors = []
    spectrum_colors.append('#03bcee')
    spectrum_colors.append('#010101')
    spectrum_colors.append('#e32926')
    spectrum_colors.append('#999999')
    spectrum_colors.append('#a1ce63')
    spectrum_colors.append('#ebc6c4')

    figs = []

    for sample_name, spectrum_orig in spectrumDict.items():

        if normalize_to_1:
            spectrum = __np.array(spectrum_orig) / __np.sum(spectrum_orig)
        else:
            spectrum = spectrum_orig

        f, axarr = __plt.subplots(1, 6, sharey=True)
        f.set_size_inches(20, 3)

        for i in range(6):
            axarr[i].bar(ind, spectrum[i * 16:(i + 1) * 16], width, color=spectrum_colors[i],
                         edgecolor=spectrum_colors[i])
            axarr[i].xaxis.set_ticks(__np.linspace(0, 16, 17))
            axarr[i].set_xticks(ind + width)
            axarr[i].xaxis.set_tick_params(size=0)
            axarr[i].yaxis.set_tick_params(size=0)
            axarr[i].set_xticklabels(mut_xlabel[i * 16:(i + 1) * 16], rotation='vertical', size=9)
            axarr[i].text(0.5, 1.08, mut_title[i], size=12, ha="center", transform=axarr[i].transAxes)

            axarr[i].add_patch(
                __Rectangle((0, 1.01), 1, 0.05, color=spectrum_colors[i], transform=axarr[i].transAxes,
                            clip_on=False))
            axarr[i].spines['right'].set_visible(False)
            axarr[i].spines['top'].set_visible(False)
            axarr[i].set_ylim(0, max(spectrum) * 1.1)
            if i == 0:
                axarr[i].set_ylabel(sample_name + '\n', fontsize=12)
            if i != 0:
                axarr[i].spines['left'].set_visible(False)

        if normalize_to_1:
            vals = axarr[0].get_yticks()
            axarr[0].set_yticklabels(['{:,.1%}'.format(x) for x in vals])

        if return_string:
            figfile = __BytesIO()
            f.savefig(figfile, bbox_inches='tight', format='png')
            __plt.close()
            figfile.seek(0)
            figdata_png = __base64.b64encode(figfile.getvalue())
            figs.append(figdata_png)
        else:
            __plt.show()
            figs.append(f)
            __plt.close(f)

    return figs


def plot_indel_spectrum(spectrumDict, return_string=False, normalize_to_1=False):
    """

    Plots the indel spectrum, given a dictionary containing 83-element vectors as values.

    :param spectrumDict: a dictionary containing spectra as values and sample names as keys (dictionary)
    :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: True) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

    """
    try:
        __sns.set(style="white", font="DejaVu Sans")
    except:
        pass

    mut_xlabel = ['1', '2', '3', '4', '5', '6+'] * 2 + ['0', '1', '2', '3', '4', '5+'] * 2
    mut_xlabel += ['1', '2', '3', '4', '5', '6+'] * 4 + ['0', '1', '2', '3', '4', '5+'] * 4
    mut_xlabel += ['1', '1', '2', '1', '2', '3', '1', '2', '3', '4', '5+']

    mut_title = ['C', 'T', 'C', 'T'] + ['2', '3', '4', '5+'] * 3

    subplot_sizes = [6] * 12 + [1, 2, 3, 5]
    width = 0.4

    spectrum_colors = []
    spectrum_colors.append('#fdbe6f')
    spectrum_colors.append('#ff8001')
    spectrum_colors.append('#b0dd8b')
    spectrum_colors.append('#36a12e')
    spectrum_colors.append('#fdcab5')
    spectrum_colors.append('#fc8a6a')
    spectrum_colors.append('#f14432')
    spectrum_colors.append('#bc141a')
    spectrum_colors.append('#d0e1f2')
    spectrum_colors.append('#94c4df')
    spectrum_colors.append('#4a98c9')
    spectrum_colors.append('#1764ab')
    spectrum_colors.append('#e2e2ef')
    spectrum_colors.append('#b6b6d8')
    spectrum_colors.append('#8683bd')
    spectrum_colors.append('#61409b')

    text_color = ['black', 'white', 'black', 'white'] + ['black', 'black', 'black', 'white'] * 3

    figs = []

    for sample_name, spectrum_orig in spectrumDict.items():
        if __np.sum(spectrum_orig) == 0:
            print('No indels were found in sample ' + sample_name)
        else:
            if normalize_to_1:
                id_spectrum = __np.array(spectrum_orig) / __np.sum(spectrum_orig)
            else:
                id_spectrum = spectrum_orig

            f, axarr = __plt.subplots(1, len(subplot_sizes), sharey=True, gridspec_kw={'width_ratios': subplot_sizes})
            f.set_size_inches(20, 3)

            for i in range(len(subplot_sizes)):
                ind = __np.arange(subplot_sizes[i])
                start_pos = __np.int(__np.sum(subplot_sizes[:i]))
                end_pos = start_pos + subplot_sizes[i]
                axarr[i].bar(ind, id_spectrum[start_pos:end_pos], width, color=spectrum_colors[i],
                             edgecolor=spectrum_colors[i])
                axarr[i].xaxis.set_ticks(__np.linspace(0, subplot_sizes[i], subplot_sizes[i] + 1))
                axarr[i].set_xticks(ind + width)
                axarr[i].xaxis.set_tick_params(size=0)
                axarr[i].yaxis.set_tick_params(size=0)
                axarr[i].set_xticklabels(mut_xlabel[start_pos:end_pos], rotation='horizontal', size=9)
                axarr[i].text(0.5, 1.05, mut_title[i], size=12, ha="center", transform=axarr[i].transAxes,
                              color=text_color[i])

                axarr[i].add_patch(
                    __Rectangle((0, 1.03), 1, 0.08, color=spectrum_colors[i], transform=axarr[i].transAxes,
                                clip_on=False))

                axarr[i].spines['right'].set_visible(False)
                axarr[i].spines['top'].set_visible(False)
                axarr[i].set_ylim(0, max(id_spectrum) * 1.1)
                if i == 0:
                    axarr[i].set_ylabel(sample_name + '\n', fontsize=12)
                if i != 0:
                    axarr[i].spines['left'].set_visible(False)

            if normalize_to_1:
                vals = axarr[0].get_yticks()
                axarr[0].set_yticklabels(['{:,.1%}'.format(x) for x in vals])

            axarr[0].text(1.1, 1.2, "1 bp deletion", size=14, ha="center", transform=axarr[0].transAxes)
            axarr[0].text(1.1, -0.2, "homopolymer length", size=13, ha="center", transform=axarr[0].transAxes)
            axarr[2].text(1.1, 1.2, "1 bp insertion", size=14, ha="center", transform=axarr[2].transAxes)
            axarr[2].text(1.1, -0.2, "homopolymer length", size=13, ha="center", transform=axarr[2].transAxes)
            axarr[5].text(1.1, 1.2, ">1 bp deletions at repeats\n(deletion length)", size=14, ha="center",
                          transform=axarr[5].transAxes)
            axarr[5].text(1.1, -0.2, "number of repeat units", size=13, ha="center", transform=axarr[5].transAxes)
            axarr[9].text(1.1, 1.2, ">1 bp insertions at repeats\n(insertion length)", size=14, ha="center",
                          transform=axarr[9].transAxes)
            axarr[9].text(1.1, -0.2, "number of repeat units", size=13, ha="center", transform=axarr[9].transAxes)
            axarr[14].text(0.7, 1.2, "deletions with microhomology\n(deletion length)", size=14, ha="center",
                           transform=axarr[14].transAxes)
            axarr[14].text(0.7, -0.2, "microhomology length", size=13, ha="center", transform=axarr[14].transAxes)

            if return_string:
                figfile = __BytesIO()
                f.savefig(figfile, bbox_inches='tight', format='png')
                __plt.close()
                figfile.seek(0)
                figdata_png = __base64.b64encode(figfile.getvalue())
                figs.append(figdata_png)
            else:
                __plt.show()
                figs.append(f)
                __plt.close(f)

    return figs


def plot_DNV_heatmap(matrixDict, return_string=False, normalize_to_1=False):
    """

    Plot DNVs (dinucleotide variations) as a heatmap for a database of mutations.

    :param matrixDict: a dictionary containing 12x12 element matrices as values and sample names as keys (dictionary)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
    :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, nothing.

    """

    base_changes = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G', 'A>C', 'A>G', 'A>T', 'G>A', 'G>C', 'G>T']

    figs = []

    for sample_name, matrix in matrixDict.items():

        if __np.sum(matrix) == 0:
            print('No DNVs were found in sample ' + sample_name)
        else:
            if normalize_to_1:
                matrix_to_plot = matrix / __np.sum(matrix)
            else:
                matrix_to_plot = matrix
            df_m = __pd.DataFrame(matrix_to_plot, columns=base_changes, index=base_changes)
            colormap = __plt.cm.YlGnBu
            mask = __np.zeros_like(matrix_to_plot)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if i >= mask.shape[0] - j:
                        mask[i, j] = True
            with __sns.axes_style("white"):
                if not normalize_to_1:
                    ax = __sns.heatmap(df_m, square=True, mask=mask, cmap=colormap, annot=True)
                else:
                    ax = __sns.heatmap(df_m, square=True, mask=mask, cmap=colormap, annot=False)

            __plt.xlabel("\n3' base change", fontsize=14)
            __plt.ylabel("5' base change\n", fontsize=14)
            __plt.title(sample_name + '\n', fontsize=16);

            if return_string:
                figfile = __BytesIO()
                __plt.savefig(figfile, bbox_inches='tight', format='png')
                __plt.close()
                figfile.seek(0)
                figdata_png = __base64.b64encode(figfile.getvalue())
                figs.append(figdata_png)

            else:
                f = __plt.gcf()
                figs.append(f)
                __plt.show()
                __plt.close()

    return figs


def plot_DNV_spectrum(spectrumDict, return_string=False, normalize_to_1=False):
    """

    Plots the DNV spectrum, given a dictionary containing the spectra as values.

    :param spectrumDict: a dictionary containing DNV spectra as values and sample names as keys (dictionary)
    :param normalize_to_1: If True, results are plotted as percentages, instead of counts. (default: False) (bool)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

    """
    try:
        __sns.set(style="white", font="DejaVu Sans")
    except:
        pass

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

    mut_xlabel = [k.split('>')[1] for k in DBS_types]
    mut_title = ['AC>NN', 'AT>NN', 'CC>NN', 'CG>NN', 'CT>NN', 'GC>NN', 'TA>NN', 'TC>NN', 'TG>NN', 'TT>NN']

    subplot_sizes = [9, 6, 9, 6, 9, 6, 6, 9, 9, 9]
    width = 0.4

    spectrum_colors = []
    spectrum_colors.append('#a6cee3')
    spectrum_colors.append('#1f78b4')
    spectrum_colors.append('#b2df8a')
    spectrum_colors.append('#33a02c')
    spectrum_colors.append('#fb9a99')
    spectrum_colors.append('#e31a1c')
    spectrum_colors.append('#fdbf6f')
    spectrum_colors.append('#ff7f00')
    spectrum_colors.append('#cab2d6')
    spectrum_colors.append('#6a3d9a')

    figs = []

    for sample_name, spectrum_orig in spectrumDict.items():
        if __np.sum(spectrum_orig) == 0:
            print('No DNVs were found in sample ' + sample_name)
        else:
            if normalize_to_1:
                spectrum = __np.array(spectrum_orig) / __np.sum(spectrum_orig)
            else:
                spectrum = spectrum_orig

            f, axarr = __plt.subplots(1, len(subplot_sizes), sharey=True, gridspec_kw={'width_ratios': subplot_sizes})
            f.set_size_inches(20, 3)

            for i in range(len(subplot_sizes)):
                ind = __np.arange(subplot_sizes[i])
                start_pos = __np.int(__np.sum(subplot_sizes[:i]))
                end_pos = start_pos + subplot_sizes[i]
                axarr[i].bar(ind, spectrum[start_pos:end_pos], width, color=spectrum_colors[i],
                             edgecolor=spectrum_colors[i])
                axarr[i].xaxis.set_ticks(__np.linspace(0, subplot_sizes[i], subplot_sizes[i] + 1))
                axarr[i].set_xticks(ind + width)
                axarr[i].xaxis.set_tick_params(size=0)
                axarr[i].yaxis.set_tick_params(size=0)
                axarr[i].set_xticklabels(mut_xlabel[start_pos:end_pos], rotation='vertical', size=9)
                axarr[i].text(0.5, 1.08, mut_title[i], size=12, ha="center", transform=axarr[i].transAxes)

                axarr[i].add_patch(
                    __Rectangle((0, 1.01), 1, 0.05, color=spectrum_colors[i], transform=axarr[i].transAxes,
                                clip_on=False))
                axarr[i].spines['right'].set_visible(False)
                axarr[i].spines['top'].set_visible(False)
                axarr[i].set_ylim(0, max(spectrum) * 1.1)
                if i == 0:
                    axarr[i].set_ylabel(sample_name + '\n', fontsize=12)
                if i != 0:
                    axarr[i].spines['left'].set_visible(False)

            if normalize_to_1:
                vals = axarr[0].get_yticks()
                axarr[0].set_yticklabels(['{:,.1%}'.format(x) for x in vals])

            if return_string:
                figfile = __BytesIO()
                f.savefig(figfile, bbox_inches='tight', format='png')
                __plt.close()
                figfile.seek(0)
                figdata_png = __base64.b64encode(figfile.getvalue())
                figs.append(figdata_png)
            else:
                __plt.show()
                figs.append(f)
                __plt.close(f)

    try:
        __sns.set_style("whitegrid")
    except:
        pass

    return figs


def __plot_spectrum_decomposition(sample_names, theta_matrix, sig_names, spectrum_type=None):
    """

    Plot the final set of proportions for the signatures.

    :param sample_names: The list of sample names analysed. (list of str)
    :param theta_matrix: The final set of mixture proportions for each sample. (numpy.array)
    :param sig_names: The names of the signatures. (numpy.array)
    :param spectrum_type: The type of mutations analysed ("SNV", "DNV", "indel"). (default: None) (str)

    """

    try:
        __sns.set_style("white")
    except:
        pass

    ind = __np.arange(len(sig_names))
    width = 0.4

    c = '#a6cee3'

    if spectrum_type == 'DNV':
        c = '#b2df8a'
    elif spectrum_type == 'SNV':
        c = '#fdbf6f'
    elif spectrum_type == 'indel':
        c = '#cab2d6'

    figs = []

    for sID, sample_name in enumerate(sample_names):
        f, ax = __plt.subplots()
        f.set_size_inches(20, 3)

        ax.bar(ind, theta_matrix[sID], width, color=c, edgecolor=c)
        ax.xaxis.set_ticks(__np.linspace(0, len(sig_names), len(sig_names) + 1))
        ax.set_xticks(ind)
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)
        ax.set_xticklabels(sig_names, rotation='vertical', size=12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_ylabel(sample_name + '\n', fontsize=12)

        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

        for vID, v in enumerate(theta_matrix[sID]):
            if (v > 0):
                if (len(sig_names) < 25):
                    ax.text(ind[vID], v + 0.05, str(round(v * 100, 2)) + '%', size=10, horizontalalignment='center')
                else:
                    ax.text(ind[vID], v + 0.05, str(round(v * 100, 2)) + '%', size=10,
                            rotation='vertical', horizontalalignment='center',
                            verticalalignment='bottom')

        __plt.show()
        figs.append(f)
        __plt.close(f)

    return figs


def __get_rainfall_data_for_sample(mutations_dataframe, chromosomes, chrom_length):
    """

    :param mutations_dataframe: a pandas DataFrame containing mutations (pandas.DataFrame)
    :param chromosomes: a list of chromosomes for the genome (list of str)
    :param chrom_length: a list of chromosome lengths (list of int)

    :returns: a dictionary containing points to plot for each mutation type

    """
    mutation_types = ['C > A', 'G > T', 'C > G', 'G > C', 'C > T', 'G > A', 'T > A', 'A > T', 'T > C', 'A > G', 'T > G',
                      'A > C']

    mutation_type_dict = {i: [] for i in range(8)}

    for chrom in list(mutations_dataframe['chr'].unique()):
        abs_chrom_pos = sum(chrom_length[:chromosomes.index(chrom)])
        if mutations_dataframe[mutations_dataframe['chr'] == chrom].shape[0] > 1:
            pos_before = -42
            for rID, r in mutations_dataframe[mutations_dataframe['chr'] == chrom].sort_values(by='pos').iterrows():
                if pos_before != -42:
                    if r['type'] == 'SNV':
                        muttype = int(__np.floor(mutation_types.index(r['ref'] + ' > ' + r['mut']) / 2))
                    elif r['type'] == 'INS':
                        muttype = 6
                    elif r['type'] == 'DEL':
                        muttype = 7
                    mutation_type_dict[muttype].append([abs_chrom_pos + r['pos'], r['pos'] - pos_before])
                pos_before = r['pos']

    return mutation_type_dict


def plot_rainfall(mutations_dataframe,
                  chromosomes=None, chrom_length=None, ref_fasta=None,
                  sample_names=None,
                  return_string=False,
                  muttypes=['SNV', 'INS', 'DEL'],
                  unique_only=True,
                  plot_range=None):
    """

    Plot a rainfall plot of the mutations. The horizontal axis is the genomic position of each mutation and the vertical axis is the genomic
    difference measured from the previous mutation.

    :param mutations_dataframe: the pandas.DataFrame containing all mutations (pandas.DataFrame)
    :param chromosomes: a list of chromosomes to be plotted (default: None) (list of str)
    :param chrom_length: a list of chrom lengths (default: None) (list of int)
    :param ref_fasta: the path to the reference fasta file (default: None) (str)
    :param sample_names: the list of sample names to be plotted (default: None) (str)
    :param return_string: If True, only a temporary plot is generated and its base64 code is returned, that can be included in HTML files. (default: False) (bool)
    :param muttypes: the list of mutation types to be plotted (default: ["SNV", "INS", "DEL"]) (any elements of the default list)
    :param unique_only: If True, only unique mutations are plotted for each sample. (default: False) (boolean)
    :param plot_range: the genomic range to be plotted (default: None, the whole genome is plotted) (str, example: "chr9:123134-143441414")

    :returns: If the return_string value is True, a base64 encoded string of the image. Otherwise, a list of matplotlib figures.

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
        if len(chrom_list) > 0:
            return chrom_list, [newlens[newchroms.index(c)] for c in chrom_list]
        else:
            return sorted(newchroms), [newlens[newchroms.index(c)] for c in sorted(newchroms)]

    try:
        __sns.set(style="white", font="DejaVu Sans")
    except:
        pass

    if chromosomes is None or chrom_length is None:
        if ref_fasta is None or not __os.path.isfile(ref_fasta):
            raise ValueError('Error: supply either "chromosomes" AND "chrom_length" or "ref_fasta" arguments.')
        else:
            if not __os.path.isfile(ref_fasta + '.fai'):
                error_msg = 'Error: No faidx file found for reference genome file "' + ref_fasta + '", cannot proceed.'
                error_msg += '\n'
                error_msg += 'Use the samtools command: samtools faidx [ref.fasta]'
                raise ValueError(error_msg)
            else:
                if chromosomes is None:
                    chromosomes = sorted(list(mutations_dataframe['chr'].unique()))
                chromosomes, chrom_length = get_default_chroms_with_len(ref_fasta, chromosomes)

    spectrum_colors = ['#03bcee', '#010101', '#e32926', '#999999', '#a1ce63', '#ebc6c4', '#79159e', '#ffcc00']
    mutation_types = ['C > A', 'G > T', 'C > G', 'G > C', 'C > T', 'G > A', 'T > A', 'A > T', 'T > C', 'A > G', 'T > G',
                      'A > C']

    if plot_range is not None:
        range_chr = plot_range.split(':')[0]
        if ':' in plot_range:
            range_posmin = int(plot_range.split(':')[1].split('-')[0])
            range_posmax = int(plot_range.split(':')[1].split('-')[1])
        else:
            range_posmin = 0
            range_posmax = chrom_length[chromosomes.index(range_chr)]
        if range_posmin == range_posmax:
            raise ValueError('Error: unreadable range format. Make sure to set the range either by chromosome '
                             '(eg. "chr9") or by specific region (eg. "chr9:164574-345346").')

    if unique_only:
        mutations_dataframe = mutations_dataframe[~mutations_dataframe['sample_name'].str.contains(',')]

    if sample_names is None:
        sample_names = sorted(
            list(set([item for s in list(mutations_dataframe['sample_name'].unique()) for item in s.split(',')])))

    figs = []

    for sample in sample_names:
        mutation_types_dict = __get_rainfall_data_for_sample(
            mutations_dataframe[mutations_dataframe['sample_name'].str.contains(sample)],
            chromosomes, chrom_length)

        total_num_of_muts = 0
        if 'SNV' in muttypes:
            total_num_of_muts += sum([len(mutation_types_dict[k]) for k in range(6)])
        if 'INS' in muttypes:
            total_num_of_muts += len(mutation_types_dict[6])
        if 'DEL' in muttypes:
            total_num_of_muts += len(mutation_types_dict[7])

        if total_num_of_muts < 2:
            print('There are not enough mutations in sample ' + sample + ' for a rainfall plot to be created.')
            continue

        f, ax = __plt.subplots()
        f.set_size_inches(20, 10)

        max_value = 0

        plots = []
        legend_titles = []

        zeropoint = 0
        if plot_range is not None:
            zeropoint = sum(chrom_length[:chromosomes.index(range_chr)])

        if 'SNV' in muttypes:
            for i in range(6):
                plots.append(
                    __plt.scatter(__np.array([u[0] for u in mutation_types_dict[i]]) - zeropoint,
                                  [u[1] for u in mutation_types_dict[i]],
                                  c=spectrum_colors[i], edgecolor=spectrum_colors[i], s=50))
                legend_titles.append(mutation_types[i * 2])
                max_value = __np.max([max_value, __np.max([u[1] for u in mutation_types_dict[i]])])

        if 'INS' in muttypes:
            plots.append(__plt.scatter(__np.array([u[0] for u in mutation_types_dict[6]]) - zeropoint,
                                       [u[1] for u in mutation_types_dict[6]],
                                       c=spectrum_colors[6], edgecolor=spectrum_colors[6], s=50))
            legend_titles.append('INS')
            max_value = __np.max([max_value, __np.max([u[1] for u in mutation_types_dict[6]])])
        if 'DEL' in muttypes:
            plots.append(__plt.scatter(__np.array([u[0] for u in mutation_types_dict[7]]) - zeropoint,
                                       [u[1] for u in mutation_types_dict[7]],
                                       c=spectrum_colors[7], edgecolor=spectrum_colors[7], s=50))
            legend_titles.append('DEL')
            max_value = __np.max([max_value, __np.max([u[1] for u in mutation_types_dict[7]])])

        ax.set_yscale('log')
        ax.yaxis.grid(True, c='lightgrey', lw=1.5, linestyle='dotted')
        if plot_range is not None:
            ax.set_xlim(range_posmin, range_posmax)
        else:
            ax.set_xlim(0, sum(chrom_length))
        ax.set_ylim(1, max_value * 1.1)
        ax.legend(tuple(plots),
                  tuple(legend_titles),
                  loc='center left', scatterpoints=1, bbox_to_anchor=(1, 0.5))

        if plot_range is None:
            for i in range(len(chrom_length)):
                __plt.plot((sum(chrom_length[:i]), sum(chrom_length[:i])), (1, 10e9), 'lightgray')
                ax.text((sum(chrom_length[:(i)]) + __np.float(chrom_length[i]) / 2) / sum(chrom_length), -0.1,
                        chromosomes[i],
                        verticalalignment='bottom', horizontalalignment='center',
                        transform=ax.transAxes,
                        color='gray', fontsize=12, rotation='vertical')
        else:
            ax.text(0.5, -0.1,
                    range_chr,
                    verticalalignment='bottom', horizontalalignment='center',
                    transform=ax.transAxes,
                    color='gray', fontsize=12)

        ax.set_ylabel('genomic distance of each mutation from the previous one\n', size=15)
        ax.set_xlabel('\n\n\ngenomic position of mutation', size=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.set_title(sample + '\n', size=15)
        if return_string:
            figfile = __BytesIO()
            f.savefig(figfile, bbox_inches='tight', format='png')
            __plt.close()
            figfile.seek(0)
            figdata_png = __base64.b64encode(figfile.getvalue())
            figs.append(figdata_png)
        else:
            figs.append(f)
            __plt.show()
            __plt.close(f)

    return figs
