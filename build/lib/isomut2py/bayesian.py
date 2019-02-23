try:
    from isomut2py import io

    import theano.tensor as __tt
    import numpy as __np
    import pymc3 as __pm
    import pandas as __pd
    from scipy import stats as __stats
    import gc as __gc
    import logging as __logging
except ImportError:
    print('ImportError in isomut2py.ploidyEstBayesian, Bayesian estimate functions will not work.')


def estimate_hapcov_infmix(cov_min=None, hc_percentile=None,
                           chromosomes=None, output_dir=None, cov_max=None, level=0, cov_sample=None,
                           user_defined_hapcov=None):
    """

    Estimates the haploid coverage of the sample.
    If the user_defined_hapcov attribute is set manually, it sets the value of estimated_hapcov to that.
    Otherwise, a many-component (20) Gaussian mixture model is fitted to the coverage histogram of the sample 10 times. Each time,
    the haploid coverage is estimated from the center of the component with the maximal weight in the model. The final estimate of the haploid
    coverage is calculated as the qth percentile of the 10 measurements, with q = hc_percentile.

    :param user_defined_hapcov: if not None, its value is returned (default: None) (float)
    :param cov_sample: a sample of the coverage distribution of the investigated sample, if None, it is loaded from the temporary files of the output_dir (default: None) (array-like)
    :param cov_max: the maximum value of the coverage for a position to be considered in the estimation (default: None) (int)
    :param output_dir: the path to the output directory of the PloidyEstimator object, where temporary files are located (default: None) (str)
    :param chromosomes: list of chromosomes for the sample (default: None) (array-like)
    :param hc_percentile: the percentile value to use for calculating the estimated haploid coverage from 10 subsequent estimations (default: None) (int)
    :param cov_min: the maximum value of the coverage for a position to be considered in the estimation (default: None) (int)
    :param level: the level of indentation used in verbose output (default: 0) (int)

    :returns: (estimated haploid coverage (float), sample from coverage distribution (array-like))

    """

    # if a sample of the coverage distribution is not provided, generate one
    if cov_sample is None:
        cov_sample = io.get_coverage_distribution(chromosomes=chromosomes,
                                                  output_dir=output_dir,
                                                  cov_max=cov_max,
                                                  cov_min=cov_min)

    # if the user has set the haploid coverage manually, use that
    if user_defined_hapcov is not None:
        return user_defined_hapcov, cov_sample


    # fit a finite mixture model in 10 chains with pymc3
    K = 20  # number of mixture components
    number_of_chains = 10  # number of chains to use
    iterations = 15000  # number of iterations
    burn_beginning = 10000  # number of iterations to ignore from the beginning (where it might not have converged yet)

    def stick_breaking(beta):
        portion_remaining = __tt.concatenate([[1], __tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    def fit_infinite_mixture_model(coverage_dist, K, number_of_chains_to_use, number_of_iterations, burn_period):
        covdist_standard = (coverage_dist - coverage_dist.mean()) / coverage_dist.std()
        N = len(covdist_standard)

        __gc.collect()

        with __pm.Model() as model:
            alpha = __pm.Gamma('alpha', 1., 1.)
            beta = __pm.Beta('beta', 1., alpha, shape=K)
            w = __pm.Deterministic('w', stick_breaking(beta))

            tau = __pm.Gamma('tau', 1., 1., shape=K)
            lambda_ = __pm.Uniform('lambda', 0, 5, shape=K)
            mu = __pm.ExGaussian('mu', mu=-4, sigma=__np.sqrt(1 / (lambda_ * tau)), nu=5, shape=K)
            obs = __pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                                     observed=covdist_standard)

        # logger = __logging.getLogger("pymc3")
        # logger.propagate = False

        __logging.getLogger("pymc3").setLevel(__logging.WARNING)

        with model:
            step1 = __pm.Metropolis(vars=[alpha, beta, tau, lambda_, mu], verbose=0)

            tr = __pm.sample(draw=number_of_iterations-burn_period, tune=burn_period,
                             step=[step1], njobs=number_of_chains_to_use,
                             progressbar=False, verbose=0, compute_convergence_checks=False)

        # trace = tr[burn_period:]
        # return trace
        return tr


    trace = fit_infinite_mixture_model(coverage_dist=cov_sample,
                                       K=K,
                                       number_of_chains_to_use=number_of_chains,
                                       number_of_iterations=iterations,
                                       burn_period=burn_beginning)

    chains_to_use = [c for c in range(number_of_chains)]
    hc_all_ests = []
    for c in chains_to_use:
        comp_weights = trace.get_values('w', chains=[c]).mean(axis=0)
        comp_weights_sortidx = __np.argsort(comp_weights)
        standard_means = trace.get_values('mu', chains=[c]).mean(axis=0)[comp_weights_sortidx]
        true_means = standard_means * cov_sample.std() + cov_sample.mean()
        hc_maxweight = true_means[0] / (__np.round(true_means[0] / true_means[true_means > cov_min].min()))
        hc_all_ests.append(hc_maxweight)

    hc_all_ests = __np.array(hc_all_ests)
    hc_maxweight = __np.percentile(a=hc_all_ests[__np.isfinite(hc_all_ests)], q=hc_percentile)

    del trace

    return hc_maxweight, cov_sample


def fit_gaussians(estimated_hapcov,
                  chromosomes=None, output_dir=None, cov_max=None, cov_min=None, level=0, cov_sample=None):
    """

    Fits a 7-component Gaussian mixture model to the coverage distribution of the sample, using the appropriate attributes of the PloidyEstimation
    object. The center of the first Gaussian is initialized from a narrow region around the value of the estimated_hapcov attribute. The centers of
    the other Gaussians are initialized in a region around the value of estimated_hapcov multiplied by consecutive whole numbers.

    The parameters of the fitted model (center, sigma and weight) for all seven Gaussians are both saved to the GaussDistParams.pkl file (in
    output_dir, for later reuse) and set as the value of the distribution_dict attribute.

    :param cov_sample: a sample of the coverage distribution of the investigated sample, if None, it is loaded from the temporary files of the output_dir (default: None) (array-like)
    :param cov_min: the maximum value of the coverage for a position to be considered in the estimation (default: None) (int)
    :param output_dir: the path to the output directory of the PloidyEstimator object, where temporary files are located. If not None, distribution parameters are saved there as GaussDistParams.pkl. (default: None) (str)
    :param chromosomes: list of chromosomes for the sample (default: None) (array-like)
    :param estimated_hapcov: the estimated value for the haploid coverage, used as prior (float)
    :param level: the level of indentation used in verbose output (default: 0) (int)

    :returns: dictionary containing the fitted parameters of the 7 Gaussians

    """

    def get_samples(coverage_distribution, estimated_haploid_cov, number_of_iterations, burn_period):
        K = 7
        halfwidth_of_uniform = 0.2

        __gc.collect()

        model = __pm.Model()
        with model:
            p = __pm.Dirichlet('p', a=__np.array([1., 1., 1., 1., 1., 1., 1.]), shape=K)
            c1 = __pm.Uniform('c1', (1 - halfwidth_of_uniform) * estimated_haploid_cov,
                              (1 + halfwidth_of_uniform) * estimated_haploid_cov)
            means = __tt.stack([c1, c1 * 2, c1 * 3, c1 * 4, c1 * 5, c1 * 6, c1 * 7])
            order_means_potential = __pm.Potential('order_means_potential',
                                                   __tt.switch(means[1] - means[0] < 0, -__np.inf, 0)
                                                   + __tt.switch(means[2] - means[1] < 0, -__np.inf, 0))
            sds = __pm.Uniform('sds', lower=0, upper=estimated_haploid_cov / 2, shape=K)
            category = __pm.Categorical('category',
                                        p=p,
                                        shape=len(coverage_distribution))
            points = __pm.Normal('obs',
                                 mu=means[category],
                                 sd=sds[category],
                                 observed=coverage_distribution)
        with model:
            step1 = __pm.Metropolis(vars=[p, sds, means])
            step2 = __pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2, 3, 4, 5, 6])

            __logging.getLogger("pymc3").setLevel(__logging.WARNING)

            tr = __pm.sample(draw=number_of_iterations-burn_period, tune=burn_period,
                             step=[step1, step2], progressbar=False, verbose=0, compute_convergence_checks=False)
        # trace = tr[burn_period:]
        # return trace
        return tr

    if cov_sample is None:
        cov_sample = io.get_coverage_distribution(chromosomes=chromosomes,
                                                  output_dir=output_dir,
                                                  cov_max=cov_max,
                                                  cov_min=cov_min)

    iterations2 = 15000
    burn_beginning2 = 10000

    # logger = __logging.getLogger("pymc3")
    # logger.propagate = False

    trace2 = get_samples(coverage_distribution=cov_sample,
                         estimated_haploid_cov=estimated_hapcov,
                         number_of_iterations=iterations2,
                         burn_period=burn_beginning2)

    std_trace = trace2.get_values('sds', chains=[0])
    p_trace = trace2.get_values('p', chains=[0])
    sigma = std_trace.mean(axis=0)
    p = p_trace.mean(axis=0)
    mu = __np.array([trace2.get_values('c1', chains=[0]).mean() * (i + 1) for i in range(7)])

    prior_dict = {'mu': mu, 'sigma': sigma, 'p': p}

    del trace2

    if output_dir:
        io.save_obj(prior_dict, output_dir + '/GaussDistParams')

    return prior_dict


def PE_on_chrom(chrom, output_dir, windowsize_PE, shiftsize_PE, distribution_dict):
    """

    Runs the whole ploidy estimation pipeline on a given chromosome, using the appropriate attributes of the PloidyEstimation object by running
    PE_on_range() multiple times.
    Prints the results to the file: [output_dir]/PE_fullchrom_[chrom].txt.

    :param distribution_dict: a dictionary containing the fitted parameters of the Gaussian mixture model to the coverage distribution. (dict with keys 'mu', 'sigma' and 'p')
    :param shiftsize_PE: shiftsize for moving average over regions (int)
    :param windowsize_PE: windowsize for moving average over regions (int)
    :param output_dir: the path to the directory where temporary files are located (str)
    :param chrom: the name of the chromosome to analyse (str)

    """

    if distribution_dict == None:
        raise ValueError('Error: fitted distribution parameters are not defined, cannot proceed.')

    df = __pd.read_csv(output_dir + '/' + 'PEtmp_fullchrom_' + chrom + '.txt', sep='\t',
                       names=['chrom', 'pos', 'cov', 'mut_freq', 'GC']).sort_values(by='pos')

    df['chrom'] = df['chrom'].apply(str)
    pos_all = __np.array(list(df['pos']))
    total_ploidy_a = __np.array([0] * len(pos_all))
    total_loh_a = __np.array([0] * len(pos_all))
    est_num_a = __np.array([0] * len(pos_all))
    posstart = df['pos'].min()
    posmax = df['pos'].max()
    while (posstart < posmax):
        p, loh = PE_on_range(dataframe=df, rmin=posstart, rmax=posstart + windowsize_PE,
                             all_mu=distribution_dict['mu'],
                             all_sigma=distribution_dict['sigma'],
                             prior=distribution_dict['p'])
        if (p > 0):
            total_ploidy_a += p * (pos_all >= posstart) * (pos_all <= posstart + windowsize_PE)
            total_loh_a += loh * (pos_all >= posstart) * (pos_all <= posstart + windowsize_PE)
            est_num_a += 1 * (pos_all >= posstart) * (pos_all <= posstart + windowsize_PE)
        posstart += shiftsize_PE
    df['total_ploidy'] = total_ploidy_a
    df['total_loh'] = total_loh_a
    df['est_num'] = est_num_a

    df['ploidy'] = (df['total_ploidy'] / df['est_num']).round()
    df['LOH'] = (df['total_loh'] / df['est_num']).round()

    df = df[(~df['ploidy'].isnull()) & (~df['LOH'].isnull())]
    df['ploidy'] = df['ploidy'].astype(int)
    df['LOH'] = df['LOH'].astype(int)

    df[['chrom', 'pos', 'cov', 'mut_freq', 'ploidy', 'LOH']].to_csv(
        output_dir + '/PE_fullchrom_' + chrom + '.txt', sep='\t', index=False)
    # return df[['chrom', 'pos', 'cov', 'mut_freq', 'ploidy', 'LOH', 'total_ploidy', 'est_num']]


def PE_on_range(dataframe, rmin, rmax, all_mu, all_sigma, prior, cov_min=0, cov_max=100000):
    """

    Run the ploidy estimation on a given range of a chromosome.

    :param dataframe: The dataframe read from the temporary files for a given chromosome, containing information about the genomic position, the measured coverage and the frequency of the non-reference bases aligned to the position. (pandas.DataFrame)
    :param rmin: The lower bound of the genomic range considered for the analysis. (int)
    :param rmax: The upper bound of the genomic range considered for the analysis. (int)
    :param all_mu: The centers of the seven Gaussians fitted to the coverage distribution. (list of float)
    :param all_sigma: The sigmas of the seven Gaussians fitted to the coverage distribution. (list of float)
    :param prior: The weights of the seven Gaussians fitted to the coverage distribution. (list of float in the range(0,1))

    :returns: (most_probable_ploidy, most_probable_LOH)

        - most_probable_ploidy: the estimated ploidy for the genomic region (int, in range(1,8))
        - most_probable_LOH: the estimated LOH status for the genomic region (1: LOH, 0: no LOH)

    """

    temp = dataframe[(dataframe['pos'] >= rmin) & (dataframe['pos'] <= rmax)
                     & (dataframe['cov'] >= cov_min) & (dataframe['cov'] <= cov_max)]

    if (temp.shape[0] == 0 or temp['pos'].max() == temp['pos'].min()):
        return 0, 0

    # get cov mean
    cov_mean = temp['cov'].mean()

    posteriors = []
    for ploidy in range(len(all_mu)):
        likelihood = __stats.norm.pdf(cov_mean, loc=all_mu[ploidy], scale=all_sigma[ploidy])
        posteriors.append(likelihood * prior[ploidy])
    posteriors = __np.array(posteriors)
    most_probable_ploidy = __np.argmax(posteriors) + 1

    if most_probable_ploidy == 1 or most_probable_ploidy == 3:
        return most_probable_ploidy, 0

    mf_list = __np.array(temp[temp['mut_freq'] >= 0.5]['mut_freq'])
    if len(mf_list) == 0:
        return most_probable_ploidy, 0

    else:
        check_freqs = {2: [1 / 2, 1],
                       4: [3 / 4, 1 / 2, 1],
                       5: [4 / 5, 3 / 5, 1],
                       6: [5 / 6, 1 / 2, 2 / 3, 1],
                       7: [6 / 7, 5 / 7, 4 / 7, 1]}
        allele_freq_conclusions = []

        tested_all_allele_freqs = check_freqs[most_probable_ploidy]
        for af in tested_all_allele_freqs:
            if (af == 0.5):
                likelihood = __stats.halfnorm.pdf(mf_list, loc=af, scale=2 / cov_mean).prod()
            elif (af == 1):
                likelihood = __stats.halfnorm.pdf(2 - mf_list, loc=af, scale=2 / cov_mean).prod()
            else:
                likelihood = __stats.norm.pdf(mf_list, loc=af, scale=2 / cov_mean).prod()
            allele_freq_conclusions.append(likelihood)
        allele_freq_conclusions = __np.array(allele_freq_conclusions)

        if (__np.argmax(allele_freq_conclusions) == 0):
            most_probable_loh = 0
        else:
            most_probable_loh = 1

    return most_probable_ploidy, most_probable_loh
